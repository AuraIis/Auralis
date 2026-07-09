"""Background-producer + pinned double-buffer wrapper around a batch loader.

Hides per-batch data-fetch latency (the random ``np.memmap`` reads that a prior
audit measured at ~15% of wall-clock) behind the previous micro-batch's GPU
compute, and makes the host->device copy genuinely asynchronous by staging into
PINNED host buffers (``non_blocking=True`` from pageable memory is a no-op).
Opt-in via ``data.prefetch: true``; the default training path is unchanged.

Correctness contract (unit-tested on CPU, see tests/training/test_prefetch.py):

* A SINGLE producer thread exclusively owns the wrapped loader's ``__next__``,
  so the loader's numpy RNGs stay single-threaded and the emitted batch stream is
  BYTE-IDENTICAL to the unwrapped loader at the same seed (this is why we do NOT
  use ``torch.utils.data.DataLoader(num_workers>0)`` — forking the stateful PCG64
  generators breaks byte-identical resume).
* ``get_rng_state()`` returns the wrapped loader's RNG snapshot taken *after the
  last CONSUMED batch* (== the state that produces the NEXT unconsumed batch), so
  a checkpoint restores to exactly the next batch. Batches the producer fetched
  ahead but the trainer never consumed are simply regenerated on resume.

GPU note: pinned-slot reuse is fenced with a per-slot CUDA event (the producer
waits on the consumer's post-copy event before overwriting a slot). This host
logic is CPU-testable, but the event fencing and the compute overlap must be
validated on the actual GPU before a long run:
    - compute-sanitizer (racecheck) on ~50 steps  -> proves no slot-reuse race
    - nsys 30s capture -> proves the H2D copy overlaps compute (no host gap
      before each micro's first kernel) and data_frac drops toward ~0
The default is OFF, so an unvalidated build cannot destabilize a real run.
"""
from __future__ import annotations

import queue
import threading

import torch


class _Slot:
    """A reusable pinned host buffer + the CUDA event guarding its reuse."""

    __slots__ = ("pinned", "event")

    def __init__(self, shape, dtype) -> None:
        self.pinned = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.event = torch.cuda.Event()


class PrefetchLoader:
    """Drop-in wrapper around ``MixedDataLoader`` adding background prefetch.

    Returns batches whose ``input_ids`` are already on ``device`` (labels alias
    input_ids — the model only slices labels, never mutates them). Any attribute
    not overridden here delegates to the wrapped loader.
    """

    def __init__(self, loader, device, queue_depth: int = 2) -> None:
        self.loader = loader
        self.device = torch.device(device)
        self._is_cuda = self.device.type == "cuda"
        self.queue_depth = max(1, int(queue_depth))
        self._q: queue.Queue = queue.Queue(maxsize=self.queue_depth)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._exc: BaseException | None = None
        self._started = False
        self._lock = threading.Lock()
        # Slot ring: strictly more slots than the max number of in-flight batches
        # (queue_depth queued + 1 being filled) so the producer never overwrites a
        # slot the consumer has not yet consumed-and-fenced.
        self._n_slots = self.queue_depth + 2
        self._slots: list[_Slot] | None = None
        self._slot_i = 0
        # State that reproduces the NEXT batch to be consumed. Initialised to the
        # loader's construction state so a checkpoint taken before consuming any
        # batch still resumes from batch 0 (not from a prefetched-ahead position).
        self._next_rng = loader.get_rng_state()

    # -- attribute delegation (rows_per_language, close, num_tokens, ...) --------
    def __getattr__(self, name):
        # only reached when normal lookup fails -> forward to the wrapped loader
        return getattr(self.loader, name)

    # -- RNG / resume contract --------------------------------------------------
    def get_rng_state(self) -> dict:
        return self._next_rng

    def set_rng_state(self, state: dict) -> None:
        # Resume: stop the producer, rewind the loader, drop everything prefetched.
        self._shutdown_thread()
        self.loader.set_rng_state(state)
        self._next_rng = state
        self._drain_queue()

    # -- iteration --------------------------------------------------------------
    def __iter__(self):
        self._ensure_started()
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._exc is not None:
            raise self._exc
        item = self._q.get()
        if item is None:  # producer stopped or raised
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        payload, rng_after = item
        # This batch is now consumed; the state that produces the *next* one is
        # what a checkpoint must save.
        self._next_rng = rng_after
        kind, obj = payload
        if kind == "slot":
            ids = obj.pinned.to(self.device, non_blocking=True)
            obj.event.record()  # fence: producer waits on this before reusing the slot
        else:
            ids = obj  # CPU path: already an independent tensor
        return {"input_ids": ids, "labels": ids}

    # -- producer ---------------------------------------------------------------
    def _ensure_started(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._exc = None
            self._thread = threading.Thread(
                target=self._produce, name="prefetch-producer", daemon=True
            )
            self._thread.start()
            self._started = True

    def _produce(self) -> None:
        try:
            while not self._stop.is_set():
                batch = next(self.loader)
                # Snapshot AFTER producing: this is the state that yields the NEXT
                # batch, so the consumer can hand it to the checkpoint verbatim.
                rng_after = self.loader.get_rng_state()
                ids = batch["input_ids"]
                if self._is_cuda:
                    if self._slots is None:
                        self._slots = [
                            _Slot(ids.shape, ids.dtype) for _ in range(self._n_slots)
                        ]
                    slot = self._slots[self._slot_i]
                    self._slot_i = (self._slot_i + 1) % self._n_slots
                    slot.event.synchronize()  # wait out any in-flight H2D from this slot
                    slot.pinned.copy_(ids)
                    payload = ("slot", slot)
                else:
                    payload = ("cpu", ids.clone())
                if not self._put((payload, rng_after)):
                    break  # stop requested while blocked on a full queue
        except Exception as exc:  # propagate to the consumer thread
            self._exc = exc
            self._put_sentinel()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _put_sentinel(self) -> None:
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def _drain_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _shutdown_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._drain_queue()  # unblock a producer parked on a full queue
            self._thread.join(timeout=5.0)
        self._thread = None
        self._started = False

    def close(self) -> None:
        self._shutdown_thread()
        inner_close = getattr(self.loader, "close", None)
        if callable(inner_close):
            inner_close()

    def __del__(self) -> None:
        try:
            self._shutdown_thread()
        except Exception:
            pass


__all__ = ["PrefetchLoader"]
