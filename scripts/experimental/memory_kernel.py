"""Prototype chat-memory extraction for future User-DoRA experiments.

This is deliberately isolated under ``scripts/experimental``. It does not train
an adapter yet; it turns chat logs into structured memories plus tiny SFT-style
pairs that can later feed a LoRA/DoRA job.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


MEMORY_VERSION = "auralis-memory-v0.1"

PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:\\[^\s\"']+)|(?:/mnt/[^\s\"']+)|(?:\\\\[^\s\"']+))"
)
HARDWARE_RE = re.compile(
    r"\b("
    r"RTX|NVIDIA|AMD|Ryzen|5950X|3090|5090|GPU|VRAM|Unraid|BITBASTION|"
    r"B550|AORUS|NVMe|SSD|HBA|LSI|SAS"
    r")\b",
    re.IGNORECASE,
)
PREFERENCE_RE = re.compile(
    r"\b("
    r"ich will|ich möchte|ich moechte|ich brauche|mir ist wichtig|"
    r"bevorzug|soll|sollte|immer|nicht|vermeiden|lieber|am besten"
    r")\b",
    re.IGNORECASE,
)
TEMPORARY_RE = re.compile(
    r"\b("
    r"gerade|grade|läuft|laeuft|status|fertig|heute|morgen|jetzt|"
    r"aktuell|container|download|clean|checkpoint|step_\d+"
    r")\b",
    re.IGNORECASE,
)
PROJECT_RE = re.compile(r"\b(Auralis|Helix|Training|Dataset|Tokenizer|DoRA|LoRA|RAG)\b")


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class Memory:
    id: str
    type: str
    scope: str
    content: str
    importance: float
    confidence: float
    source: dict[str, str] = field(default_factory=dict)
    expires: str | None = None
    train_into_adapter: bool = False
    tags: list[str] = field(default_factory=list)
    version: str = MEMORY_VERSION


def load_chat(path: Path) -> list[ChatMessage]:
    """Load a simple chat JSON file.

    Accepted formats:
    - ``[{"role": "user", "content": "..."}]``
    - ``{"messages": [{"role": "user", "content": "..."}]}``
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["messages"] if isinstance(raw, dict) and "messages" in raw else raw
    if not isinstance(rows, list):
        raise ValueError("Chat JSON must be a list or an object with a messages list.")

    messages: list[ChatMessage] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Message {i} is not an object.")
        role = str(row.get("role", "user")).strip() or "user"
        content = str(row.get("content", "")).strip()
        timestamp = row.get("timestamp")
        if content:
            messages.append(ChatMessage(role=role, content=content, timestamp=timestamp))
    return messages


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_text(text: str, max_chars: int = 420) -> str:
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0]
    return f"{cut}…"


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class MemoryExtractor:
    """Small deterministic extractor for first-pass testing.

    Later this can be swapped for an LLM extractor, while keeping the same
    output schema and tests.
    """

    def __init__(self, chat_id: str = "chat") -> None:
        self.chat_id = chat_id

    def extract(self, messages: list[ChatMessage]) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[str] = set()
        for idx, msg in enumerate(messages, start=1):
            if msg.role not in {"user", "assistant", "system"}:
                continue
            content = compact_text(msg.content)
            candidates = self._extract_from_message(idx, msg, content)
            for memory in candidates:
                key = f"{memory.type}:{memory.scope}:{normalize_key(memory.content)}"
                if key in seen:
                    continue
                seen.add(key)
                memories.append(memory)
        return memories

    def _extract_from_message(
        self, idx: int, msg: ChatMessage, content: str
    ) -> list[Memory]:
        source = {
            "chat_id": self.chat_id,
            "message_index": str(idx),
            "role": msg.role,
        }
        if msg.timestamp:
            source["timestamp"] = msg.timestamp

        tags = self._tags_for(content)
        out: list[Memory] = []
        is_temp = bool(TEMPORARY_RE.search(content))
        has_path = bool(PATH_RE.search(content))
        has_hardware = bool(HARDWARE_RE.search(content))
        has_preference = msg.role == "user" and bool(PREFERENCE_RE.search(content))
        has_project = bool(PROJECT_RE.search(content))
        is_memory_policy = (
            "tempor" in content.lower()
            and "status" in content.lower()
            and ("dora" in content.lower() or "lora" in content.lower())
        )
        is_temp = is_temp and not is_memory_policy

        if has_preference:
            out.append(
                self._memory(
                    idx,
                    "user_preference",
                    "global" if not has_project else "auralis",
                    content,
                    0.86,
                    0.78,
                    source,
                    tags,
                    train=not is_temp,
                )
            )

        for match in PATH_RE.finditer(content):
            path = match.group("path").rstrip(".,;)")
            out.append(
                self._memory(
                    idx,
                    "path_fact",
                    "auralis" if has_project else "system",
                    f"Relevanter Pfad: {path}",
                    0.82,
                    0.9,
                    source,
                    sorted(set(tags + ["path"])),
                    train=False,
                )
            )

        if has_hardware:
            out.append(
                self._memory(
                    idx,
                    "hardware_fact",
                    "system",
                    content,
                    0.78,
                    0.76,
                    source,
                    sorted(set(tags + ["hardware"])),
                    train=not is_temp,
                )
            )

        if has_project and not has_preference and not has_hardware and not is_temp:
            out.append(
                self._memory(
                    idx,
                    "project_fact",
                    "auralis",
                    content,
                    0.72,
                    0.68,
                    source,
                    tags,
                    train=True,
                )
            )

        if is_temp:
            out.append(
                self._memory(
                    idx,
                    "temporary_status",
                    "auralis" if has_project else "session",
                    content,
                    0.45,
                    0.72,
                    source,
                    sorted(set(tags + ["temporary"])),
                    train=False,
                    expires=(datetime.now(UTC) + timedelta(days=2)).isoformat(),
                )
            )

        return out

    def _memory(
        self,
        idx: int,
        memory_type: str,
        scope: str,
        content: str,
        importance: float,
        confidence: float,
        source: dict[str, str],
        tags: list[str],
        train: bool,
        expires: str | None = None,
    ) -> Memory:
        return Memory(
            id=f"{self.chat_id}_{idx:04d}_{memory_type}",
            type=memory_type,
            scope=scope,
            content=content,
            importance=importance,
            confidence=confidence,
            source=source,
            expires=expires,
            train_into_adapter=train,
            tags=tags,
        )

    @staticmethod
    def _tags_for(content: str) -> list[str]:
        tags: set[str] = set()
        if "Auralis" in content or "Helix" in content:
            tags.add("auralis")
        if "DoRA" in content or "LoRA" in content:
            tags.add("adapter")
        if "RAG" in content:
            tags.add("rag")
        if "Training" in content or "train" in content.lower():
            tags.add("training")
        if "Dataset" in content or "daten" in content.lower():
            tags.add("data")
        return sorted(tags)


def build_adapter_pairs(memories: list[Memory]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for memory in memories:
        if not memory.train_into_adapter:
            continue
        if memory.type == "user_preference":
            instruction = "Welche dauerhafte Nutzerpräferenz soll Auralis beachten?"
        elif memory.type == "hardware_fact":
            instruction = "Welche stabile Information zum lokalen System soll Auralis kennen?"
        elif memory.type == "project_fact":
            instruction = "Welche stabile Projektinformation soll Auralis kennen?"
        else:
            instruction = "Welche dauerhafte Memory soll Auralis beachten?"
        pairs.append(
            {
                "instruction": instruction,
                "input": f"Memory-Typ: {memory.type}; Scope: {memory.scope}",
                "output": memory.content,
            }
        )
    return pairs


def build_kernel_corpus(memories: list[Memory]) -> list[str]:
    blocks: list[str] = []
    for memory in memories:
        if not memory.train_into_adapter:
            continue
        tag = {
            "user_preference": "preference",
            "project_fact": "fact",
            "hardware_fact": "fact",
        }.get(memory.type, "memory")
        blocks.append(
            "\n".join(
                [
                    f"<|{tag}|>",
                    f"Typ: {memory.type}",
                    f"Scope: {memory.scope}",
                    f"Inhalt: {memory.content}",
                    f"Wichtigkeit: {memory.importance:.2f}",
                    "<|end|>",
                ]
            )
        )
    return blocks


def write_report(path: Path, memories: list[Memory], pairs: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for memory in memories:
        counts[memory.type] = counts.get(memory.type, 0) + 1
    lines = [
        "# Auralis Memory Kernel Smoke Report",
        "",
        f"- Memories: {len(memories)}",
        f"- Adapter train pairs: {len(pairs)}",
        f"- Trainable memories: {sum(1 for m in memories if m.train_into_adapter)}",
        f"- Non-trainable/temporary/path memories: {sum(1 for m in memories if not m.train_into_adapter)}",
        "",
        "## Types",
        "",
    ]
    lines.extend(f"- {k}: {v}" for k, v in sorted(counts.items()))
    lines.extend(["", "## Sample Memories", ""])
    for memory in memories[:8]:
        lines.append(
            f"- `{memory.type}` train={memory.train_into_adapter}: {memory.content}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_extract(args: argparse.Namespace) -> None:
    messages = load_chat(args.input)
    extractor = MemoryExtractor(chat_id=args.chat_id)
    memories = extractor.extract(messages)
    pairs = build_adapter_pairs(memories)
    kernel_blocks = build_kernel_corpus(memories)

    write_jsonl(args.memories, [asdict(m) for m in memories])
    write_jsonl(args.adapter_jsonl, pairs)
    args.kernel_txt.parent.mkdir(parents=True, exist_ok=True)
    args.kernel_txt.write_text("\n\n".join(kernel_blocks) + "\n", encoding="utf-8")
    write_report(args.report, memories, pairs)

    print(
        f"wrote {len(memories)} memories, {len(pairs)} adapter pairs, "
        f"{len(kernel_blocks)} kernel blocks"
    )


def run_smoke(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ich möchte, dass Auralis auf Deutsch direkt und praktisch antwortet. "
                    "Große Trainingsdaten sollen bevorzugt auf /mnt/disk5 liegen."
                ),
                "timestamp": "2026-05-13T12:00:00Z",
            },
            {
                "role": "assistant",
                "content": (
                    "Status: clean-v3.1 läuft gerade noch an german_legacy und der "
                    "Download ist fertig."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Mein Unraid Server nutzt einen Ryzen 9 5950X, eine RTX PRO 5000 "
                    "mit 48GB VRAM und ein B550 AORUS ELITE V2."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Für User-Chats will ich tägliche Memory-Zusammenfassungen in "
                    "LoRA oder DoRA trainieren, aber temporären Status nicht dauerhaft."
                ),
            },
        ]
    }
    input_path = out / "sample_chat.json"
    input_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    run_extract(
        argparse.Namespace(
            input=input_path,
            chat_id="smoke_chat",
            memories=out / "memories.jsonl",
            adapter_jsonl=out / "adapter_train.jsonl",
            kernel_txt=out / "kernel_memory.txt",
            report=out / "report.md",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract", help="extract memories from chat JSON")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--chat-id", default="chat")
    extract.add_argument("--memories", type=Path, required=True)
    extract.add_argument("--adapter-jsonl", type=Path, required=True)
    extract.add_argument("--kernel-txt", type=Path, required=True)
    extract.add_argument("--report", type=Path, required=True)
    extract.set_defaults(func=run_extract)

    smoke = sub.add_parser("smoke", help="run a local example extraction")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.set_defaults(func=run_smoke)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
