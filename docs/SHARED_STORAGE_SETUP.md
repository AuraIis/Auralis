# Shared Storage Setup — Auralis v2 Workflow

**Status:** active workflow as of 2026-04-25
**Replaces:** the earlier "edit local, scp/rsync to server" cycle (which
caused the canary-runde2 config-sync incident).

## Idea

The repo lives on a single SMB share on BITBASTION. Both the Windows dev
machine and the Linux training host see the same files. **No sync, no
mirroring, no drift.** Edits in `configs/training/foo.yaml` from Windows
are immediately visible to the trainer in the container.

Training data (`tokenized/`, `cleaned/`, `checkpoints/`, `runs/`) stays
on the **local SSD** of the training host — reading it over SMB
would push training throughput down by 50–100×.

```
┌───────────────────────┐                    ┌───────────────────────┐
│ Windows dev machine   │                    │ Linux training host   │
│   I:\AuralisV2\       │ ←── SMB share ──→  │   /mnt/bitbastion/    │
│                       │   (Code, Configs,   │      auralis/         │
│                       │    Docs, scripts)   │      AuralisV2/       │
└───────────────────────┘                    └─────────┬─────────────┘
                                                       │ bind-mount
                                                       ▼
                                             ┌───────────────────────┐
                                             │ Container             │
                                             │   /workspace/auralis  │ ← Code (SMB)
                                             │   /workspace/v2data   │ ← Data (local SSD)
                                             └───────────────────────┘
```

## One-time setup

### 1. SMB mount on the training host

Add to `/etc/fstab`:

```fstab
//BITBASTION/Auralis  /mnt/bitbastion/auralis  cifs  credentials=/root/.smbcreds,uid=0,gid=0,iocharset=utf8,nounix,vers=3.0,cache=strict,actimeo=10  0  0
```

`/root/.smbcreds` (chmod 600):

```
username=<user>
password=<pass>
domain=WORKGROUP
```

Mount:

```bash
mkdir -p /mnt/bitbastion/auralis
mount -a
ls /mnt/bitbastion/auralis/AuralisV2   # should show the repo contents
```

**Mount options explained:**
- `nounix` — prevents cifs from trying to create Linux symlinks (doesn't work over SMB anyway)
- `cache=strict` — safe caching, prevents stale reads when Windows has just written
- `actimeo=10` — attribute cache 10 seconds, good for code/configs (not for high-frequency IO)

### 2. Start the container with a bind-mount

```bash
docker run -d --name auralis-train --gpus all \
  -v /mnt/bitbastion/auralis/AuralisV2:/workspace/auralis \
  -v /mnt/v2data:/workspace/v2data \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRITON_OVERRIDE_ARCH=sm89 \
  <image>
```

Important:
- `PYTHONDONTWRITEBYTECODE=1` — prevents the container from writing `__pycache__/` to the SMB share (would collide with Windows)
- `TRITON_OVERRIDE_ARCH=sm89` — Blackwell sm_120 Triton workaround (see LESSONS.md L-012)

### 3. Windows side

```cmd
setx PYTHONDONTWRITEBYTECODE 1
```

(open a new terminal so it takes effect). Prevents the `__pycache__` collision from Windows as well.

### 4. Editor configuration

**VS Code** (`.vscode/settings.json` is already in the repo, if not):
```json
{
  "files.eol": "\n"
}
```

Other editors: set to LF. The `.gitattributes` in the repo enforces LF on commit, but your editor should not write CRLF in the first place.

## Pitfalls (KNOWN ISSUES)

### Git operations only from one side

The git repo lives on SMB. If you run `git commit`/`git pull` from Windows **and** Linux in parallel, the repo can get corrupted (lock files, index race).

**Rule:** Git operations exclusively from Windows. On Linux, only **read**, never commits or branches.

### Line endings

`.gitattributes` is configured (`* text=auto eol=lf`). That covers the commit path. In the working tree you have to make sure yourself that your Windows editor writes LF (see above).

Symptom of the CRLF bug on Linux:
```
/usr/bin/env: 'python\r': No such file or directory
```
Fix: convert the file with `dos2unix scripts/foo.py`.

### Symlinks

Don't work over SMB with `nounix`. If you need a symlink in the repo: use a relative path in the config instead, or copy the file.

### `__pycache__` collisions

Solved with `PYTHONDONTWRITEBYTECODE=1` on both sides. If you forget the env var, you'll see `__pycache__/` directories flickering back and forth between Windows and Linux — no data corruption, just unsightly.

### SMB performance

Reading code/configs is OK (~10-30 MB/s is enough for `import` in seconds). **Never** pump training data or checkpoints over SMB — those must stay on the local SSD.

If `python -c "import auralis"` suddenly takes >5s: the SMB cache is probably cold. Running `find /workspace/auralis -name "*.py" > /dev/null` once warms the cache.

### File locking

CIFS has no POSIX locking by default. If you follow along on Linux with `tail -f train.log` while the trainer is writing: usually OK, but can occasionally hang. Workaround: keep log files in `runs/` — those live on the local SSD anyway, not on SMB.

## Verification

After setup, quick smoke test:

```bash
# On Windows:
echo "ping from win" > I:\AuralisV2\.smb_test

# On the Linux training host:
cat /mnt/bitbastion/auralis/AuralisV2/.smb_test
# → "ping from win"

# In the container:
docker exec auralis-train cat /workspace/auralis/.smb_test
# → "ping from win"

# Clean up:
rm I:\AuralisV2\.smb_test
```

If all three see the content immediately: setup is in place.

## What this removes

- `scripts/dev/auto_sync.*` (deleted 2026-04-25, was a workaround for exactly this problem)
- Manual scp calls after config edits
- `docker cp` for config updates
- "why doesn't the new config work" debug sessions

## What is still necessary with this

- Container restart after code changes that are only read at import time
  (Python loads modules once — running training processes don't see code edits
  until restart, that's a Python thing, not SMB-specific)
- Data pipeline outputs (`cleaned/`, `tokenized/`) live on the NAS SSD,
  not on SMB — those still have to be orchestrated separately
- Checkpoints and runs land on the local training SSD — backup/sync
  of those is a different question than "code sync"
