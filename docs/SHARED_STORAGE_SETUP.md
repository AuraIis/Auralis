# Shared Storage Setup — Auralis v2 Workflow

**Status:** active workflow as of 2026-04-25
**Replaces:** the earlier "edit local, scp/rsync to server" cycle (which
caused the canary-runde2 config-sync incident).

## Idea

The repo lives on a single SMB share on BITBASTION. Both the Windows dev
machine and the Linux training host see the same files. **No sync, no
mirroring, no drift.** Edits in `configs/training/foo.yaml` from Windows
are immediately visible to the trainer in the container.

Trainings-Daten (`tokenized/`, `cleaned/`, `checkpoints/`, `runs/`) bleiben
auf der **lokalen SSD** des Trainings-Hosts — sie über SMB zu lesen
würde die Trainings-Throughput um 50–100× drücken.

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
                                             │   /workspace/v2data   │ ← Daten (lokale SSD)
                                             └───────────────────────┘
```

## One-time setup

### 1. SMB-Mount auf dem Trainings-Host

`/etc/fstab` ergänzen:

```fstab
//BITBASTION/Auralis  /mnt/bitbastion/auralis  cifs  credentials=/root/.smbcreds,uid=0,gid=0,iocharset=utf8,nounix,vers=3.0,cache=strict,actimeo=10  0  0
```

`/root/.smbcreds` (chmod 600):

```
username=<user>
password=<pass>
domain=WORKGROUP
```

Mounten:

```bash
mkdir -p /mnt/bitbastion/auralis
mount -a
ls /mnt/bitbastion/auralis/AuralisV2   # sollte den Repo-Inhalt zeigen
```

**Mount-Optionen erklärt:**
- `nounix` — verhindert dass cifs versucht Linux-Symlinks zu erzeugen (geht über SMB sowieso nicht)
- `cache=strict` — sicheres Caching, verhindert Stale-Reads wenn Windows gerade geschrieben hat
- `actimeo=10` — Attribut-Cache 10 Sekunden, gut für Code/Configs (nicht für hochfrequente IO)

### 2. Container starten mit Bind-Mount

```bash
docker run -d --name auralis-train --gpus all \
  -v /mnt/bitbastion/auralis/AuralisV2:/workspace/auralis \
  -v /mnt/v2data:/workspace/v2data \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRITON_OVERRIDE_ARCH=sm89 \
  <image>
```

Wichtig:
- `PYTHONDONTWRITEBYTECODE=1` — verhindert dass der Container `__pycache__/` auf den SMB-Share schreibt (würde mit Windows kollidieren)
- `TRITON_OVERRIDE_ARCH=sm89` — Blackwell sm_120 Triton-Workaround (siehe LESSONS.md L-012)

### 3. Windows-Seite

```cmd
setx PYTHONDONTWRITEBYTECODE 1
```

(neues Terminal öffnen, damit es greift). Verhindert die `__pycache__`-Kollision auch von Windows aus.

### 4. Editor-Konfiguration

**VS Code** (`.vscode/settings.json` ist schon im Repo, falls nicht):
```json
{
  "files.eol": "\n"
}
```

Andere Editoren: auf LF stellen. Die `.gitattributes` im Repo erzwingt LF beim Commit, aber dein Editor sollte gar nicht erst CRLF schreiben.

## Stolpersteine (KNOWN ISSUES)

### Git-Operationen nur von einer Seite

Git-Repo liegt auf SMB. Wenn du von Windows **und** Linux parallel `git commit`/`git pull` machst, kann das Repo korrupt werden (Lock-Files, Index-Race).

**Regel:** Git-Operationen ausschließlich von Windows. Auf Linux nur **lesen**, nie commits oder branches.

### Zeilenenden

`.gitattributes` ist konfiguriert (`* text=auto eol=lf`). Das deckt den Commit-Pfad ab. Im Working-Tree musst du selbst aufpassen dass dein Windows-Editor LF schreibt (siehe oben).

Symptom bei CRLF-Bug auf Linux:
```
/usr/bin/env: 'python\r': No such file or directory
```
Fix: Datei mit `dos2unix scripts/foo.py` konvertieren.

### Symlinks

Gehen über SMB mit `nounix` nicht. Wenn du einen Symlink im Repo brauchst: nutz stattdessen einen relativen Pfad in der Config oder kopier die Datei.

### `__pycache__` Kollisionen

Mit `PYTHONDONTWRITEBYTECODE=1` auf beiden Seiten gelöst. Falls du das Env-Var vergisst, siehst du `__pycache__/` Verzeichnisse die zwischen Windows und Linux hin und her flackern — keine Datenkorruption, nur unschön.

### SMB-Performance

Code/Configs lesen ist OK (~10-30 MB/s reicht für `import` in Sekunden). **Niemals** Trainings-Daten oder Checkpoints über SMB pumpen — die müssen auf lokaler SSD bleiben.

Wenn `python -c "import auralis"` plötzlich >5s braucht: wahrscheinlich SMB-Cache cold. Einmal `find /workspace/auralis -name "*.py" > /dev/null` warmt den Cache auf.

### File-Locking

CIFS hat kein POSIX-Locking by default. Wenn du auf Linux mit `tail -f train.log` mitliest während der Trainer schreibt: meist OK, kann aber gelegentlich hängen. Workaround: log-Files in `runs/` halten — die liegen sowieso auf der lokalen SSD, nicht auf SMB.

## Verifikation

Nach dem Setup, schneller Smoke-Test:

```bash
# Auf Windows:
echo "ping from win" > I:\AuralisV2\.smb_test

# Auf Linux Trainings-Host:
cat /mnt/bitbastion/auralis/AuralisV2/.smb_test
# → "ping from win"

# Im Container:
docker exec auralis-train cat /workspace/auralis/.smb_test
# → "ping from win"

# Aufräumen:
rm I:\AuralisV2\.smb_test
```

Wenn alle drei den Inhalt sofort sehen: Setup steht.

## Was damit weg ist

- `scripts/dev/auto_sync.*` (gelöscht 2026-04-25, war Workaround für genau dieses Problem)
- Manuelle scp-Aufrufe nach Config-Edits
- `docker cp` für Konfig-Updates
- "warum funktioniert die neue Config nicht" Debug-Sessions

## Was damit weiterhin nötig bleibt

- Container-Restart nach Code-Änderungen die nur beim Import gelesen werden
  (Python lädt Module einmal — laufende Trainings-Prozesse sehen Code-Edits
  erst nach Restart, das ist ein Python-Sache, nicht SMB-spezifisch)
- Daten-Pipeline-Outputs (`cleaned/`, `tokenized/`) liegen auf der NAS-SSD,
  nicht auf SMB — die müssen weiterhin separat orchestriert werden
- Checkpoints und Runs landen auf der lokalen Trainings-SSD — Backup/Sync
  davon ist eine andere Frage als "Code-Sync"
