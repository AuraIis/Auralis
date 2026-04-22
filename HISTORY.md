# HISTORY — Auralis v2

Chronologisch, Milestones. Append-Only.

---

## 2026-04-22 — Projekt-Start Auralis v2
- Brief + 7 Phasen-SPECs + `SPEC_DATASETS.md` in `Doc/` finalisiert
- Projekt-Skelett angelegt (`pyproject.toml`, Verzeichnisbaum, `.gitignore`)
- `STATUS.md`, `LESSONS.md`, `HISTORY.md` initialisiert
- Memory-System für Claude Code befüllt (User / Projekt / Feedback / v1-Lessons / v1-Datasets)
- **Entscheidung: Modellgröße 1B** (statt 2-3B) — schneller, günstiger, v1-Kapazität verdoppelt aber bleibt klein.
- **Entscheidung: Daten-Root `//BITBASTION/Auralis/AuralisV2/`** (NAS, 25 TB frei); v1-SFT-Pool `I:/Auralis/NEWGPT/data/` bleibt lokal.
- Phase-0-Vorarbeit komplett: Baseline (50 Fragen), byte-gleicher Prompt-Builder + Tests, Daten-Config + 3 Phase-1-Download-Scripts + v1-Inventory-Script.
- Leitlinie verankert: synthetische Daten-Generierung ist erwünscht, wenn Open-Source-Quellen dünn sind (DeepSeek V3 / Qwen 3.5 30B lokal — v1-erprobt).
