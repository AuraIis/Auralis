# Helix im Auralis Hub — Serving-Architektur

**Status: produktiv.** Helix ist im Hub (`http://192.168.178.5:3100`) als auswählbares Modell,
in klar getrennten Modi. Kein Nachtraining — die Probleme wurden auf der **Serving-Ebene** gelöst.

## Warum ein Shim (kein Ollama / llama.cpp)
Helix' Hybrid-Architektur (Mamba-2 / **GLA** / Sparse-Attn) wird von llama.cpp nicht unterstützt
→ kein GGUF, kein echtes Ollama-Modell. Stattdessen läuft ein **PyTorch-Server**
(`scripts/serving/helix_ollama_server.py`) im Container `auralis-blackwell` auf `:11434`, der die
zwei Ollama-Endpoints nachäfft, die der Hub nutzt:
- `GET /api/tags` → Modell-Liste (die Varianten/Schalter)
- `POST /v1/chat/completions` → OpenAI-SSE-Streaming
Der Hub zeigt per `OLLAMA_BASE_URL=http://auralis-blackwell:11434` darauf (`.env`, Backup `.env.bak.helix`);
`auralis-blackwell` ist ans Netz `auralis-hub_default` angeschlossen. **Keine Hub-Code-Änderung.**

## Modi (= Modell-Varianten im Dropdown)
| Variante | Adapter | LoRA-α | wofür |
|---|---|---|---|
| **helix-chat** (Default) | code | **0.5** | bester Casual-Allrounder (OFAT-Sweep-Gewinner) |
| helix-corrective | corrective | **0.5** | knappe Antworten |
| helix-corrective-precise | corrective | 1.0 | Fakten + **ehrliche Absage** |
| helix-corrective-tools | corrective | 1.0 | **Mathe rechnet echt** |
| helix-corrective-think | corrective | 0.5 | Schritt-für-Schritt |
| helix-grounded | grounded | 1.0 | nur aus Kontext antworten |
| helix-code | code | 1.0 | Python-Funktionen (experimentell) |

## WICHTIG: warum corrective auf α=0.5 läuft (nicht 1.0)
Gemessen: `sft_corrective_v3` @**α=1.0 über-dominiert** sein Abstain-/Enzyklopädie-Muster — bei offenem
Chat erzeugt das „Ein Hund ist eine Katze", Wort-Verhunzen (Goblin→„Stob") und Abstain-Spam. Bei **α=0.5**
verschwinden diese Fehler-Modi, während Fakten („Paris.") sauber bleiben. Der **Code-Adapter** hat diese
Abstain-Dominanz nicht → er ist der beste Casual-Chat (kohärent + stoppt sauber), daher `helix-chat`.
Ein Adapter ist nicht „besser/schlechter" — **α verschiebt Verhalten.**

## Query-Rewriter (serverseitig, kein Training)
0.9B ist phrasierungs-fragil: `katze` → Abstain, aber `Was ist eine Katze?` → kohärent. Der Shim
normalisiert **nackte Fragmente** (≤3 Wörter, kein `?`/`.`/Zahl/Frage-/Befehlswort) zu `Was ist {X}?`,
bevor das Modell sie sieht. **Akronyme** (gpu, cpu, ram, vram, ai, ki, llm, api, …) werden großgeschrieben
→ `gpu` → `Was ist GPU?`. Mathe, ganze Sätze, grounded & code bleiben unangetastet. Das nimmt dem kleinen
Modell die Interpretationslast — fixt Phrasierung dort, wo Wissen da ist (katze, hardware, gpu);
echte Wissenslücken (goblin) bleiben ehrlich Absage.

## Tool-Ausführung (`-tools`)
Der Shim fängt `<tool:python>…</tool>` ab, **führt den Code per Subprozess aus** (5 s Timeout, sanitisiert
arithmetisches `x`/`×`→`*`) und injiziert `<result>`. Mathe wird dadurch korrekt (245×4=980, 55432×34=1884688).
Es gibt **keinen** Tool-Executor im Hub selbst — der lebt im Shim.

## Routing-Gate
`scripts/serving/serving_gate.py` prüft den Betrieb (kein Training), zuletzt **8/8**:
katze (kein Abstain) · hardware/gpu (erklärt) · goblin (ehrlich unsicher) · hallo (normal) ·
Paris (korrekt) · 5×5 (Tool→25) · grounded (1500 extrahiert).

## Betrieb
Shim ist ein **nohup-Prozess** (kein Service) → nach Container-/Host-Neustart weg. Neustart:
```
ssh root@BITBASTION 'docker network connect auralis-hub_default auralis-blackwell 2>/dev/null; docker exec -d auralis-blackwell bash -lc "cd /workspace/v2data/diag && nohup python -u helix_ollama_server.py > /workspace/v2data/diag/helix_srv.log 2>&1"'
```
Echte Ollama-Modelle (.37) zurückholen:
```
ssh root@BITBASTION 'cd /mnt/user/appdata/auralis-hub && cp .env.bak.helix .env && docker compose up -d --force-recreate --no-deps api'
```

## OFAT-Serving-Sweep (ein Knopf pro Test, gemessen statt geraten)
`scripts/serving/sweep_serving.py` — lädt das Modell einmal, variiert EINEN Knopf pro Zeile, scort 24
feste Prompts auf 8 Metriken (stop / len / degen / false_tool / facts / abstain_unknown / topic).
Befunde:
- **adapter_alpha ist der einzige große Hebel.** code @ α=1.0 → Fakten nur 0.75; **α=0.5 → Fakten 1.0**,
  Routing-Gate 8/8 (α=0.6 wäre Abstain-Unknown 1.0, bricht aber Grüße → Gate 7/8). → `helix-chat` = α=0.5.
- **Decode war schon optimal:** rep_penalty 1.05–1.20 = kein Unterschied; **greedy** schlägt Sampling
  (temp=0.4 → Fakten 0.75); max_new 80 schneidet ab (stop 0.87), ≥120 reicht. → keine Decode-Änderung nötig.
Methode: Gewinner einer Stufe übernehmen, dann nächste Stufe — nie nach Gefühl.

## Ehrliche Grenzen
Modi + Rewriter + Tools + Grounded lösen **Serving/Phrasierung**. NICHT lösbar (0.9B-Decke, mehrfach vermessen):
dünnes Weltwissen (goblin), Drift bei langen Antworten, Fragilität. Dafür **kein Nachtraining** (Whack-a-Mole)
— der richtige Weg ist größeres Base / RAG fürs Wissen. Siehe `MILESTONE_GROUNDED_v4.md`, `MILESTONE_CODE_SKILL.md`.
