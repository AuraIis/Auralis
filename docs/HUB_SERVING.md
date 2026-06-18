# Helix in the Auralis Hub — Serving Architecture

**Status: in production.** Helix is in the hub (`http://192.168.178.5:3100`) as a selectable model,
in clearly separated modes. No retraining — the problems were solved at the **serving level**.

## Why a shim (no Ollama / llama.cpp)
Helix's hybrid architecture (Mamba-2 / **GLA** / Sparse-Attn) is not supported by llama.cpp
→ no GGUF, no real Ollama model. Instead a **PyTorch server** runs
(`scripts/serving/helix_ollama_server.py`) in the container `auralis-blackwell` on `:11434`, which mimics the
two Ollama endpoints the hub uses:
- `GET /api/tags` → model list (the variants/switches)
- `POST /v1/chat/completions` → OpenAI SSE streaming
The hub points at it via `OLLAMA_BASE_URL=http://auralis-blackwell:11434` (`.env`, backup `.env.bak.helix`);
`auralis-blackwell` is attached to the network `auralis-hub_default`. **No hub code change.**

## Modes (= model variants in the dropdown)
| Variant | Adapter | LoRA-α | for what |
|---|---|---|---|
| **helix-chat** (Default) | code | **0.5** | best casual all-rounder (OFAT sweep winner) |
| helix-corrective | corrective | **0.5** | concise answers |
| helix-corrective-precise | corrective | 1.0 | facts + **honest refusal** |
| helix-corrective-tools | corrective | 1.0 | **math actually computes** |
| helix-corrective-think | corrective | 0.5 | step by step |
| helix-grounded | grounded | 1.0 | answer only from context |
| helix-code | code | 1.0 | Python functions (experimental) |

## IMPORTANT: why corrective runs at α=0.5 (not 1.0)
Measured: `sft_corrective_v3` @**α=1.0 over-dominates** with its abstain/encyclopedia pattern — in open
chat that produces "A dog is a cat", word-mangling (Goblin→"Stob") and abstain spam. At **α=0.5**
these failure modes disappear, while facts ("Paris.") stay clean. The **code adapter** does not have this
abstain dominance → it is the best casual chat (coherent + stops cleanly), hence `helix-chat`.
An adapter is not "better/worse" — **α shifts behavior.**

## Query Rewriter (server-side, no training)
0.9B is phrasing-fragile: `katze` → abstain, but `Was ist eine Katze?` → coherent. The shim
normalizes **bare fragments** (≤3 words, no `?`/`.`/number/question/command word) to `Was ist {X}?`,
before the model sees them. **Acronyms** (gpu, cpu, ram, vram, ai, ki, llm, api, …) are uppercased
→ `gpu` → `Was ist GPU?`. Math, full sentences, grounded & code stay untouched. This takes the
interpretation load off the small model — fixes phrasing where knowledge is present (katze, hardware, gpu);
real knowledge gaps (goblin) stay an honest refusal.

## Tool Execution (`-tools`)
The shim intercepts `<tool:python>…</tool>`, **runs the code in a subprocess** (5 s timeout, sanitizes
arithmetic `x`/`×`→`*`) and injects `<result>`. Math becomes correct as a result (245×4=980, 55432×34=1884688).
There is **no** tool executor in the hub itself — it lives in the shim.

## Routing Gate
`scripts/serving/serving_gate.py` checks operation (no training), most recently **8/8**:
katze (no abstain) · hardware/gpu (explains) · goblin (honestly uncertain) · hallo (normal) ·
Paris (correct) · 5×5 (tool→25) · grounded (1500 extracted).

## Operation
The shim is a **nohup process** (not a service) → gone after container/host restart. Restart:
```
ssh root@BITBASTION 'docker network connect auralis-hub_default auralis-blackwell 2>/dev/null; docker exec -d auralis-blackwell bash -lc "cd /workspace/v2data/diag && nohup python -u helix_ollama_server.py > /workspace/v2data/diag/helix_srv.log 2>&1"'
```
Bring back the real Ollama models (.37):
```
ssh root@BITBASTION 'cd /mnt/user/appdata/auralis-hub && cp .env.bak.helix .env && docker compose up -d --force-recreate --no-deps api'
```

## OFAT Serving Sweep (one knob per test, measured not guessed)
`scripts/serving/sweep_serving.py` — loads the model once, varies ONE knob per line, scores 24
fixed prompts on 8 metrics (stop / len / degen / false_tool / facts / abstain_unknown / topic).
Findings:
- **adapter_alpha is the only big lever.** code @ α=1.0 → facts only 0.75; **α=0.5 → facts 1.0**,
  routing gate 8/8 (α=0.6 would be abstain-unknown 1.0, but breaks greetings → gate 7/8). → `helix-chat` = α=0.5.
- **Decode was already optimal:** rep_penalty 1.05–1.20 = no difference; **greedy** beats sampling
  (temp=0.4 → facts 0.75); max_new 80 cuts off (stop 0.87), ≥120 is enough. → no decode change needed.
Method: take the winner of one stage, then the next stage — never by feel.

## Honest limits
Modes + rewriter + tools + grounded solve **serving/phrasing**. NOT solvable (0.9B ceiling, measured repeatedly):
thin world knowledge (goblin), drift on long answers, fragility. For that **no retraining** (whack-a-mole)
— the right way is a larger base / RAG for the knowledge. See `MILESTONE_GROUNDED_v4.md`, `MILESTONE_CODE_SKILL.md`.
