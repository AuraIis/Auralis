# SFT Teacher Pipeline — rough draft (after Phase 1)

Status: **rough draft**. Detail + Qwen prompt templates + judge calibration
set will be worked out after Canary round 2 in `INVESTIGATIVE_SYNTH_DATA.md`.

## Guiding idea

Two separate SFT streams, both fed from the same seed passages:

1. **Content-SFT** — teaches domain knowledge  
   Input: question about topic X  
   Output: fact-precise answer, with source anchor when the seed text offered one

2. **Structural-SFT** — teaches **process** instead of fact ("learn to learn")  
   Input: raw content from seed  
   Output: structured decomposition (Q&A list, causal chain, bullet summary,
   confidence-marked sub-questions)

Michael's core formulation: "the model should learn to learn with its existing
data". Stream 2 is exactly that.

## Pipeline phases

```
cleaned/*.filtered.txt
       ↓
[ seed_collector.py ]           <- CPU, now implemented
       ↓
seeds/YYYY-MM-DD/{technical,factual,procedural,opinion,narrative,general}.jsonl
       ↓
[ qwen_synth_sft.py ]           <- after Phase 1; Qwen 3.6 35B Apex via LocalAI
       ↓                           (two output paths per seed)
sft_synth/YYYY-MM-DD/
    content/*.jsonl             ← OSS-Instruct output
    structural/*.jsonl          ← "decompose this text" output
       ↓
[ qwen_evol_instruct.py ]       <- optional, iterative complexity increase
       ↓
sft_synth_evolved/YYYY-MM-DD/...
       ↓
[ qwen_judge.py ]               <- judge-LLM gates
       ↓
sft_curated/YYYY-MM-DD/...      ← final SFT training data
```

## Why combine OSS-Instruct + Evol-Instruct?

- **OSS-Instruct** (WizardCoder / Magicoder): seed document inspires
  realistic user tasks. Prevents the "AI-imitates-benchmark" sound.
- **Evol-Instruct** (WizardLM): simple task → more complex through
  constraints / reasoning depth / breadth. Solves the problem that pure
  OSS-Instruct tasks are mostly too easy.

Combined: OSS gives realism + topical grounding, Evol gives
difficulty levels. Curriculum effect in the training data.

## Judge-LLM gates (mandatory for all samples)

From `DATA_PIPELINE_V2.md` §4:

1. Dedup against existing train samples
2. Dedup within batch (near-dup MinHash)
3. Contradiction to `facts.yaml` → drop (for topic LoRAs later)
4. Length gate (20 ≤ answer_len ≤ 800 chars)
5. Style consistency (cosine-sim vs. reference style)

**Additionally for factual / science-adjacent categories** (from
Mode-B agreement 2026-04-24):

6. Ground-truth anchor mandatory (source cited, judge checks existence)
7. Consensus calibration-set check: if a sample contradicts one of the ~100
   consensus positions → drop or only with peer-reviewed source
8. No "cui bono" narratives in Mode-B topics (ongoing science)

## Two concrete prompt types (detail in INVESTIGATIVE_SYNTH_DATA.md)

### (A) Content-SFT prompt to Qwen

```
SYSTEM: Du bist Lehrer-Assistent. Aus dem folgenden Textausschnitt
erzeuge 3 realistische Nutzerfragen und ihre präzisen Antworten.
Jede Antwort muss aus dem Textausschnitt belegbar sein — keine
Halluzinationen, kein extern hinzugefügtes Wissen.
Format: JSON {frage, antwort, quellen_zitat}.

USER: [SEED-PASSAGE hier]
```

### (B) Structural-SFT prompt to Qwen

```
SYSTEM: Du bist Wissens-Strukturator. Der folgende Rohtext soll in
eine lernbare Form zerlegt werden. Erzeuge:
  1. zentrale_aussage (1 Satz)
  2. unter_aussagen (Liste, je 1 Satz)
  3. nachfragen (Liste, je ein Q&A-Paar)
  4. offene_fragen (was der Text NICHT beantwortet)
  5. confidence_pro_aussage (high / medium / low / uncertain)
Kein externer Kontext. Nur das was der Text hergibt.

USER: [SEED-PASSAGE hier]
```

## Important non-goals

- **No** endless crawl. The seed numbers per source are cappable;
  mass without quality is worthless.
- **No** synthetic generation *while* Phase 1 is running —
  a GPU conflict with canary runs + pretraining would be fatal.
- **No** separate "creative generator" mode. We stay with seed →
  transformation. Purely generative (Qwen invents topic + text) without a seed
  tends toward benchmark sound and style collapse.

## Next concrete steps

Now:
- [x] `scripts/data/seed_collector.py` exists
- [x] `configs/data/seed_collection.yaml` exists
- [ ] Run the collector in the container (CPU-only, parallel to the pipeline)

After Canary round 2 is done:
- [ ] `INVESTIGATIVE_SYNTH_DATA.md` with full prompts (Mode A / Mode B)
- [ ] `eval/scientific_consensus.yaml` with 100 calibration questions
- [ ] `scripts/data/qwen_client.py` — OpenAI-compatible LocalAI wrapper
- [ ] `scripts/data/qwen_synth_sft.py`
- [ ] `scripts/data/qwen_judge.py`

After Phase 1 pretraining:
- [ ] First full SFT generation round on ~300k seeds
- [ ] QC statistics, sample review, judge calibration
- [ ] Phase 3 SFT training with TRL on the curated samples

## §7. Creative writing — songs / texts (four pillars, no lyrics reproduction)

Goal: the model can independently write texts (songs, poems, prose) and
understands **why** certain texts move people. The latter piece is the more
important one — structure alone does not make a song work, the
**emotional resonance** does.

Four pillars fed in parallel:

| Pillar | Data source | What is learned |
|---|---|---|
| **A — theory** | songwriting books, blogs, music-theory papers (S2ORC subset) | craft: form, rhyme, meter, hook principles |
| **B — public-domain texts** | Gutenberg folk songs, hymns before 1925, classical poems (Goethe, Heine, Shakespeare) | what concrete texts look like in form |
| **C — Qwen generation** | teacher produces original texts following explicit structural specs | application + variation |
| **D — reception discourse** | music reviews, Genius annotations (analytical text, NOT lyrics), songwriter interviews, music-psychology papers, Reddit r/Music discussions | **why** people like songs |

**Copyright rule, non-negotiable**: no training on copyright-protected song
lyrics, not even as an AI paraphrase. A paraphrase is legally an
adaptation (UrhG § 23) and requires consent. Pillar C
(Qwen generates entirely new) delivers the "has-a-text" ability; pillar
D (discourse about songs) delivers the "knows-what-moves" ability. Together
they reach the goal without license risk.

**Emotional-resonance patterns** (subcategory of pillar D):
- Specificity → universality ("your black cat slept on the
  pillow" instead of abstract "grief") — concrete images trigger shared
  memories
- Anchor to the big six: love, loss, longing, defiance, contempt,
  connectedness
- Truth > conformity — why Cash's "Hurt" cover hits harder than
  the original
- Context sensitivity: which style fits which emotion
  (battle rap ≠ lullaby, both valid in their context)

Reception license matrix (pillar D):

| Source | License | Status |
|---|---|:-:|
| Genius annotations (via API) | CC-BY-SA (analytical text) | ✓ |
| Music-psychology papers via OpenAlex | Open Access / CC variants | ✓ |
| Pitchfork / Rolling Stone / laut.de reviews | press-quotation right, individual quotes OK, mass crawl ToS-tricky | ⚠ curate manually |
| Reddit r/Music, r/hiphopheads etc. | older Pushshift dumps on HF, new API closed | ⚠ snapshot-based |
| Songwriter interviews (NPR Tiny Desk, Song Exploder) | press quotation, transcripts partly free | ✓ individual episodes |
| Songfacts / SongMeanings | user content, individual quote OK | ⚠ |

Download path in the pipeline: W1.3+W1.4 today (theory + public domain),
W1.5 only after individual license check (reception sources).

## §8. Troubleshooting / problem-solving (Windows, hardware, phone, network)

Goal: the model can diagnose in a structured way. The same meta-pattern as
code-write+test from §1 — only at the system/hardware level.

**Training pattern** (every SFT sample has this structure):

```
1. Problem description (often vague / emotional: "my pc is slow")
2. Clarification round (OS version? What was already tried? Since when?)
3. Diagnosis chain (systematic: cable → driver → config → software → hardware)
4. Solution (concrete steps, one after another)
5. Verification (how do you know it works now?)
6. Escalation (if not: what next? Who else can help?)
```

The model learns: **clarify first, then solve**. Important difference from
"reproduce a Google search result" — real users describe problems imprecisely;
the AI must ask targeted questions before it speculates.

**Data sources (W1.6, currently running):**

- `HuggingFaceH4/stack-exchange-preferences` — Q&A with upvote-based
  preferences. Contains Stack Overflow + superuser + serverfault +
  askubuntu + apple + android + electronics + networkengineering.
  CC-BY-SA. Perfect for Phase 4 ORPO preference pairs additionally.

**Later (W2):**
- Full per-site dumps per Stack-Exchange subdomain, where we can filter
  precisely for "superuser", "serverfault", "askubuntu"
- Arch Wiki, Ubuntu Wiki, Gentoo Wiki as the Linux troubleshooting core
- Subset from `bigcode/the-stack-github-issues` for bug-report → fix-PR pairs

**Later (W3, Qwen stage after Phase 1):**
- **German translation + adaptation** of the English Stack-Exchange seeds
  by Qwen. 90 % of the troubleshooting world is English, but a
  German user searches in German — we close the gap synthetically.
- Evol-Instruct on problem descriptions: simple ("mouse doesn't work")
  → complex ("mouse does weird things with a USB hub plus a printer + other
  devices on Win11 24H2 after the last update")
- **Emotion calibration**: many troubleshooting requests come in annoyed
  ("I've been trying for 3 hours..."). The model should: acknowledge briefly
  (not over the top), then help in a structured way. The Stack-Exchange tone is
  the reference: terse, empathetic, factual.
