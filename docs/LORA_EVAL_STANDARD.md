# LoRA Eval Standard (Phase 5 preliminary work)

This document defines which gates every new LoRA-adapter training round
must pass before it is marked "production". Inspiration:
v1 lessons L-002 (LoRA memorization) and the brief §3.3 §6.

## 1. Mandatory artifacts per adapter

For every LoRA merge / every published version, the adapter folder must
contain, mandatorily:

```
adapter_<name>_vNN/
    adapter.safetensors
    adapter_config.yaml            # rank, target_modules, alpha, dropout, method
    facts.yaml                     # only for topic LoRAs (fact table, sources)
    train.jsonl                    # ~1000 samples, 80 facts × paraphrases
    val.jsonl                      # 20 facts DISJOINT from train
    oov_probes.jsonl               # >=20 questions outside the topic
    manifest.yaml                  # training run details (see §4)
    quality_report.md              # scores from the gates below
```

## 2. Mandatory gates (all must pass)

### 2.1 Disjoint train/val split (v1 lesson L-002)

- ``facts.yaml`` facts disjoint between train and val, not just different paraphrases
- Val contains at least 20 facts, at least 3 per sub-category
- Code: `python scripts/lora/check_split_disjoint.py --adapter <dir>`

### 2.2 OOD questions (style bleed / regression)

- at least 20 questions OUTSIDE the topic (math LoRA ⇒ literature questions etc.)
- Metric: `base_oov_score` (base model without adapter) vs. `adapter_oov_score`
- Gate: `adapter_oov_score >= base_oov_score - 0.02` (max 2 percentage points regression)

### 2.3 Base regression on the 50 baseline questions

- We run ``eval/baseline_questions.yaml`` WITH the adapter loaded
- Gate: ≥ 95% of the base score (no general quality loss)

### 2.4 Factual Retention (Core Gate)

- Val facts, 1:1 wording from ``facts.yaml``
- Gate: ≥ 70% correct (at 20 facts = ≥ 14)
- NO train-loss gate below 0.05 (= v1 memorization trap), instead a val-loss plateau at 0.2–0.3

### 2.5 Style Bleed (new)

- "neutral" questions without a topic, the adapter MUST answer like the base (no style drift)
- Automatic: cosine sim of the embeddings of (base_answer, adapter_answer) ≥ 0.9

## 3. Router-LoRA specific gates (from point 16)

### 3.1 Labeled routing data

The router LoRA is tested against a `router_eval.jsonl` with
at least 200 samples, distributed across six classes:

- `level_0_smalltalk`
- `level_1_factual_recall`
- `level_2_tools_needed`
- `level_3_topic_knowledge`
- `level_4_reasoning`
- `level_5_high_risk`

Metric: **per-class accuracy**, not just overall.

### 3.2 Per-class gates

| Class | Min accuracy | Confusion constraint |
|---|--:|---|
| level_0_smalltalk | 95% | must not route to level_5 |
| level_1_factual_recall | 90% | |
| level_2_tools_needed | 80% | |
| level_3_topic_knowledge | 80% | correct topic label ≥ 75% |
| level_4_reasoning | 75% | |
| level_5_high_risk | **95%** | must never route to level_0 — safety |

### 3.3 Safety fence

- level_5_high_risk → level_0 routing: **strict zero tolerance**.
- Any router release with a single such error is a release blocker.

## 4. Mandatory manifest fields

```yaml
adapter:
  name: ...
  version: ...
  method: dora | mora | galore    # MUST for factual vs pattern distinction
  rank: 16 | 32 | 64
  alpha: ...
  target_modules: [q_proj, k_proj, v_proj, ...]
train:
  base_model: checkpoints/phase4_aligned/best.pt
  base_model_sha: ...
  tokenizer_sha: ...
  samples_train: 800-1500
  samples_val: 20+ (disjunct)
  optimizer: {name, lr, betas, weight_decay}
  early_stop: {patience, monitor: val_loss, min_delta}
  final_val_loss: ...
  final_val_factual_acc: ...
results:
  baseline_base_score: 0.82
  baseline_adapter_score: 0.80      # ≥ 0.78 to pass
  oov_base_score: 0.65
  oov_adapter_score: 0.63           # ≥ 0.63 (base - 0.02)
  factual_val_acc: 0.78             # ≥ 0.70
  style_bleed_cos_sim: 0.94         # ≥ 0.90
```

## 5. Procedure before each adapter release

```
1. Train runs with early-stop on val_loss
2. python scripts/lora/gate_eval.py --adapter <dir> --base <ckpt>
   Checks all five gates. Exit code != 0 ⇒ release blocked.
3. On PASS: git tag -a lora/<name>/vNN
4. Reference the adapter manifest automatically in STATUS.md
```

The gate script (`scripts/lora/gate_eval.py`) is implemented in Phase 5;
this document describes WHAT it must do.

## 6. Anti-patterns (v1 experience)

- ❌ Train loss 0.0099 = memorization, not learning
- ❌ Val facts are paraphrases of the train facts
- ❌ Base regression not measured
- ❌ Style drift on neutral questions not measured
- ❌ Router without labeled routing data ("sounds plausible")
- ❌ Release without manifest SHAs (later not traceable)

Each of these points occurred at least once in v1 and cost
work. Gates are the countermeasure.
