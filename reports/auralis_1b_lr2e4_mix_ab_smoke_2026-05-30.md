# Auralis 1B LR 2e-4 Mix A/B Smoke

Date: 2026-05-30

## Purpose

This short A/B test checks whether the next foundation step should be a German-heavy trunk (`92% German / 8% English`) or a balanced bilingual mix. Both runs used the same model, data directory, learning rate, schedule, batch, eval cadence, and bpb conversion constants. Both were stopped after the first eval at step 100 to avoid burning GPU on an undecided strategy.

## Runs

| Run | Mix | Step | Train Loss | Val Loss | German Val | German BPB | English Val | English BPB | BPB Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 German Stem Smoke | DE 0.92 / EN 0.08 | 100 | 8.0887 | 8.339 | 8.379 | 2.826 | 8.607 | 2.471 | 1.144 |
| Balanced LR2e-4 Smoke | EN 0.55 / DE 0.45 | 100 | 7.8379 | 7.198 | 8.606 | 2.903 | 6.154 | 1.767 | 1.643 |

## Interpretation

The German-heavy run made German bpb slightly better at step 100, but at a high cost: English stayed much worse and total validation loss was significantly worse.

The balanced run learned the overall distribution better and improved English strongly, while German lagged. This supports the review concern: a full `92/8` German-first foundation run would be a strategic bet, not a safe default.

## Decision

Do not promote `92% German / 8% English` as the real foundation default.

Use a bilingual foundation run with moderate German priority instead:

- not pure German-first
- not English-dominant
- likely starting point: `55% German / 45% English` or `50/50`
- monitor `bpb_german`, `bpb_english`, and `bpb_gap_max`
- adapt the mix only if bpb proves a sustained imbalance

## Product Intent

The safer product interpretation is:

> German-strong bilingual assistant, not German-only and not English-first.

That means German can be the product priority, but the foundation should still learn both languages together to preserve cross-lingual transfer and avoid later continual-learning shocks.

## Next Recommended Step

Create a real bilingual foundation ramp config with moderate German priority, for example:

```yaml
mix_ratios:
  german: 0.55
  english: 0.45
```

Run it longer than the 100-step smoke, with bpb checkpoints at 100, 250, 500, and 1000. Promote only if:

- loss remains stable
- German bpb improves
- English bpb does not collapse
- bpb gap trends down or stays controlled
- generated samples become less degenerate

SFT, repair, honesty, and code should remain deferred until the foundation text behavior is visibly coherent.
