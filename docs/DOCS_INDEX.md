# Auralis Docs Index

Dieser Index trennt aktuelle Arbeitsdoku, Projektidee, historische Specs und
Experimente. Wenn sich Dateien widersprechen, gilt zuerst `STATUS.md`, dann
dieser Index, dann die jeweilige aktuelle Arbeitsdoku.

## Current Truth

- [../STATUS.md](../STATUS.md) - aktueller Stand, aktive Richtung, aktuelle
  Runs und offene Aufgaben.
- [../README.md](../README.md) - Einstieg und Link-Hub.
- [../eval/README.md](../eval/README.md) - wie Checkpoints bewertet werden.

## Core Idea And Model Specs

- [../Doc/AURALIS_V2_PROJECT_BRIEF.md](../Doc/AURALIS_V2_PROJECT_BRIEF.md) -
  die grosse Idee: kleines Basismodell, modulare Adapter, Tools, Router,
  Memory/LoRA-System, Trainingsphasen.
- [../Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md](../Doc/SPECs/SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md) -
  technische Helix-v2-Architektur: hybrider Stack aus Mamba/SSM, GLA und
  Sparse Attention.
- [../tokenizer/quality_report.md](../tokenizer/quality_report.md) -
  Tokenizer-Effizienz und Roundtrip-Qualitaet.

## Data And Pipeline

- [data_cleaning_pipeline_v3.md](data_cleaning_pipeline_v3.md) - aktueller
  Cleaning-Ansatz fuer bessere Pretraining-Daten.
- [dataset_market_app.md](dataset_market_app.md) - Dataset-Suche, Bewertung
  und Mix-Planung.
- [DATA_PIPELINE_V2.md](DATA_PIPELINE_V2.md) - aelterer Pipeline-Stand,
  weiter als Referenz nuetzlich.
- [../data/eval/pretrain_clean_v2_audit_v3.md](../data/eval/pretrain_clean_v2_audit_v3.md) -
  wichtiges Daten-Audit aus der Rettungsphase.
- [../data/eval/training_data_cleaning_report.md](../data/eval/training_data_cleaning_report.md) -
  Cleaning-Report.

## Experiments

- [experimental/knowledge_dna_v2.md](experimental/knowledge_dna_v2.md) -
  Knowledge-DNA mit `<memory>`/`<recall>` als Booster-Idee.
- [experimental/knowledge_kernel.md](experimental/knowledge_kernel.md) -
  separater Knowledge-Kernel-Test.
- [experimental/memory_kernel.md](experimental/memory_kernel.md) -
  Memory-Kernel-Prototyp.
- [experimental/math_reasoning_dna.md](experimental/math_reasoning_dna.md) -
  geparkte Idee fuer Rechen-/Reasoning-DNA mit mentalem Arbeitsraum.

Experiment-Regel: Nichts aus `docs/experimental/` geht in den echten
Pretraining-Mix, bevor eine Ablation ein klares Signal zeigt.

## Historical Specs

Diese Dateien sind wertvoll, aber nicht automatisch aktueller Run-Plan:

- [../Doc/SPECs/SPEC_PHASE_0_TOKENIZER.md](../Doc/SPECs/SPEC_PHASE_0_TOKENIZER.md)
- [../Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md](../Doc/SPECs/SPEC_PHASE_1_PRETRAINING.md)
- [../Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md](../Doc/SPECs/SPEC_PHASE_2_CONTINUED_BILINGUAL.md)
- [../Doc/SPECs/SPEC_PHASE_3_SFT.md](../Doc/SPECs/SPEC_PHASE_3_SFT.md)
- [../Doc/SPECs/SPEC_PHASE_4_ORPO_ALIGNMENT.md](../Doc/SPECs/SPEC_PHASE_4_ORPO_ALIGNMENT.md)
- [../Doc/SPECs/SPEC_PHASE_5_LORA_SYSTEM.md](../Doc/SPECs/SPEC_PHASE_5_LORA_SYSTEM.md)
- [../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md](../Doc/SPECs/SPEC_MULTI_GPU_TRAINING.md)

## References

- [../Doc/REFERENCES/attention_and_position_encoding.md](../Doc/REFERENCES/attention_and_position_encoding.md)
- [../Doc/REFERENCES/data_pipeline_v1.md](../Doc/REFERENCES/data_pipeline_v1.md)
- [../Doc/REFERENCES/data_pipelines_and_frameworks.md](../Doc/REFERENCES/data_pipelines_and_frameworks.md)
- [../Doc/REFERENCES/mora_integration.md](../Doc/REFERENCES/mora_integration.md)

## Known Cleanup Debt

- Einige alte Markdown-Dateien enthalten Mojibake/Encoding-Reste
  (kaputte Umlaut-/Dash-Sequenzen). Das ist Doku-Schmutz, kein
  Trainingsblocker.
- Alte Phasen-Specs enthalten Groessen wie 2-3B/3B, waehrend aktuelle
  Canary-Arbeit bei 500M liegt.
- Alte Pfade wie `curated_40b`, `phase1_pretrain` und `tokenized/phase1`
  koennen historisch korrekt sein, sind aber nicht mehr der aktuelle Default.
