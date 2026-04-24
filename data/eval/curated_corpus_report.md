# Curated Corpus Preparation Report

| Language | Source | Policy | Exists | GB | Target Tokens | Notes |
|---|---|---|:-:|--:|--:|---|
| english | `fineweb_edu` | keep | yes | 40.00 | 14.00B | English backbone. |
| english | `fineweb2_en` | acquire | no | 0.00 | 8.00B | Modern web EN top-up. |
| english | `wikipedia_en` | keep | yes | 12.00 | 3.00B | Stable factual anchor. |
| english | `dolma` | acquire | no | 0.00 | 3.50B | Diversity top-up; use the script-filtered subset only. |
| english | `openmath` | keep | yes | 8.00 | 1.50B | Useful, but deliberately capped. |
| german | `german_commons` | acquire | no | 0.00 | 4.50B | Primary curated German source; do not hide it inside a merged blob. |
| german | `fineweb2_de` | acquire | no | 0.00 | 2.50B | Modern web German. |
| german | `wikipedia_de` | acquire | no | 0.00 | 1.00B | Factual German anchor. |
| code | `the_stack_v2` | acquire | no | 0.00 | 1.70B | Preferred final code backbone. |
| code | `open_web_math` | keep_small | yes | 0.88 | 0.30B | Small structured reasoning top-up only. |
