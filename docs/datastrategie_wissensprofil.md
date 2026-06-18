# Data Strategy: Knowledge Profiles instead of Overall Val-Loss

**Guiding principle:**
> **Further data collection is done on the basis of *knowledge profiles*, not on the basis of
> the overall val-loss.**

Many projects blindly train more web text and hope for miracles. Instead, we measure
*which kind of knowledge* the model learns well/poorly and then collect
**targeted** data for the gaps. For scaling to 3B+, that is more valuable than another
0.05 val-loss.

## How the profile is measured (`scripts/eval/fact_recall_eval.py`)
- Contrastive margin `NLL(wrong) − NLL(right)` per fact, **multiple distractors**.
- **Distractor difficulty levels separated** (easy / med / hard), to decouple *knowledge quality* from
  *test difficulty* (e.g. gold symbol: Au vs Berlin = easy, vs Fe = med, vs Ag
  = hard).
- **Top-k** check = recall proximity (does the right candidate come out on top?).
- 5 categories: geo / sci / hist / lang / **tech** (= *concepts*, **not** trained
  code — this run had 0% code; high tech values ≠ programming ability).

## Profile — as of step 35k (n=57)
| Category | strict | easy | med | hard | top-10 |
|---|---|---|---|---|---|
| History | 92% | 100% | 100% | 92% | 58% |
| Geography | 83% | 100% | 100% | 83% | 100% |
| Tech concepts | 80% | 100% | 90% | 80% | 30% |
| Science | 67% | 100% | 67% | 67% | 17% |
| Language/Translation | 64% | 73% | 91% | 64% | 18% |
| **Overall** | **77%** | **95%** | **89%** | **77%** | **46%** |

## Reading (honest)
- **95% easy** = solid floor, **no guessing**. The architecture takes up knowledge.
- **Gradient 95→89→77** = the weakness is **fine discrimination**, not missing knowledge.
- **Strong:** history, geography (Wikipedia strength; for geo the correct city
  is a top token 100% of the time).
- **Weaker:** **science** (symbols/fine values: Au/Ag, Jupiter/Saturn) and
  **language/translation** (cross-lingual). Both also with **low recall proximity**
  (top-10 17-18%) → knowledge present, but not well "at the surface".
- **"29% science" from earlier was overstated** (small battery + only hard
  distractors). With difficulty levels: science easy 100%, strict 67%.

## Derived data priorities (if 50k confirms the profile)
NO more blind web text. Targeted:
1. **Science / school knowledge / textbooks / fact-QA / symbol tables** — the fine-
   symbolic gap (chemistry, physics, astronomy).
2. **Cross-lingual / translation data** — parallel de↔en text, dictionaries.
3. **Real code** — *only if coding is a goal*: currently 0% code trained
   (~677M code tokens are ready, but that is rather little for solid coding).

## Gate
These priorities become binding **only after the 50k run**: the profile will be re-measured at
step 50k with the same (extended) battery. If the pattern holds, the targeted data collection
is justified — not before, from the gut.
