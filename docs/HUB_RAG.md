# helix-rag / helix-web v0 — RAG Knowledge Mode

**Safe local Wikipedia + Web RAG prototype. Strong on direct facts. Safe on fakes.
Weak on indirect extraction and ambiguous short terms.** It works around Helix 0.9B's thin
world knowledge by RETRIEVING knowledge and having the grounded reader READ it — instead of guessing. No new SFT.

## Architecture (v0, lean, no heavy deps)
```
Question → Query Rewriter → Retriever → Top-k context → helix-grounded reads → no hit = "don't know"
```
- **`helix-rag`** = local **de-Wikipedia** (`wikimedia/wikipedia 20231101.de`, title + intro), **SQLite FTS5 (BM25)**
  under `/workspace/v2data/rag/dewiki.fts5.db` (**2,838,744 articles**, NOT in git). Builder `rag_build.py` (~1000/s, ~48 min).
- **Hard title rule** (`rag_finalize.py` builds an indexed `titlemap`): for "What is X?" the article
  with **title == X** wins over any sports club/township/secondary article; BM25 fill-up afterward. Disambiguation pages/
  name lists filtered, extremely short stubs (<80 characters) removed.
- **v0.1 redirect/alias resolution** (`rag_aliases.py` builds `aliasmap`: **1,797,661 aliases** from the real de-Wiki
  `redirect.sql` + `page.sql` join): term → alias → exact title — **Katze→Hauskatze, Hund→Haushund, Auto→Automobil,
  USA→Vereinigte Staaten**. On an exact/alias hit **no BM25 admixture** (clean single context; otherwise
  the reader pulls in secondary articles like "Ein dicker Hund").
- **`helix-web`** = live **DuckDuckGo** (`ddgs`, no API key, `region=de-de`), top-4 snippets as context —
  for **unknown / current** items (e.g. "Capital of Australia" → Canberra ✅).
- **Reader = `helix-grounded`** (deliberately): honest (refuses instead of inventing). corrective@0.5 extracts more,
  **but confabulates** (fake→invented tool, mixes context + model knowledge) = "hallucination with source decoration".
  For RAG the rule is: better "not in the text" than confidently invented nonsense.

## Gate (separated metrics — `rag_gate.py`, full index)
`retrieval_hit 6/6 · reader_hit 6/6 · safe_abstain 2/2 · bad_answer 0/2`. Honest nuance (the substring metric is lenient):
- **Clean:** Frankreich→Paris, GPU, Berlin. **Medium:** Hardware (circular), Photosynthesis (niche sentence).
- **Fail:** "Katze" — de-Wiki calls the animal article **"Hauskatze"**, "Katze" is a disambiguation page → no exact title.
  The reader then fabulates "does not exist". (Redirect/synonym resolution missing → v1.)
- **Safe:** Fakes (Moxthal/Glaztronk) → **0 invented answers**, honest refusal. That is the core promise.

→ Finding cleanly separable: **retriever strong** (title rule), **reader is the variable** (clean on direct
facts, weak on indirect phrasing "X *was elected* chancellor", never confabulates on fakes).

## v0 limits (= model size / data form, not a bug)
1. **Indirect extraction:** grounded reads "X *is* Y" cleanly, "X *was* Y *elected*" not. → larger reader.
2. **Ambiguous short terms** — in **v0.1 largely solved** via the redirect/alias table (Katze/Hund/Auto/USA
   work now). The rest: real synonyms/paraphrases without a redirect → embeddings.
3. **Reader extraction** sometimes grabs a subordinate clause instead of the definition. → larger reader.

## v1 (NOT both at once — first v0/v0.1 stable)
1. **Embeddings for retrieval** (`sentence-transformers` + `faiss`) → real synonyms/paraphrases beyond the aliases.
2. **Larger reader (3B)** for indirect extraction + cleaner definition extraction.

## Operation
Modes `helix-rag` / `helix-web` in the shim (`scripts/serving/helix_ollama_server.py`). They need the DB on the server
+ `ddgs` (pip) for the web. Shim restart see [[auralis-hub-helix-serving]]. Rebuild index: `rag_build.py` (LIMIT=0)
→ `rag_finalize.py`.
