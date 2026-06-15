# helix-rag / helix-web v0 — RAG-Wissensmodus

**Sicherer lokaler Wikipedia- + Web-RAG-Prototyp. Stark bei direkten Fakten. Sicher bei Fakes.
Schwach bei indirekter Extraktion und mehrdeutigen Kurzbegriffen.** Umgeht Helix 0.9Bs dünnes
Weltwissen, indem es Wissen RETRIEVT und der grounded-Reader es LIEST — statt zu raten. Kein neues SFT.

## Architektur (v0, schlank, keine schweren Deps)
```
Frage → Query-Rewriter → Retriever → Top-k Kontext → helix-grounded liest → kein Treffer = "weiss es nicht"
```
- **`helix-rag`** = lokales **de-Wikipedia** (`wikimedia/wikipedia 20231101.de`, Titel + Intro), **SQLite FTS5 (BM25)**
  unter `/workspace/v2data/rag/dewiki.fts5.db` (**2.838.744 Artikel**, NICHT in git). Builder `rag_build.py` (~1000/s, ~48 min).
- **Harte Titelregel** (`rag_finalize.py` baut eine indizierte `titlemap`): bei „Was ist X?" gewinnt der Artikel
  mit **Titel == X** über jeden Sportverein/Township/Nebenartikel; danach BM25-Auffüllung. Begriffsklärungen/
  Namenslisten gefiltert, extrem kurze Stubs (<80 Zeichen) raus.
- **v0.1 Redirect/Alias-Auflösung** (`rag_aliases.py` baut `aliasmap`: **1.797.661 Aliase** aus dem echten de-Wiki
  `redirect.sql` + `page.sql`-Join): Term → Alias → Exakt-Titel — **Katze→Hauskatze, Hund→Haushund, Auto→Automobil,
  USA→Vereinigte Staaten**. Bei Exakt-/Alias-Treffer **keine BM25-Beimischung** (sauberer Einzel-Kontext; sonst zieht
  der Reader Nebenartikel wie „Ein dicker Hund").
- **`helix-web`** = live **DuckDuckGo** (`ddgs`, kein API-Key, `region=de-de`), Top-4 Snippets als Kontext —
  für **Unbekanntes / Aktuelles** (z. B. „Hauptstadt von Australien" → Canberra ✅).
- **Reader = `helix-grounded`** (bewusst): ehrlich (lehnt ab statt zu erfinden). corrective@0.5 extrahiert mehr,
  **konfabuliert aber** (Fake→erfundenes Werkzeug, vermischt Kontext+Modellwissen) = „Halluzination mit Quellen-Deko".
  Bei RAG gilt: lieber „steht nicht im Text" als selbstbewusst erfundener Mist.

## Gate (getrennte Metriken — `rag_gate.py`, voller Index)
`retrieval_hit 6/6 · reader_hit 6/6 · safe_abstain 2/2 · bad_answer 0/2`. Ehrliche Nuance (Substring-Metrik ist gnädig):
- **Sauber:** Frankreich→Paris, GPU, Berlin. **Mittel:** Hardware (zirkulär), Photosynthese (Nischensatz).
- **Fail:** „Katze" — de-Wiki nennt den Tier-Artikel **„Hauskatze"**, „Katze" ist Begriffsklärung → kein Exakt-Titel.
  Reader fabuliert dann „existiert nicht". (Redirect-/Synonym-Auflösung fehlt → v1.)
- **Sicher:** Fakes (Moxthal/Glaztronk) → **0 erfundene Antworten**, ehrliche Absage. Das ist die Kernzusage.

→ Befund klar trennbar: **Retriever stark** (Titelregel), **Reader ist die variable Größe** (sauber bei direkten
Fakten, schwach bei indirekter Formulierung „X *wurde gewählt zum* Kanzler", konfabuliert nie auf Fakes).

## v0-Grenzen (= Modellgröße / Datenform, kein Bug)
1. **Indirekte Extraktion:** grounded liest „X *ist* Y" sauber, „X *wurde* Y *gewählt*" nicht. → größerer Reader.
2. **Mehrdeutige Kurzbegriffe** — in **v0.1 weitgehend gelöst** über die Redirect/Alias-Tabelle (Katze/Hund/Auto/USA
   funktionieren jetzt). Rest: echte Synonyme/Paraphrasen ohne Redirect → Embeddings.
3. **Reader-Extraktion** greift mal einen Nebensatz statt der Definition. → größerer Reader.

## v1 (NICHT beides gleichzeitig — erst v0/v0.1 stabil)
1. **Embeddings fürs Retrieval** (`sentence-transformers` + `faiss`) → echte Synonyme/Paraphrasen jenseits der Aliase.
2. **Größerer Reader (3B)** für indirekte Extraktion + sauberere Definitions-Extraktion.

## Betrieb
Modi `helix-rag` / `helix-web` im Shim (`scripts/serving/helix_ollama_server.py`). Brauchen die DB auf dem Server
+ `ddgs` (pip) fürs Web. Shim-Neustart siehe [[auralis-hub-helix-serving]]. Index neu bauen: `rag_build.py` (LIMIT=0)
→ `rag_finalize.py`.
