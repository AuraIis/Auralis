import urllib.request, shutil, pathlib
DST = pathlib.Path("/workspace/v2data/data/raw/anneal_candidates"); DST.mkdir(parents=True, exist_ok=True)
FILES = [
 ("fineweb2_de_shard0.parquet", "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/deu_Latn/train/000_00000.parquet"),
 ("python_edu_0.parquet", "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/python-edu/train-00000-of-00002.parquet"),
 ("python_edu_1.parquet", "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/python-edu/train-00001-of-00002.parquet"),
 ("cosmopedia_v2_shard0.parquet", "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-00000-of-00104.parquet"),
]
for name, url in FILES:
    out = DST/name
    if out.exists() and out.stat().st_size > 1_000_000:
        print("skip (vorhanden)", name, out.stat().st_size, flush=True); continue
    print("lade", name, "...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(out,"wb") as f:
        shutil.copyfileobj(r, f, length=4*1024*1024)
    print("fertig", name, round(out.stat().st_size/1e9,2),"GB", flush=True)
print("ALL_DONE", flush=True)
