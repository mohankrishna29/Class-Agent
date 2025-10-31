# scripts/query.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"

# ---- config (match ingest.py) ----
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")  # "local" or "openai"
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "thenlper/gte-small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # used only if provider=openai

# ---- load index + meta ----
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

meta_path_parquet = INDEX_DIR / "meta.parquet"
meta_path_jsonl = INDEX_DIR / "meta.jsonl"
if meta_path_parquet.exists():
    meta = pd.read_parquet(meta_path_parquet)
elif meta_path_jsonl.exists():
    meta = pd.read_json(meta_path_jsonl, lines=True)
else:
    raise FileNotFoundError("No meta.parquet or meta.jsonl found in index/")

# ---- embed function ----
if EMBED_PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI()
    def embed(q: str) -> np.ndarray:
        emb = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
        v = np.array(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(v)
        return v
else:
    # local (SentenceTransformers)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBED_LOCAL_MODEL)
    def embed(q: str) -> np.ndarray:
        v = _model.encode([q], normalize_embeddings=True)
        return np.array(v, dtype="float32")

def retrieve(query: str, k=5):
    v = embed(query)
    D, I = index.search(v, k)
    rows = meta.iloc[I[0]].to_dict(orient="records")
    # attach scores
    for j, r in enumerate(rows):
        r["_score"] = float(D[0][j])
    return rows

if __name__ == "__main__":
    print(f"Provider={EMBED_PROVIDER} | Model={EMBED_LOCAL_MODEL if EMBED_PROVIDER!='openai' else EMBED_MODEL}")
    while True:
        q = input("Ask a question (or 'q' to quit): ")
        if q.strip().lower() == "q":
            break
        ctx = retrieve(q, k=5)
        print("\nTop contexts:")
        for i, r in enumerate(ctx, 1):
            cite = f"{r.get('source_id','?')}#{r.get('locator','?')}"
            preview = (r.get('text','') or "")[:300].replace("\n"," ")
            print(f"[{i}] score={r['_score']:.3f}  {cite}\n{preview}...\n")
