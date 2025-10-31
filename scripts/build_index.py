# add at the very top, before imports that load torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)



import os, json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---- Config ----
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # try "BAAI/bge-m3" later
DATA_FILES = [
    "data_processed/chunks.jsonl",       # PPT slides
    "data_processed/chunks_pdf.jsonl",   # PDF pages (optional)
]
INDEX_PATH = "embeddings/class_index.faiss"
META_PATH  = "embeddings/meta.npy"
TEXTS_PATH = "embeddings/texts.npy"
BATCH_SIZE = 16  # was 64
model = SentenceTransformer(MODEL_NAME, device="cpu")
USE_COSINE = True   # cosine (preferred). If False -> L2

def iter_chunks(paths):
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

def main():
    # 1) Load texts & metadata
    texts, meta = [], []
    for row in iter_chunks(DATA_FILES):
        txt = (row.get("text") or "").strip()
        if not txt:
            txt = " "  # keep placeholder
        texts.append(txt)
        meta.append({k: v for k, v in row.items() if k != "text"})
    if not texts:
        raise SystemExit("No chunks found. Did you run the parsers?")

    # 2) Load local embedding model
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # 3) Encode in batches
    print(f"Encoding {len(texts)} chunks with {MODEL_NAME} ...")
    # normalize for cosine if using IP index
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=USE_COSINE
    ).astype("float32")

    # 4) Build FAISS index
    dim = embeddings.shape[1]
    if USE_COSINE:
        index = faiss.IndexFlatIP(dim)   # cosine via dot-product on normalized vectors
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 5) Save artifacts
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(meta, dtype=object))
    np.save(TEXTS_PATH, np.array(texts, dtype=object))
    print(f"Saved index -> {INDEX_PATH}\nSaved meta -> {META_PATH}\nSaved texts -> {TEXTS_PATH}")

if __name__ == "__main__":
    main()
