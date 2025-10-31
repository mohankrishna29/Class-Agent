# app/rag.py  — RAG service (stable on macOS/ARM, lazy CPU embeddings)

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------- Stability knobs for macOS/ARM ----------
# (These are harmless on other platforms too.)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# ---------------------------------------------------

import numpy as np
import faiss

# If you want .env loading (optional):
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----- Paths -----
ROOT = Path(__file__).resolve().parents[1]  # project root (one above app/)
INDEX_PATH = ROOT / "embeddings" / "class_index.faiss"
META_PATH  = ROOT / "embeddings" / "meta.npy"
TEXTS_PATH = ROOT / "embeddings" / "texts.npy"

# ----- Model/Index settings -----
# If your FAISS index was built from normalized embeddings (recommended for cosine),
# keep normalize_embeddings=True in encode.
MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
USE_COSINE = True   # informational; index already built
DEFAULT_K  = int(os.getenv("RETRIEVE_K_DEFAULT", "5"))


class RAGService:
    """
    Loads FAISS + corpus arrays once; lazily loads the SentenceTransformer
    on first query (CPU-only) to avoid MPS/Torch crashes on macOS.
    """
    def __init__(
        self,
        k_default: int = DEFAULT_K,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        texts_path: Optional[Path] = None,
        model_name: str = MODEL_NAME,
    ):
        self.k_default = k_default
        self.index_path = Path(index_path or INDEX_PATH)
        self.meta_path  = Path(meta_path  or META_PATH)
        self.texts_path = Path(texts_path or TEXTS_PATH)
        self.model_name = model_name

        # Load FAISS and arrays once
        self.index = faiss.read_index(str(self.index_path))
        self.meta  = np.load(str(self.meta_path), allow_pickle=True)
        self.texts = np.load(str(self.texts_path), allow_pickle=True)

        # Lazy embedding model
        self._emb = None  # type: ignore

    # ---- Lazy init of sentence-transformers on CPU ----
    def _ensure_model(self):
        if self._emb is None:
            # Import here (not at module load) to avoid initializing Torch early
            from sentence_transformers import SentenceTransformer
            # Force CPU for stability on M1
            self._emb = SentenceTransformer(self.model_name, device="cpu")

    def _embed_query(self, q: str) -> np.ndarray:
        self._ensure_model()
        v = self._emb.encode([q], normalize_embeddings=True)  # keep normalized for cosine/IP
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        # FAISS expects shape (n, d)
        return v

    @staticmethod
    def _format_locator(d: dict) -> str:
        if "slide" in d:
            return f"{d['source']} (slide {d['slide']})"
        if "page" in d:
            return f"{d['source']} (page {d['page']})"
        return str(d.get("source", "?"))

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return top-k chunks for question with metadata expected by pipelines."""
        k = k or self.k_default
        qv = self._embed_query(question)
        D, I = self.index.search(qv, k)

        out: List[Dict[str, Any]] = []
        for pos, i in enumerate(I[0]):
            d = self.meta[i]
            t = self.texts[i]
            out.append({
                "where":     self._format_locator(d),
                "text":      str(t),
                "source_id": d.get("source_id", d.get("source", "")),
                "locator":   d.get("slide", d.get("page", None)),
                "_score":    float(D[0][pos]),
            })
        return out
