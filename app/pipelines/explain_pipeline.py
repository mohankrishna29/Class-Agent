# app/pipelines/explain_pipeline.py
from __future__ import annotations
from typing import List, Dict, Any, Callable

SYSTEM_EXPLAIN = """\
Ghost Policy — Explain Mode:
- Explain concepts using ONLY course sources retrieved by RAG.
- Always cite slide/page/section from the provided chunks.
- Do not produce assessment solutions; keep examples generic.
"""

def _load_prompt(path: str, fallback: str) -> str:
    try:
        txt = open(path, "r").read()
    except Exception:
        txt = fallback
    try:
        gp = open("prompts/ghost_policy.txt").read()
        txt = txt.replace("{GHOST_POLICY}", gp)
    except Exception:
        txt = txt.replace("{GHOST_POLICY}", "")
    return txt

def answer_from_course(
    query: str,
    retriever,
    llm_generate,
    k: int = 5,
) -> str:
    chunks = retriever(query, k=k)
    context = "\n\n".join(
        f"[{c.get('source_id','?')}] p{c.get('page','?')}: {c.get('text','')[:800]}"
        for c in chunks
    )
    user = f"{query}\n\nCourse context (use and cite):\n{context}"
    return llm_generate(SYSTEM_EXPLAIN, user)
