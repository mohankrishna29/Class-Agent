# app/pipelines/web_pipeline.py
from __future__ import annotations
from typing import List, Dict, Any, Callable

SYSTEM_LAB = """\
Ghost Policy — Lab Help Mode:
- Goal: help finish the activity safely.
- Prefer official docs; include step-by-step remediation, diagnostics, and cautions.
- Never include secrets; show env var NAMES only.
- Provide links/citations for any external guidance.
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

def lab_help(
    query: str,
    rag_retriever,
    llm_generate,
    external_search: Callable[[str], List[Dict[str, str]]],
    k: int = 3,
) -> str:
    chunks = rag_retriever(query, k=k)
    if not chunks:
        ext = external_search(query)  # [{'title':..., 'url':..., 'snippet':...}, ...]
        ext_text = "\n".join(f"- {d['title']} — {d['url']}\n  {d.get('snippet','')}" for d in ext[:6])
        user = f"{query}\n\nUseful external references:\n{ext_text}"
    else:
        ctx = "\n\n".join(
            f"[{c.get('source_id','?')}] p{c.get('page','?')}: {c.get('text','')[:800]}"
            for c in chunks
        )
        user = f"{query}\n\nCourse lab context:\n{ctx}\n\nIf insufficient, state what else to check and reference official docs."
    return llm_generate(SYSTEM_LAB, user)
