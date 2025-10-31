# app/pipelines/hint_pipeline.py
from __future__ import annotations


from typing import List, Dict, Any, Callable

SYSTEM_HINT_FALLBACK = """\
{GHOST_POLICY}
You are a helpful course assistant in Hint-Only mode for assessment-like prompts.
Output format:
1) One-line stance that you cannot give the answer.
2) 2–3 bullet hints (no final values).
3) Where to look (cite course material IDs).
4) Optional analogous mini-example (different inputs).
5) Ask for their attempt.
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

def answer_assessment_guard(
    query: str,
    retriever: Callable[[str, int], List[Dict[str, Any]]],
    llm_generate: Callable[[str, str], str],
    k: int = 5,
) -> str:
    chunks = retriever(query, k=k)
    context = "\n\n".join(
        f"[{c.get('where','?')}] {c.get('text','')[:600]}" for c in chunks
    )
    system = _load_prompt("prompts/hint_mode_system.txt", SYSTEM_HINT_FALLBACK)
    user = f"{query}\n\nCourse context:\n{context}"
    return llm_generate(system, user)
