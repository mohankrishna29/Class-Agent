# app/router/intent_router.py
from __future__ import annotations
import re
from typing import Literal, TypedDict

Label = Literal["ASSESSMENT_GUARD", "COURSE_EXPLAIN", "LAB_HELP"]

ASSESSMENT_PATTERNS = [
    r"\b(midterm|final|quiz|exam)\b",
    r"\bQuestion\s*\d+\b", r"\bQ\s*\d+\b",
    r"\bMCQ\b", r"\banswer key\b", r"\bmarks?\b", r"\bpoints?\b",
    r"\bchoose (one|the best)\b", r"\bwhat is the output\b", r"\bprove that\b",
    r"\boption\s*[A-D]\b", r"\b[A-D]\)\s"
]
LAB_PATTERNS = [
    r"\berror\b", r"\bexception\b", r"\btimeout\b", r"\btraceback\b",
    r"\bconnect(ing)?\b", r"\bpermission\b", r"\bAccessDenied\b",
    r"\baws\b|\biam\b|\bs3\b|\bdynamodb\b|\bterraform\b|\bcli\b|\bsdk\b|\bboto3\b"
]
EXPLAIN_PATTERNS = [
    r"\bexplain\b", r"\boverview\b", r"\bcompare\b", r"\bhow does\b", r"\bwhy\b",
    r"\bwalk me through\b", r"\bintuition\b"
]

def _score(msg: str, patterns) -> int:
    return sum(bool(re.search(p, msg, flags=re.I)) for p in patterns)

def route_intent(msg: str) -> Label:
    s_assess = _score(msg, ASSESSMENT_PATTERNS)
    s_lab    = _score(msg, LAB_PATTERNS)
    s_expl   = _score(msg, EXPLAIN_PATTERNS)

    # conservative tie-break: ASSESSMENT > LAB > EXPLAIN
    ranked = sorted(
        [("ASSESSMENT_GUARD", s_assess), ("LAB_HELP", s_lab), ("COURSE_EXPLAIN", s_expl)],
        key=lambda x: x[1],
        reverse=True
    )
    label = ranked[0][0]
    # fallback if everything is zero
    return label if ranked[0][1] > 0 else "COURSE_EXPLAIN"
