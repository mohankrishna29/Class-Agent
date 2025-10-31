# app/router/post_filters.py
import re

BLOCK_PATTERNS = [
    r"\b(the\s+answer\s+is)\b",
    r"\b(correct\s+option\s+is)\b",
    r"^\s*[A-D]\s*$",          # single letter on a line
    r"^\s*\d+(\.\d+)?\s*$",    # single numeric value
]

def redact_final_answers(text: str) -> str:
    redacted = text
    for pat in BLOCK_PATTERNS:
        redacted = re.sub(pat, "[redacted]", redacted, flags=re.I | re.M)
    return redacted
