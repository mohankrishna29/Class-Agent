# scripts/map_questions_docx.py
from pathlib import Path
import re, json, uuid, os
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
QB_ROOT = ROOT / "data_raw" / "question_banks"
OUT = ROOT / "data_clean"
OUT.mkdir(parents=True, exist_ok=True)

LEAK = os.getenv("LEAK_ANSWERS", "false").lower() == "true"

# Heuristics: works for most MCQ docs that look like:
# Q: ... (or just a question line)
# A) ..., B) ..., C) ..., D) ...
# Answer: B (or Correct: B), Explanation: ...
Q_PAT = re.compile(r"^(?:Q[:\.\)]\s*)?(.*\S.*)$", re.IGNORECASE)
CHOICE_PAT = re.compile(r"^[A-H][\)\.]\s+.+")
ANS_PAT = re.compile(r"^(?:answer|correct)\s*[:\-]\s*(.+)$", re.IGNORECASE)
EXPL_PAT = re.compile(r"^(?:explanation|why|rationale)\s*[:\-]\s*(.+)$", re.IGNORECASE)
MODULE_HINT = re.compile(r"^(module|topic|domain)[:\-]\s*(.+)$", re.IGNORECASE)

def parse_docx(path: Path):
    doc = Document(path)
    # flatten paragraphs; ignore empty/whitespace-only
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    items = []

    i = 0
    while i < len(lines):
        # find a question line
        qline = lines[i]
        # skip headers like "Module: X"
        if MODULE_HINT.match(qline):
            i += 1
            continue

        # treat any non-choice, non-answer line as start of a Q
        if CHOICE_PAT.match(qline) or ANS_PAT.match(qline) or EXPL_PAT.match(qline):
            i += 1
            continue

        # capture question (may span multiple lines until a choice starts)
        qbuf = [qline]
        i += 1
        while i < len(lines) and not (CHOICE_PAT.match(lines[i]) or ANS_PAT.match(lines[i]) or EXPL_PAT.match(lines[i]) or MODULE_HINT.match(lines[i])):
            qbuf.append(lines[i])
            i += 1
        question = " ".join(qbuf).strip()

        # capture choices
        choices = []
        while i < len(lines) and CHOICE_PAT.match(lines[i]):
            choices.append(lines[i])
            i += 1

        answer = None
        explanation = None
        module = None

        # optional trailing metadata (order can vary)
        j = i
        while j < len(lines) and (ANS_PAT.match(lines[j]) or EXPL_PAT.match(lines[j]) or MODULE_HINT.match(lines[j])):
            m = ANS_PAT.match(lines[j])
            if m: answer = m.group(1).strip()
            m = EXPL_PAT.match(lines[j])
            if m: explanation = m.group(1).strip()
            m = MODULE_HINT.match(lines[j])
            if m: module = m.group(2).strip()
            j += 1
        i = j

        if not question:
            continue

        # Build text for embedding
        text_parts = [question]
        if choices:
            text_parts.append("\n".join(choices))
        if explanation:
            text_parts.append("Explanation: " + explanation)
        if LEAK and answer:
            text_parts.append("Answer: " + answer)

        text = "\n".join(text_parts).strip()

        items.append({
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "source_id": f"question_banks/{path.name}",
            "locator": "auto",
            "mimetype": "application/x-question",
            "title": module or "",
            "extra": {
                "answer_key": answer if answer else "",
                "choices_count": len(choices),
                "module": module or "",
            },
            "hash": "",  # fill after
        })

    # add hashes
    import hashlib
    for it in items:
        it["hash"] = hashlib.sha256(it["text"].encode("utf-8")).hexdigest()
    return items

def main():
    # scan both extracted folders
    sources = []
    for sub in ["extracted_1", "extracted_2"]:
        base = QB_ROOT / sub
        if base.exists():
            sources.extend(base.rglob("*.docx"))

    if not sources:
        print("No .docx files found under question_banks/extracted_*")
        return

    # write one JSONL per docx
    for docx_path in sources:
        items = parse_docx(docx_path)
        if not items:
            continue
        out = OUT / f"{docx_path.stem}.questions.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"Wrote {out} ({len(items)} items)")

if __name__ == "__main__":
    main()
