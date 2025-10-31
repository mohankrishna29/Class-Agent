import os, json, re, sys
from glob import glob
import fitz  # PyMuPDF

RAW_DIR = os.path.join("data_raw")     # looks for *.pdf here
OUT_FILE = os.path.join("data_processed", "chunks_pdf.jsonl")

def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_pdf_file(path: str):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text")  # plain text; good first pass
        pages.append({
            "source": os.path.basename(path),
            "source_path": path,
            "doc_type": "pdf",
            "page": i,
            "text": clean(txt)
        })
    return pages

def main():
    pdf_files = sorted(glob(os.path.join(RAW_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}. Put your textbook PDF there.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    total = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for f in pdf_files:
            try:
                chunks = parse_pdf_file(f)
                for c in chunks:
                    c["text"] = c.get("text", "")
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += len(chunks)
            except Exception as e:
                print(f"[WARN] Failed to parse {f}: {e}", file=sys.stderr)

    print(f"Done. Wrote {total} page chunks to {OUT_FILE}")

if __name__ == "__main__":
    main()
