# scripts/canvas_extract.py
from pathlib import Path
import zipfile, shutil
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data_stage" / "canvas_extracts"
RAW_ASSETS = ROOT / "data_raw" / "course_assets"

def extract_imscc(imscc_path: Path, run_id: str):
    out = STAGE / run_id
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(imscc_path) as z:
        z.extractall(out)

    # move supported assets from the extracted tree into course_assets/canvas/<run_id>/
    for p in out.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf",".pptx",".docx",".md",".txt",".png",".jpg",".jpeg",".html",".htm"}:
            dest = RAW_ASSETS / "canvas" / run_id / p.relative_to(out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    # optional: convert HTML pages to .txt for easier ingestion
    for html in (RAW_ASSETS / "canvas" / run_id).rglob("*.htm*"):
        try:
            soup = BeautifulSoup(html.read_text(errors="ignore"), "html.parser")
            for t in soup(["script","style","nav","footer","header"]): t.decompose()
            text = "\n".join(x.get_text(" ", strip=True) for x in soup.find_all(["h1","h2","h3","p","li"]))
            html.with_suffix(".txt").write_text(text)
        except Exception:
            pass  # skip weird pages

if __name__ == "__main__":
    import sys
    imscc = Path(sys.argv[1]).expanduser().resolve()
    run_id = sys.argv[2]  # e.g., aws_academy or ift_320_2025_fall
    extract_imscc(imscc, run_id)
    print(f"Done. Extracted to {STAGE/run_id} and copied assets into {RAW_ASSETS/'canvas'/run_id}")
