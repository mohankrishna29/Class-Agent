from __future__ import annotations
import os
from pathlib import Path
from fastapi import UploadFile, File, BackgroundTasks
import sys, subprocess

from fastapi import FastAPI, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.rag import RAGService
from app.router.intent_router import route_intent
from app.router.post_filters import redact_final_answers
from app.pipelines.hint_pipeline import answer_assessment_guard
from app.pipelines.explain_pipeline import answer_from_course
from app.pipelines.web_pipeline import lab_help
from web.search import simple_search
from openai import OpenAI
import os

from .rag import RAGService  # uses your real RAG

API_TOKEN = os.getenv("API_TOKEN", "dev-token")

# Create app and RAG service
app = FastAPI(title="Class RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = OpenAI()
rag = RAGService()  # loads FAISS/meta/texts once at startup

# ----- Auth helper -----
def require_auth(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid auth")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

# ----- Serve the minimal UI -----
ROOT = Path(__file__).resolve().parents[1]   # repo root
UI_DIR = ROOT / "web"                        # expects web/index.html
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

@app.get("/")
def root():
    # redirect homepage to the UI
    return RedirectResponse(url="/ui")

# ----- API models -----
class AskPayload(BaseModel):
    question: str
    k: int = 5

# ----- API routes -----
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
@app.post("/ask")
def ask(payload: AskPayload, authorization: str | None = Header(default=None)):
    """
    Handles the student's question using the ghost-instruction router.
    - Detects if it's an assessment, concept, or lab query.
    - Routes to the correct pipeline.
    - Applies output filters for assessment mode.
    """
    require_auth(authorization)

    query = payload.question.strip()
    label = route_intent(query)

    def retrieve(q, k=payload.k):
        return rag.retrieve(q, k=k)

    def llm_generate(system_prompt: str, user_prompt: str) -> str:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Optional fallback
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                fallback = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
                resp = client.chat.completions.create(
                    model=fallback,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            raise e

    # --- Route based on intent ---
    if label == "ASSESSMENT_GUARD":
        print("🧩 Hint Mode triggered")
        out = answer_assessment_guard(query, retrieve, llm_generate, k=payload.k)
        out = redact_final_answers(out)
        return {"mode": label, "text": out}

    elif label == "LAB_HELP":
        print("🔧 Lab Help Mode triggered")
        out = lab_help(query, retrieve, llm_generate, external_search=simple_search, k=payload.k)
        return {"mode": label, "text": out}

    else:  # COURSE_EXPLAIN
        print("📘 Explain Mode triggered")
        out = answer_from_course(query, retrieve, llm_generate, k=payload.k)
        return {"mode": label, "text": out}


ROOT = Path(__file__).resolve().parents[1]

@app.post("/upload")
def upload(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    require_auth(authorization)
    dest_dir = ROOT / "data_raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename
    with dest.open("wb") as f:
        f.write(file.file.read())
    return {"saved": str(dest)}

def _run_reindex():
    # Runs your existing indexing script; writes logs to manifests/reindex.log
    log_dir = ROOT / "manifests"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "reindex.log"
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "build_index.py")]
    with log_file.open("w") as lf:
        subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, check=False)

@app.post("/reindex")
def reindex(background: BackgroundTasks, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    background.add_task(_run_reindex)
    return {"status": "started", "log": str(ROOT / 'manifests' / 'reindex.log')}

@app.post("/reload")
def reload_index(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    info = rag.reload()
    return {"reloaded": True, **info}