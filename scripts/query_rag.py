# scripts/query_rag.py  — CLEAN ROUTER CLI (no faiss/torch here)

from openai import OpenAI
import os

# Router + pipelines
from app.router.intent_router import route_intent
from app.router.post_filters import redact_final_answers
from app.pipelines.hint_pipeline import answer_assessment_guard
from app.pipelines.explain_pipeline import answer_from_course
from app.pipelines.web_pipeline import lab_help
from web.search import simple_search

# Use the service from app.rag (it loads FAISS etc. internally)
from app.rag import RAGService

# Instantiate once
_rag = RAGService()

# Adapter so pipelines can call retrieve(query, k)
def retrieve(query: str, k: int = 5):
    return _rag.retrieve(query, k=k)

# Minimal LLM wrapper
def llm_generate(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    import os
    client = OpenAI()

    primary = os.getenv("OPENAI_MODEL", "gpt-4o")
    fallback = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

    def call(model):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        ).choices[0].message.content.strip()

    try:
        return call(primary)
    except Exception as e:
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            return call(fallback)
        raise


import json, time
def handle_query(query: str):
    label = route_intent(query)
    print("[router]", json.dumps({"ts": time.time(), "label": label}) )

    if label == "ASSESSMENT_GUARD":
        print("\n🧩 [Assessment-style detected — switching to Hint Mode]")
        result = answer_assessment_guard(query, retrieve, llm_generate, k=5)
        result = redact_final_answers(result)

    elif label == "LAB_HELP":
        print("\n🔧 [Lab troubleshooting detected — using Web Help Mode]")
        result = lab_help(query, retrieve, llm_generate, external_search=simple_search, k=3)

    else:
        print("\n📘 [Concept explain mode — using Course RAG]")
        result = answer_from_course(query, retrieve, llm_generate, k=5)

    print("\n" + result)

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'q' to quit): ")
        if q.lower() == "q":
            break
        handle_query(q)
