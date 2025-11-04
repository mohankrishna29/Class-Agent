🧠 CourseIQ — Class-Specific AI Teaching Assistant
Retrieval-Augmented Generation (RAG) System for Course-Exclusive Knowledge


📘 Overview

CourseIQ is an AI-powered Teaching Assistant built specifically for students of a single course.
Unlike generic AI tools, CourseIQ knows only what the class knows — it is trained exclusively on the course’s materials such as slides, Canvas resources, PDFs, and lecture notes.

It is designed to:
- Help students understand class concepts faster.
- Guide them toward relevant course references instead of giving direct answers.
- Prevent academic dishonesty, ensuring AI support without compromising learning integrity.
- CourseIQ essentially acts as a smart, context-aware TA that lives inside the class ecosystem.


🎯 Project Motivation & Objectives

Modern AI tools (like ChatGPT) can easily provide ready-made answers — often beyond course scope — making it difficult for educators to ensure fair learning.
This project was built to counter that, with three clear goals:

Bounded Knowledge:
CourseIQ only accesses the instructor’s provided materials. No external data or web access.

Guided Learning, Not Cheating:
If a student asks a question resembling an exam or assignment (short factual or numerical query), CourseIQ does not answer directly — instead, it offers conceptual hints and class-based references.

Human-Like Assistance:
It emulates how a teaching assistant would respond — offering hints, clarifications, and citations rather than spoon-fed solutions.

⚙️ System Architecture

The application is structured as a modular Retrieval-Augmented Generation (RAG) pipeline:
                   ┌────────────────────────────┐
                   │  Course Materials (Raw)    │
                   │  PDFs, PPTs, Canvas Files  │
                   └──────────────┬─────────────┘
                                  │
                     [Data Extraction & Cleaning]
                                  │
                   ┌──────────────▼─────────────┐
                   │   Chunking & Embedding     │
                   │   (OpenAI Embeddings API)  │
                   └──────────────┬─────────────┘
                                  │
                     [Vector Storage: FAISS DB]
                                  │
                   ┌──────────────▼─────────────┐
                   │  Query Processing Layer    │
                   │  Semantic Search (top-k)   │
                   └──────────────┬─────────────┘
                                  │
                     [Context + Query → LLM]
                                  │
                   ┌──────────────▼─────────────┐
                   │  FastAPI Backend + Web UI  │
                   │  Human-readable Answers    │
                   └────────────────────────────┘


🔬 The RAG Pipeline — Core Logic
1. Data Extraction
Handles mixed data sources: Canvas ZIP files, lecture slides (PPTX), textbooks (PDF), and HTML course resources.
Extracts and normalizes text into a uniform structure.
Automatically filters duplicate or outdated Canvas data to avoid noise.

2. Cleaning & Chunking
Texts are cleaned (removal of boilerplate HTML, empty pages, and metadata).
Each document is chunked intelligently (based on semantic boundaries) to optimize context retrieval.
Metadata (e.g., source_id, page_no, module_name) is attached to every chunk for traceability.

3. Embedding & Indexing
Uses OpenAI’s embedding model to convert each text chunk into high-dimensional vectors.
Stores embeddings locally in a FAISS vector database, ensuring fast semantic search even offline.
Maintains a metadata table (meta.parquet) to cross-link chunks and sources.

4. Semantic Search (Retrieval)
On each query, the system retrieves the top-k most semantically similar chunks.
k can be tuned (3, 5, 8, etc.) depending on how broad the retrieval should be.
Ensures the results are grounded — no hallucinations or irrelevant sources.

5. Response Generation
The query + retrieved context are passed to an LLM (OpenAI GPT model).
The response is structured as:
A clear explanation or hint.
Citations referencing exact course materials for follow-up.
For suspected exam-type questions, CourseIQ switches to “hint-only” mode.


📁 Data Organization

A deliberate and transparent folder structure was enforced to maintain traceability and reusability:
courseiq/
│
├── data_raw/            # Unprocessed materials (Canvas ZIP, PDFs, PPTs)
├── processed/           # Cleaned text and structured chunks
├── index/               # FAISS vector index + metadata parquet
├── scripts/             # RAG utilities and CLI testing tools
├── app/                 # FastAPI backend
├── frontend/            # Minimal HTML interface
├── .env                 # Environment variables (OpenAI key, port)
├── Dockerfile
└── docker-compose.yml

🚀 Running the Application
1. Local Setup
  git clone https://github.com/mohankrishna29/courseiq.git
  cd courseiq
  pip install -r requirements.txt
2. Environment Variables
  Create a .env file in the root directory:
  OPENAI_API_KEY=sk-your-key
  APP_PORT=8000
3. Start the Server
   uvicorn app.main:app --reload


💡 Key Features

Canvas + Textbook Data Integration
Unified handling of course files (Canvas exports, PDFs, PPTs, HTML notes).

Intelligent Chunking & Embedding
Fine-grained segmentation with persistent metadata tracking.

Fast Vector Search (FAISS)
High-speed semantic retrieval fully local to the device.

Ethical AI Design
Detects exam-type queries and switches to hint mode.

Citations for Transparency
Every answer is grounded and references original class materials.

Dockerized Environment
Portable, architecture-neutral build (amd64 image for M1 Mac).

🧩 Challenges & How They Were Solved
| Challenge                                     | Solution                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| Identifying old vs. new Canvas data           | Introduced metadata filtering and clean rebuild scripts.                   |
| Handling multi-format inputs (PDF, PPT, HTML) | Implemented unified extraction pipeline with robust file handlers.         |
| Retrieval accuracy vs. performance            | Tuned FAISS parameters and chunk size dynamically.                         |
| Avoiding hallucination                        | Strict context retrieval — only class-based sources fed to model.          |
| Ethical AI use                                | Added logic to classify “exam-like” questions and restrict direct answers. |
| Cross-platform Docker builds                  | Built amd64 image on M1 using multi-architecture build process.            |


🧠 Project Evolution Summary
| Version | Focus                     | Key Outcomes                                      |
| ------- | ------------------------- | ------------------------------------------------- |
| **v1**  | Base RAG setup            | Initial prototype with FAISS + OpenAI embeddings. |
| **v2**  | Canvas ingestion          | Automated text extraction and metadata tagging.   |
| **v3**  | Modular organization      | Introduced module gating and reindexing strategy. |
| **v4**  | Clean rebuild             | Simplified folder structure and data tracking.    |
| **v5**  | Docker & amd64            | Created reproducible build for professor’s demo.  |
| **v6**  | Ethical learning features | Added hint-only mode for exam-type queries.       |


🔭 Future Enhancements

Module-wise retrieval tuning (e.g., prioritize recent lectures).
Integration with ASU SSO for secure access.
Dashboard for student query analytics.
Fine-grained citation linking to slide numbers or page IDs.
Instructor control panel for content updates.

🙌 Acknowledgments

This project was developed by Mohan Krishna under the guidance of his professor as part of an academic initiative to explore responsible AI in education.
Special thanks to all contributors and the ASU learning community that inspired the ethical boundaries built into CourseIQ.

## CourseIQ — Because learning should be intelligent, grounded, and fair.
