# 🎾 Tennis Rules AI Advisor

An **agentic RAG (Retrieval-Augmented Generation) system** that answers questions about tennis rules by intelligently retrieving, self-evaluating, and synthesizing information from the official **ITF Rules of Tennis** and the **Grand Slam Rulebook**.

## 🧠 Project Overview

This project implements a **LangGraph-based agentic pipeline** with self-correction capabilities. Unlike a simple RAG pipeline, the agent can **evaluate its own retrieval quality** and **reformulate queries** when the initial search returns irrelevant results.

Key capabilities:

- 🔍 **Parent-Child Retrieval** — Semantic search over small child chunks, returning full parent rule context
- 🧠 **Self-Grading** — The agent evaluates whether retrieved excerpts actually answer the question
- 🔄 **Self-Correction** — If retrieval is poor, the agent reformulates the query and retries automatically
- 💬 **Multi-Turn Memory** — Remembers conversation history and resolves follow-up references
- ⚡ **Token Streaming** — Real-time word-by-word response via Server-Sent Events
- 🏷️ **Smart Routing** — Keyword-based routing to ITF, Grand Slam, or both rulebooks
- 🛡️ **Source Grounding** — All answers cite exact document names and page numbers

## 🏗️ Architecture

```
User Question + Thread ID
        │
        ▼
┌──────────────────────┐
│ ROUTE & CONTEXTUALIZE│  Keyword routing + standalone query resolution
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│      RETRIEVE        │  Parent-child vector search (pgvector HNSW)
└──────────┬───────────┘
           ▼
┌──────────────────────┐     irrelevant + retry <= 2
│       GRADE          │──────────────────────────┐
│  (similarity gate    │                          ▼
│   + LLM fallback)    │                   ┌─────────────┐
└──────────┬───────────┘                   │ REFORMULATE │
           │ relevant                      │   (LLM)     │
           ▼                               └──────┬──────┘
┌──────────────────────┐                          │
│     SYNTHESIZE       │◄─────────────────────────┘
│  (LLM + citations)   │
└──────────┬───────────┘
           ▼
   Save to MemorySaver → Stream to Frontend
```

### What Makes This Agentic?

| Component             | Agentic? | Description                                      |
|-----------------------|----------|--------------------------------------------------|
| Route & Contextualize | Partial  | Resolves follow-up references using chat history |
| Retrieve              | No       | Deterministic vector search                      |
| Grade                 | **Yes**  | LLM evaluates its own retrieval quality          |
| Reformulate           | **Yes**  | LLM rewrites the query for better results        |
| Conditional Edge      | **Yes**  | Graph decides whether to retry or proceed        |
| Synthesize            | No       | Standard RAG generation                          |

## 📂 Project Structure

```
TENNIS/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Centralized env config (pydantic-settings)
│   ├── main.py                   # FastAPI app + lifespan + static serving
│   │
│   ├── api/                      # HTTP layer
│   │   ├── __init__.py
│   │   ├── routes.py             # POST /ask, POST /ask/stream, GET /health
│   │   └── schemas.py            # Pydantic request/response models
│   │
│   ├── graph/                    # LangGraph agentic pipeline
│   │   ├── __init__.py
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes.py              # Node functions (route, retrieve, grade, synthesize)
│   │   ├── tools.py              # Search + formatting utilities
│   │   ├── prompts.py            # All LLM prompts
│   │   └── builder.py            # StateGraph construction + MemorySaver
│   │
│   ├── db/                       # PostgreSQL + pgvector
│   │   ├── __init__.py
│   │   ├── database.py           # Engine + session factory
│   │   ├── insert.py             # Idempotent document/chunk insertion
│   │   ├── retriever.py          # Parent-child vector similarity search
│   │   └── schema.sql            # Tables + HNSW index
│   │
│   └── ingestion/                # PDF → Database pipeline
│       ├── __init__.py
│       ├── loader.py             # PyPDF page extraction
│       ├── splitter.py           # Parent-child regex splitting
│       └── embedder.py           # BGE embedding (local CPU)
│
├── scripts/
│   ├── ingest.py                 # python scripts/ingest.py
│   ├── evaluate.py               # python scripts/evaluate.py
│   └── list_models.py            # List available Groq models
│
├── data/
│   ├── raw/                      # Source PDFs
│   └── eval/
│       └── test_set.json         # Golden evaluation dataset
│
├── frontend/
│   └── index.html                # Single-file chat UI (no build step)
│
├── tests/
│   ├── test_config.py
│   ├── test_retriever.py
│   └── test_graph.py
│
├── .env                          # Secrets (gitignored)
├── .env.example                  # Template
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md
```

## ⚙️ Tech Stack

| Layer           | Technology                                                       |
|-----------------|------------------------------------------------------------------|
| Agent Framework | **LangGraph** (StateGraph + MemorySaver checkpointer)            |
| LLM             | **Groq** — `openai/gpt-oss-20b`                                  |
| Embeddings      | **HuggingFace** — `BAAI/bge-base-en-v1.5` (768 dims, local CPU)  |
| Vector Database | **PostgreSQL 15+** + **pgvector** (HNSW index)                   |
| Backend API     | **FastAPI** (sync + async streaming via SSE)                     |
| Frontend        | **Vanilla HTML/JS** (zero dependencies, served by FastAPI)       |
| PDF Parsing     | **PyPDF**                                                        |
| Chunking        | Parent-child strategy with regex-based structural splitting      |
| Config          | **pydantic-settings** (typed, validated, single source of truth) |
| Deployment      | **Docker** + **Docker Compose**                                  |

## 🚀 Getting Started

### Option 1 — Run with Docker (Recommended)

#### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- A free [Groq API key](https://console.groq.com)

#### 2. Clone the Repository
```bash
git clone https://github.com/amijen/AI_Tennis_Assistant.git
cd tennis
```

#### 3. Configure Environment Variables
Copy the template and fill in your values:
```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
DATABASE_URL=postgresql://postgres:<your_password>@db:5432/tennis_db
POSTGRES_USER=<your_postgres_user>
POSTGRES_PASSWORD=<your_password>
POSTGRES_DB=tennis_db
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8000
APP_ENV=development
```

#### 4. Place PDFs
Download the official rulebooks and place them in `data/raw/`:
- `2026-rules-of-tennis-english.pdf`
- `grand-slam-rulebook-2026-f2.pdf`

#### 5. Build and Launch
```bash
docker-compose up --build
```

Then:
- 🌐 **Chat UI** → http://localhost:8000
- 📘 **API Docs** → http://localhost:8000/docs
- ❤️ **Health Check** → http://localhost:8000/api/health

To stop: `docker-compose down`
To reset everything (wipe DB): `docker-compose down -v`

---

### Option 2 — Run Manually (Development)

#### 1. Prerequisites
- Python 3.14
- PostgreSQL 15+ with [`pgvector`](https://github.com/pgvector/pgvector) extension
- A free [Groq API key](https://console.groq.com)

#### 2. Clone & Setup
```bash
git clone https://github.com/amijen/AI_Tennis_Assistant.git
cd tennis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Configure
```bash
cp .env.example .env
# Edit .env with your Groq API key and database credentials
```

#### 4. Setup Database
```bash
psql -U postgres -c "CREATE DATABASE tennis_db;"
psql -U postgres -d tennis_db -f app/db/schema.sql
```

#### 5. Place PDFs in `data/raw/`

#### 6. Ingest Documents
```bash
python scripts/ingest.py
```

This parses the PDFs, splits them into parent-child chunks, generates 768-dim embeddings locally, and stores everything in PostgreSQL. The pipeline is **idempotent** — running it again safely replaces old data.

#### 7. Run the Server
```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser. That's it — the chat UI is served directly by FastAPI.

## 🛣️ Roadmap

- [x] Data ingestion pipeline (PyPDF loader, parent-child splitter, BGE embedder)
- [x] Vector database with pgvector + HNSW index + parent-child retrieval
- [x] LangGraph agentic pipeline with self-grading and self-correction
- [x] Keyword-based smart routing (ITF / Grand Slam / Both)
- [x] Multi-turn conversation memory (MemorySaver checkpointer)
- [x] Contextual query resolution for follow-up questions
- [x] Source-grounded answers with real page citations
- [x] SSE token streaming to frontend
- [x] FastAPI backend with async support
- [x] Single-file chat UI with step indicators
- [x] Centralized config with pydantic-settings
- [x] Idempotent ingestion pipeline
- [x] Evaluation framework with golden test set
- [x] Docker deployment

## 🙏 Acknowledgements

- [ITF Rules of Tennis 2026](https://www.itftennis.com) — Official rulebook
- [Grand Slam Rulebook 2026](https://www.grandslam.com) — Official rulebook
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration framework
- [Groq](https://groq.com) — Fast open-source LLM inference
- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity search for PostgreSQL
- Inspired by [RagUltimateAdvisor](https://github.com/dev-it-with-me/RagUltimateAdvisor)