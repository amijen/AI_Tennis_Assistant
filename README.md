# 🎾 Tennis Rules AI Advisor

An **agentic RAG (Retrieval-Augmented Generation) system** that answers questions about tennis rules by intelligently retrieving and comparing information from the official **ITF Rules of Tennis** and the **Grand Slam Rulebook**.

## 🧠 Project Overview
This project demonstrates an end-to-end **Agentic AI architecture** where an LLM-powered agent autonomously decides which tool to use to answer user questions:

- 🔍 **Retrieval Tools** — Semantic search over the ITF and Grand Slam rulebooks (separate tools per document for precise filtering)
- ⚖️ **Compare Tool** — Side-by-side comparison of rules between ITF and Grand Slam
- 🏷️ **Intent Classifier** — Routes the question to the appropriate tool (`search_itf`, `search_gs`, `compare`, or `refuse`)
- 🛡️ **Out-of-Scope Guard** — Refuses questions outside the rulebooks (e.g., tournament results, player history)

## 🏗️ Architecture

The agent uses a **Controlled Router pattern** instead of fully autonomous tool-calling for higher reliability with open-source LLMs:

```
User Question
     ↓
1. Intent Classification (LLM + keyword heuristics)
     ↓
2. Python Routing (deterministic if/elif)
     ↓
3. Tool Execution (search_itf | search_gs | compare_rules | refuse)
     ↓
4. Answer Synthesis (LLM with strict grounding to retrieved chunks)
     ↓
Final Answer + Source Citation (Document, page X)
```

## 📂 Project Structure
```
TENNIS/
├── app/
│   └──├── db/                  # Database, schema, retriever
│      ├── ingestion/           # PDF loader, splitter, embedder
│      ├── agent/               # LangChain router & tools
│      └── main.py              # FastAPI entry point
├── scripts/
│   ├── __init__.py
│   └── ingest.py               # Ingestion pipeline
├── data/
│   ├── raw/                    # Source PDFs
│   └── processed/
├── frontend/                   # React UI 
├── requirements.txt
├── Dockerfile                  # Backend image
├── docker-compose.yml          # Full stack orchestration
└── README.md                   # You are here ! 
```
## ⚙️ Tech Stack

| Layer            | Technology                                              |
|------------------|---------------------------------------------------------|
| LLM Orchestration| LangChain (LCEL chains)                                 |
| LLM (synthesis)  | Groq — `llama-3.3-70b-versatile`                        |
| LLM (classifier) | Groq — `llama-3.1-8b-instant`                           |
| Embeddings       | HuggingFace — `BAAI/bge-base-en-v1.5` (768 dims)        |
| Vector Database  | PostgreSQL + pgvector                                   |
| Backend API      | FastAPI                                                 |
| Frontend         | React                                                   |
| PDF Parsing      | PyPDF                                                   |
| Chunking         | Parent-child strategy with regex-based structure        |
| Deployment       | Docker + Docker Compose                                 |

## 🚀 Getting Started

You can run this project in two ways: **with Docker (recommended)** or **manually for development**.

### Option 1 — Run with Docker (one command, easiest)

#### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- A free [Groq API key](https://console.groq.com)

#### 2. Clone the Repository
```bash
git clone https://github.com/amijen/AI_Tennis_Assistant.git
cd tennis
```

#### 3. Configure Environment Variables
Create a `.env` file at the project root:
```env
DB_URL="postgresql://postgres:<your_password>@localhost:5432/tennis"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CLASSIF_MODEL=llama-3.1-8b-instant
```

#### 4. ⚠️ Set Your PostgreSQL Password
Open `docker-compose.yml` and **replace `<your-password>`** in both places with the same password of your choice:

```yaml
# In the db service:
POSTGRES_PASSWORD: <your-password>     # ← change this

# In the backend service:
DB_URL: postgresql://postgres:<your-password>@db:5432/tennis   # ← and this (same value!)
```

⚠️ **Both passwords MUST match**, otherwise the backend cannot connect to the database.

#### 5. Place PDFs
Download the official rulebooks and place them in `data/raw/`:
- `2026-rules-of-tennis-english.pdf`
- `grand-slam-rulebook-2026-f2.pdf`

#### 6. Build and Launch
```bash
docker-compose up --build
```

The first build takes 5–10 minutes (downloading images, building containers, ingesting documents). Then:
- 🌐 **Frontend** → http://localhost
- ⚡ **Backend API** → http://localhost:8000
- 📘 **API Docs** → http://localhost:8000/docs

To stop: `docker-compose down`  
To reset everything (wipe DB): `docker-compose down -v`

---

### Option 2 — Run Manually (for development)

#### 1. Prerequisites
- Python 3.10+
- PostgreSQL 14+ with [`pgvector`](https://github.com/pgvector/pgvector) extension
- Node.js 18+ (for the frontend)
- A free [Groq API key](https://console.groq.com)

#### 2. Clone the Repository
```bash
git clone https://github.com/amijen/AI_Tennis_Assistant.git
cd tennis
```

#### 3. Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
Create a `.env` file at the project root:
```env
DB_URL="postgresql://postgres:<your_password>@localhost:5432/tennis"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CLASSIF_MODEL=llama-3.1-8b-instant
```

#### 5. Setup PostgreSQL Database
```bash
# Create the database
createdb tennis

# Apply the schema
psql -d tennis -f app/db/schema.sql
```

#### 6. Place PDFs
Download the official rulebooks and place them in `data/raw/`:
- `2026-rules-of-tennis-english.pdf`
- `grand-slam-rulebook-2026-f2.pdf`

#### 7. Run the Ingestion Pipeline
```bash
python -m scripts.ingest
```

This parses the PDFs, splits them into parent-child chunks, generates embeddings with `BAAI/bge-base-en-v1.5`, and stores everything in PostgreSQL.

#### 8. Test the Agent
```bash
python -m scripts.test_agent
```

#### 9. Start the API
```bash
uvicorn app.main:app --reload
```

#### 10. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 to use the chat UI.


## 🛣️ Roadmap

- [x] Data ingestion pipeline (PyPDF loader, parent-child splitter, HuggingFace embedder)
- [x] Vector database with pgvector + parent-child retrieval
- [x] Controlled router agent with 4 tools (search_itf, search_gs, compare, refuse)
- [x] LLM-powered query/topic extraction with defensive cleanup
- [x] Source-grounded answers with real page citations
- [x] FastAPI backend
- [x] React frontend
- [x] Docker deployment
- [ ] Conversation memory (multi-turn dialogues)
- [ ] Prompt refinement and evaluation framework
- [ ] Re-ranking with cross-encoder for better retrieval
- [ ] Hybrid search (semantic + BM25 keyword)

## 📚 Learning Goals

Key concepts explored:

- Agentic AI architectures with LangChain
- Retrieval-Augmented Generation (RAG) with parent-child chunking
- Vector similarity search with pgvector
- Cloud LLM inference with Groq
- Local embedding inference with HuggingFace `sentence-transformers`
- LangChain Expression Language (LCEL) chains
- Right-sizing models per task (small for classification, large for synthesis)
- Anti-hallucination patterns (source grounding, refusal escape hatch)
- Containerized full-stack deployment with Docker Compose
- Full-stack AI application development

## 🙏 Acknowledgements

Inspired by [RagUltimateAdvisor](https://github.com/dev-it-with-me/RagUltimateAdvisor).
