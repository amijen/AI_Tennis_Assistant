# 🎾 Tennis Rules AI Advisor

An **agentic RAG (Retrieval-Augmented Generation) system** that answers questions about tennis rules by intelligently retrieving and comparing information from the official **ITF Rules of Tennis** and the **Grand Slam Rulebook**.

## 🧠 Project Overview
This project demonstrates an end-to-end **Agentic AI architecture** where an LLM-powered agent autonomously decides which tool to use to answer user questions:

- 🔍 **Retrieval Tool** — Semantic search over the official tennis rulebooks  
- ⚖️ **Compare Tool** — Highlights differences between ITF and Grand Slam rules  
- 🏷️ **Classification Tool** — Identifies whether a question concerns general ITF rules or Grand Slam–specific regulations  

## 🏗️ Architecture

## 📂 Project Structure
```
TENNIS/
├── app/
│   └──├── db/                  # Database, schema, retriever
│      ├── ingestion/           # PDF loader, splitter, embedder
│      └── agent/               # LangChain agent & tools
├── scripts/
│   └── ingest.py                # Ingestion pipeline
├── data/
│   ├── raw/                     # Source PDFs
│   └── processed/
├── frontend/                    # React UI (coming soon)
├── requirements.txt
└── README.md
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

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 14+ with [`pgvector`](https://github.com/pgvector/pgvector) extension
- A free [Groq API key](https://console.groq.com)

### 2. Clone the Repository

```bash
git clone https://github.com/amijen/AI_Tennis_Assistant.git
cd tennis
```

### 3. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file at the project root:

```env
DB_URL=postgresql://user:password@localhost:5432/tennis
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CLASSIF_MODEL=llama-3.1-8b-instant
```

### 5. Setup PostgreSQL Database

```bash
# Create the database
createdb tennis

# Apply the schema
psql -d tennis -f app/db/schema.sql
```

### 6. Place PDFs

Download the official rulebooks and place them in `data/raw/`:
- `2026-rules-of-tennis-english.pdf`
- `grand-slam-rulebook-2026-f2.pdf`

### 7. Run the Ingestion Pipeline

```bash
python -m scripts.ingest
```

This will parse the PDFs, split them into parent-child chunks, generate embeddings with `BAAI/bge-base-en-v1.5`, and store everything in PostgreSQL.

### 8. Test the Agent

```bash
python -m scripts.test_agent
```

### 9. Start the API (coming next)

```bash
uvicorn app.main:app --reload
```


## 🛣️ Roadmap

- [x] Data ingestion pipeline (PyPDF loader, parent-child splitter, HuggingFace embedder)
- [x] Vector database with pgvector + parent-child retrieval
- [x] Controlled router agent with 4 tools (search_itf, search_gs, compare, refuse)
- [x] LLM-powered query/topic extraction with defensive cleanup
- [x] Source-grounded answers with real page citations
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Docker deployment

## 📚 Learning Goals

Key concepts explored:

- Agentic AI architectures with LangChain
- Retrieval-Augmented Generation (RAG) with parent-child chunking
- Vector similarity search with pgvector
- Local LLM inference with Groq
- Local embedding inference with HuggingFace `sentence-transformers`
- LangChain Expression Language (LCEL) chains
- Full-stack AI application development

## 🙏 Acknowledgements

Inspired by [RagUltimateAdvisor](https://github.com/dev-it-with-me/RagUltimateAdvisor).
