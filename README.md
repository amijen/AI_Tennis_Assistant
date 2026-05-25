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
│   └── core/
│       ├── db/                  # Database, schema, retriever
│       ├── ingestion/           # PDF loader, splitter, embedder
│       └── agent/               # LangChain agent & tools
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

| Layer            | Technology                              |
|------------------|------------------------------------------|
| LLM Orchestration| LangChain                               |
| LLM              | Ollama (llama3)                         |
| Embeddings       | Ollama (nomic-embed-text, 768 dims)     |
| Vector Database  | PostgreSQL + pgvector                   |
| Backend API      | FastAPI                                 |
| Frontend         | React                                   |
| PDF Parsing      | PyPDFLoader                             |

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 14+ with [`pgvector`](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com) installed and running

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

### 4. Setup PostgreSQL Database

```bash
# Create the database
createdb tennis

# Apply the schema
psql -d tennis -f app/core/db/schema.sql
```

### 5. Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 6. Place PDFs

Download the official rulebooks and place them in `data/raw/`:
- `2026-rules-of-tennis-english.pdf`
- `grand-slam-rulebook-2026-f2.pdf`

### 7. Run the Ingestion Pipeline

```bash
python scripts/ingest.py
```

### 8. Start the API (coming next)

```bash
uvicorn app.main:app --reload
```

## 🛣️ Roadmap

- [x] Data ingestion pipeline (loader, splitter, embedder)
- [x] Vector database with pgvector
- [ ] Retrieval layer with semantic search
- [ ] LangChain agent with 3 tools
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Docker deployment

## 📚 Learning Goals

Key concepts explored:

- Agentic AI architectures with LangChain
- Retrieval-Augmented Generation (RAG)
- Vector similarity search with pgvector
- Local LLM inference with Ollama
- Full-stack AI application development

## 🙏 Acknowledgements

Inspired by [RagUltimateAdvisor](https://github.com/dev-it-with-me/RagUltimateAdvisor).
