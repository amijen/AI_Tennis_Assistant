from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents 
from app.ingestion.embedder import embed_text
from app.db.database import engine 
from app.db.insert import insert_document, insert_chunk

from tqdm import tqdm 

def ingest(pdf_path, doc_name):
    print(f"\n{'='*50}")
    print(f"Ingesting: {doc_name}")
    print(f"File: {pdf_path}")
    print(f"{'='*50}")
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    with engine.begin() as conn:
        doc_id = insert_document(conn, doc_name)
        print(f"Document registered with ID: {doc_id}")
        for chunk in tqdm(chunks):
            content = chunk.page_content
            page = chunk.metadata.get("page", 0)
            embedding = embed_text(content)
            insert_chunk(
                conn,
                doc_id,
                content, 
                page,
                "text",
                embedding
            )
    print(f"✅ {doc_name} ingested successfully!\n")
if __name__ == "__main__":
    ingest("data/raw/2026-rules-of-tennis-english.pdf", "ITF Rules")
    ingest("data/raw/grand-slam-rulebook-2026-f2.pdf", "Grand Slam Rules")