from app.config import settings 
from app.ingestion.loader import load_pdf_as_pages
from app.ingestion.splitter import process_document 
from app.ingestion.embedder import embed_texts
from app.db.database import engine 
from app.db.insert import (
    insert_document, 
    insert_parent_chunk,
    insert_child_chunk,
    delete_document_by_name
)

from tqdm import tqdm

def ingest(pdf_path: str, doc_name: str, doc_type: str, skip_pages: int = 0):
    """
    Ingest a PDF into the database with parent-child chunks.
    If the document already exsits, it is deleted first.

    Args:
        pdf_path:  Path to the PDF file.
        doc_name:  Human-readable name (e.g. "ITF Rules").
        doc_type:  "ITF" or "Grand Slam" (drives the splitting strategy).
    """
    print(f"\n{'='*50}")
    print(f"Ingesting: {doc_name}")
    print(f"Type: {doc_type}")
    print(f"File: {pdf_path}")
    print(f"{'='*50}")

    # ── 1. Load PDF into page dicts ─────────────────────────
    pages = load_pdf_as_pages(pdf_path, skip_pages)

    # ── 2. Split: pages → parents → children ────────────────
    parents = process_document(pages, doc_type)

    # ── 3. Insert into DB ───────────────────────────────────
    with engine.begin() as conn:
        if delete_document_by_name(conn, doc_name): 
            print(f"Deleted old version of {doc_name} (CASCADED).")

        doc_id = insert_document(conn, doc_name)
        print(f"Document registered with ID: {doc_id}")

        total_children = 0

        for parent in tqdm(parents, desc="Inserting chunks"):
            # 3a. Insert parent (no embedding)
            parent_id = insert_parent_chunk(
                conn,
                document_id=doc_id,
                content=parent.content,
                page=parent.page,
                metadata=parent.metadata,
            )

            # 3b. Batch-embed all children at once 
            if parent.children:
                child_texts = [c.content for c in parent.children]
                child_embeddings = embed_texts(child_texts)

                for child, embedding in zip(parent.children, child_embeddings):
                    insert_child_chunk(
                        conn,
                        parent_id=parent_id,
                        document_id=doc_id,
                        content=child.content,
                        embedding=embedding,
                        metadata=child.metadata,
                    )
                    total_children += 1

    print(f"{doc_name} ingested successfully!\n")
    print(f"   {len(parents)} parent chunks, {total_children} child chunks\n")


if __name__ == "__main__":
    ingest(
        pdf_path="data/raw/2026-rules-of-tennis-english.pdf",
        doc_name="ITF Rules",
        doc_type="ITF",
        skip_pages = 4
    )
    ingest(
        pdf_path="data/raw/grand-slam-rulebook-2026-f2.pdf",
        doc_name="Grand Slam Rules",
        doc_type="Grand Slam",
        skip_pages = 6
    )