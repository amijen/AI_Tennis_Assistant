"""Verify the new chunking strategy worked correctly."""
from sqlalchemy import text
from app.db.database import engine


def verify():
    with engine.connect() as conn:
        # Count chunks
        stats = conn.execute(text("""
            SELECT 
                d.name,
                COUNT(DISTINCT p.id) AS parents,
                COUNT(c.id) AS children
            FROM documents d
            LEFT JOIN parent_chunks p ON p.document_id = d.id
            LEFT JOIN child_chunks c ON c.parent_id = p.id
            GROUP BY d.name
        """)).fetchall()
        
        print("\n📊 Ingestion Statistics:")
        print(f"{'Document':<25} {'Parents':<10} {'Children':<10}")
        print("-" * 50)
        for row in stats:
            print(f"{row.name:<25} {row.parents:<10} {row.children:<10}")
        
        # Show 3 sample parents
        print("\n\n🔍 Sample Parent Chunks:\n")
        samples = conn.execute(text("""
            SELECT d.name, p.metadata, LEFT(p.content, 300) AS preview
            FROM parent_chunks p
            JOIN documents d ON p.document_id = d.id
            ORDER BY RANDOM()
            LIMIT 3
        """)).fetchall()
        
        for s in samples:
            print(f"📄 {s.name}")
            print(f"   Metadata: {s.metadata}")
            print(f"   Preview: {s.preview}...")
            print("-" * 60)


if __name__ == "__main__":
    verify()