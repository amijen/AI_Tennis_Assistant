"""
Test the splitter on a small subset of pages to inspect chunking quality.
Shows every parent and child chunk in detail.
"""
from app.ingestion.loader import load_pdf_as_pages
from app.ingestion.splitter import process_document


def filter_pages(pages: list[dict], page_range: tuple[int, int]) -> list[dict]:
    """Keep only pages within the specified range (inclusive)."""
    start, end = page_range
    return [p for p in pages if start <= p["page"] <= end]


def inspect_chunks(parents: list, doc_name: str):
    """Pretty-print all parents and their children."""
    print(f"\n{'='*70}")
    print(f"📚 {doc_name}")
    print(f"   Total parents: {len(parents)}")
    print(f"   Total children: {sum(len(p.children) for p in parents)}")
    print(f"{'='*70}")
    
    for p_idx, parent in enumerate(parents, 1):
        print(f"\n{'─'*70}")
        print(f"🔵 PARENT #{p_idx} | Page {parent.page}")
        print(f"   Header: {parent.metadata.get('header', 'N/A')}")
        print(f"   Length: {len(parent.content)} chars")
        print(f"{'─'*70}")
        print(f"CONTENT:")
        print(parent.content[:500])
        if len(parent.content) > 500:
            print(f"... [truncated, total {len(parent.content)} chars]")
        
        print(f"\n   👶 CHILDREN ({len(parent.children)}):")
        for c_idx, child in enumerate(parent.children, 1):
            print(f"\n   ──── Child {c_idx} ({len(child.content)} chars) ────")
            print(f"   {child.content[:300]}")
            if len(child.content) > 300:
                print(f"   ... [truncated]")


def test_itf():
    print("\n" + "█"*70)
    print("█ TESTING ITF RULES (pages 7-8)")
    print("█"*70)
    
    # Load all pages
    all_pages = load_pdf_as_pages(
        "data/raw/2026-rules-of-tennis-english.pdf", 
        skip_pages=4
    )
    
    # Filter to just pages 7-8
    pages = filter_pages(all_pages, page_range=(7, 8))
    print(f"\n📊 Loaded {len(pages)} pages from range 7-8")
    
    if not pages:
        print("❌ No pages in range!")
        return
    
    # Show what we're working with
    for p in pages:
        print(f"   Page {p['page']}: {len(p['text'])} chars")
    
    # Process with splitter
    parents = process_document(pages, doc_type="ITF")
    
    # Inspect results
    inspect_chunks(parents, "ITF Rules — Pages 7-8")


def test_grand_slam():
    print("\n\n" + "█"*70)
    print("█ TESTING GRAND SLAM RULES (pages 7-8)")
    print("█"*70)
    
    # Load all pages
    all_pages = load_pdf_as_pages(
        "data/raw/grand-slam-rulebook-2026-f2.pdf", 
        skip_pages=6
    )
    
    # Filter to just pages 7-8
    pages = filter_pages(all_pages, page_range=(7, 8))
    print(f"\n📊 Loaded {len(pages)} pages from range 7-8")
    
    if not pages:
        print("❌ No pages in range!")
        return
    
    for p in pages:
        print(f"   Page {p['page']}: {len(p['text'])} chars")
    
    # Process with splitter
    parents = process_document(pages, doc_type="GRAND_SLAM")
    
    inspect_chunks(parents, "Grand Slam Rules — Pages 7-8")


if __name__ == "__main__":
    test_itf()
    test_grand_slam()
    
    print("\n\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    print("\nWhat to look for:")
    print("  ✅ Each parent should be a complete rule/article")
    print("  ✅ Children should be readable, coherent chunks")
    print("  ✅ Page numbers should match the source")
    print("  ✅ Headers should be detected correctly")