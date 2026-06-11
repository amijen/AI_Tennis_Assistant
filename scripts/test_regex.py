"""Test if our regex patterns match the actual PDF content."""
import re
from app.ingestion.loader import load_pdf_as_pages


def test_pattern(doc_name: str, pdf_path: str, pattern: re.Pattern, skip_pages: int):
    print(f"\n{'='*70}")
    print(f"📄 Testing pattern on: {doc_name}")
    print(f"   Pattern: {pattern.pattern}")
    print(f"{'='*70}")
    
    pages = load_pdf_as_pages(pdf_path, skip_pages=skip_pages)
    
    # Concatenate all text
    full_text = "\n\n".join(p["markdown"] for p in pages)
    
    matches = list(pattern.finditer(full_text))
    
    print(f"\n✅ Found {len(matches)} matches:")
    for i, m in enumerate(matches[:20], 1):  # Show first 20
        print(f"   {i}. '{m.group(0).strip()[:60]}'")
    
    if len(matches) > 20:
        print(f"   ... and {len(matches) - 20} more")
    
    if len(matches) == 0:
        print("\n❌ NO MATCHES! Pattern needs adjustment.")
        print("\n   First 1000 chars of doc:")
        print(full_text[:1000])


# Test ITF
itf_pattern = re.compile(
    r'^(?:#+\s+)?(\d+\.\s+[A-Z][A-Z\s]+)', 
    re.MULTILINE
)
test_pattern(
    "ITF Rules",
    "data/raw/2026-rules-of-tennis-english.pdf",
    itf_pattern,
    skip_pages=4
)

# Test Grand Slam
gs_pattern = re.compile(
    r'^(?:#+\s+)?(ARTICLE\s+[IVX\d]+|Article\s+[IVX\d]+)', 
    re.MULTILINE | re.IGNORECASE
)
test_pattern(
    "Grand Slam Rules",
    "data/raw/grand-slam-rulebook-2026-f2.pdf",
    gs_pattern,
    skip_pages=6
)