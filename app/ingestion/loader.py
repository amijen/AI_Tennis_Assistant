"""
PDF loader using PyPDF — reliable, lightweight, processes ALL pages.

Why PyPDF over Docling:
- Works on any machine (no memory issues)
- Extracts 100% of pages reliably
- Output quality is sufficient for our regex-based chunking
- No heavy ML model downloads
"""
import re
from pathlib import Path
from pypdf import PdfReader


def load_pdf_as_pages(pdf_path: str, skip_pages: int = 0) -> list[dict]:
    """
    Extract PDF content page by page using PyPDF.
    
    Args:
        pdf_path: Path to the PDF file
        skip_pages: Number of initial pages to skip (cover, TOC, etc.)
    
    Returns:
        List of dicts with keys: 'page', 'markdown' (cleaned text), 'source'
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"Reading {pdf_path.name} with PyPDF...")
    
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    
    if skip_pages > 0:
        print(f"   Total pages: {total_pages}, skipping first {skip_pages}")
    else:
        print(f"   Total pages: {total_pages}")
    
    pages = []
    for i, page in enumerate(reader.pages):
        page_no = i + 1  # 1-indexed (book convention)
        
        if page_no <= skip_pages:
            continue
        
        text = page.extract_text()
        if text and text.strip():
            cleaned = _clean_text(text)
            if cleaned:  # Skip if cleaning leaves nothing
                pages.append({
                    "page": page_no,
                    "text": cleaned,
                    "source": str(pdf_path)
                })
    
    if len(pages) == 0:
        raise RuntimeError(f"PyPDF extracted 0 pages from {pdf_path.name}!")
    
    print(f"   Extracted {len(pages)} content pages")
    return pages


def _clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts:
    - Collapse multiple spaces/tabs
    - Limit consecutive blank lines
    - Fix hyphenated line breaks (compli-\ncated → complicated)
    - Remove page numbers that appear alone on lines
    """
    # 1. Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. Collapse multiple spaces/tabs into single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3. Remove standalone page numbers (lines with just digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 4. Limit consecutive blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()