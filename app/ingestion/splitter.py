"""
Parent-Child splitter with multi-level structure support.
- ITF: Splits by Rule (1. THE COURT, 2. PERMANENT FIXTURES, ...)
- Grand Slam: Splits by Article AND sub-section (ARTICLE I > A, B, C, ...)
"""
import re
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ParentChunk:
    content: str
    page: int
    metadata: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


@dataclass
class ChildChunk:
    content: str
    page: int
    metadata: dict = field(default_factory=dict)


# ============================================================
# DOCUMENT-SPECIFIC PATTERNS
# ============================================================

ITF_CONFIG = {
    # Matches: "1. THE COURT", "23. CONTINUOUS PLAY", etc.
    "parent_pattern": re.compile(
        r'^(?:#+\s+)?(\d+\.\s+[A-Z][A-Z\s]+)', 
        re.MULTILINE
    ),
    # No sub-section splitting for ITF — rules are short enough
    "sub_pattern": None,
    "child_size": 400,
    "child_overlap": 80,
}

GRAND_SLAM_CONFIG = {
    # Matches: "ARTICLE I", "ARTICLE XII"
    "parent_pattern": re.compile(
        r'^(?:#+\s+)?(ARTICLE\s+[IVX\d]+|Article\s+[IVX\d]+)', 
        re.MULTILINE | re.IGNORECASE
    ),
    # Matches sub-sections like "A.", "B.", "1.", "(a)", etc.
    # We'll split parents into sub-parents using these
    "sub_pattern": re.compile(
        r'^\s*([A-Z]\.\s+[A-Z])', 
        re.MULTILINE
    ),
    "child_size": 400,
    "child_overlap": 80,
}


# ============================================================
# PAGE-AWARE TEXT BUILDING
# ============================================================

def _build_text_with_page_map(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """
    Concatenate pages into single text, tracking which char position = which page.
    
    Returns:
        - full_text: str
        - page_boundaries: list of (char_position, page_number)
    """
    full_text = ""
    page_boundaries = []
    
    for page_data in pages:
        page_boundaries.append((len(full_text), page_data["page"]))
        full_text += page_data["text"] + "\n\n"
    
    return full_text, page_boundaries


def _find_page_for_position(char_position: int, page_boundaries: list) -> int:
    """Find which page a character position belongs to."""
    current_page = page_boundaries[0][1] if page_boundaries else 0
    for boundary_pos, page_no in page_boundaries:
        if char_position >= boundary_pos:
            current_page = page_no
        else:
            break
    return current_page


# ============================================================
# CORE SPLITTING LOGIC
# ============================================================

def _split_by_pattern(text: str, pattern: re.Pattern, page_boundaries: list) -> list[tuple[str, str, int]]:
    """
    Split text using a regex pattern. Returns list of (header, content, page).
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    
    results = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        chunk_text = text[start:end].strip()
        header = match.group(1).strip()
        page = _find_page_for_position(start, page_boundaries)
        
        if chunk_text:
            results.append((header, chunk_text, page))
    
    return results


def split_into_parent_chunks(pages: list[dict], doc_type: str) -> list[ParentChunk]:
    """
    Split pages into parent chunks. For Grand Slam, also splits into sub-sections.
    """
    config = ITF_CONFIG if doc_type == "ITF" else GRAND_SLAM_CONFIG
    
    full_text, page_boundaries = _build_text_with_page_map(pages)
    
    # First level: split by main pattern (Rules or Articles)
    main_chunks = _split_by_pattern(full_text, config["parent_pattern"], page_boundaries)
    
    if not main_chunks:
        print(f"   No headers found for {doc_type}. Using pages as fallback.")
        return [
            ParentChunk(
                content=p["text"],
                page=p["page"],
                metadata={"doc_type": doc_type, "type": "page_fallback"}
            )
            for p in pages
        ]
    
    parents = []
    
    # If sub_pattern exists (Grand Slam), split each main chunk further
    if config["sub_pattern"]:
        for parent_idx, (main_header, main_content, main_page) in enumerate(main_chunks):
            # Try to split into sub-sections
            sub_chunks = _split_by_pattern(
                main_content, 
                config["sub_pattern"], 
                [(0, main_page)]  # Sub-chunks inherit parent page roughly
            )
            
            if sub_chunks:
                # Use sub-sections as parents
                for sub_idx, (sub_header, sub_content, _) in enumerate(sub_chunks):
                    parents.append(ParentChunk(
                        content=sub_content,
                        page=main_page,
                        metadata={
                            "doc_type": doc_type,
                            "article": main_header,
                            "section": sub_header,
                            "parent_index": f"{parent_idx}.{sub_idx}",
                        }
                    ))
            else:
                # No sub-sections found — use the whole main chunk
                parents.append(ParentChunk(
                    content=main_content,
                    page=main_page,
                    metadata={
                        "doc_type": doc_type,
                        "article": main_header,
                        "parent_index": str(parent_idx),
                    }
                ))
    else:
        # ITF: no sub-section splitting, each rule is a parent
        for parent_idx, (header, content, page) in enumerate(main_chunks):
            parents.append(ParentChunk(
                content=content,
                page=page,
                metadata={
                    "doc_type": doc_type,
                    "rule": header,
                    "parent_index": str(parent_idx),
                }
            ))
    
    print(f"    Created {len(parents)} parent chunks for {doc_type}")
    return parents


def split_parent_into_children(parent: ParentChunk, doc_type: str) -> list[ChildChunk]:
    """Split a parent chunk into smaller child chunks for embedding."""
    config = ITF_CONFIG if doc_type == "ITF" else GRAND_SLAM_CONFIG
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["child_size"],
        chunk_overlap=config["child_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    child_texts = splitter.split_text(parent.content)
    
    children = []
    for i, text in enumerate(child_texts):
        if not text.strip():
            continue
        children.append(ChildChunk(
            content=text.strip(),
            page=parent.page,
            metadata={
                **parent.metadata,
                "child_index": i,
            }
        ))
    
    return children


def process_document(pages: list[dict], doc_type: str) -> list[ParentChunk]:
    """Full pipeline: pages → parents → parents with children."""
    parents = split_into_parent_chunks(pages, doc_type)
    
    total_children = 0
    for parent in parents:
        parent.children = split_parent_into_children(parent, doc_type)
        total_children += len(parent.children)
    
    print(f"   📊 Total: {len(parents)} parents, {total_children} children")
    return parents