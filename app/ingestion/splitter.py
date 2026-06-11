"""
Parent-Child splitter — fixed sub_pattern to capture full header titles.
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
# CONFIGS
# ============================================================

ITF_CONFIG = {
    # Matches: "1. THE COURT", "31. PLAYER ANALYSIS TECHNOLOGY"
    "parent_pattern": re.compile(
        r'^(\d+\.\s+[A-Z][A-Z\s,/]+?)[ \t]*\n',
        re.MULTILINE
    ),
    "sub_pattern": None,
    "child_size": 400,
    "child_overlap": 80,
    "skip_pages": 4,
    "max_parent_chars": 4000,
}

GRAND_SLAM_CONFIG = {
    # Matches: "ARTICLE I", "ARTICLE IV"
    "parent_pattern": re.compile(
        r'^(ARTICLE\s+[IVX]+)',
        re.MULTILINE
    ),
    "sub_pattern": re.compile(
        r'^([A-Z]\.\s+[A-Z][A-Z &\'\'/,-]+?)[ \t]*\n',
        re.MULTILINE
    ),
    "child_size": 600,
    "child_overlap": 100,
    "skip_pages": 6,
    "max_parent_chars": 4000,
}


# ============================================================
# HELPERS
# ============================================================

def _clean_header(header: str) -> str:
    """Remove trailing whitespace, newlines, stray characters."""
    header = header.split("\n")[0]
    header = re.sub(r'\s+', ' ', header)
    return header.strip()


def _build_text_with_page_map(
    pages: list[dict],
    skip_pages: int = 0
) -> tuple[str, list[tuple[int, int]]]:
    full_text = ""
    page_boundaries = []
    skipped = 0

    for page_data in pages:
        if page_data["page"] <= skip_pages:
            skipped += 1
            continue
        page_boundaries.append((len(full_text), page_data["page"]))
        full_text += page_data["text"] + "\n\n"

    if skipped > 0:
        print(f"   ⏭Skipped {skipped} pages (covers/TOC)")

    return full_text, page_boundaries


def _find_page_for_position(
    char_position: int,
    page_boundaries: list
) -> int:
    current_page = page_boundaries[0][1] if page_boundaries else 0
    for boundary_pos, page_no in page_boundaries:
        if char_position >= boundary_pos:
            current_page = page_no
        else:
            break
    return current_page


# ============================================================
# SPLITTING
# ============================================================

def _split_by_pattern(
    text: str,
    pattern: re.Pattern,
    page_boundaries: list
) -> list[tuple[str, str, int]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    results = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        header = _clean_header(match.group(1))
        page = _find_page_for_position(start, page_boundaries)

        if chunk_text:
            results.append((header, chunk_text, page))

    return results


def _split_large_chunk(
    content: str,
    page: int,
    max_chars: int = 4000
) -> list[tuple[str, int]]:
    """
    Break an oversized chunk into smaller pieces.
    Tries multiple splitting strategies in order.
    """
    if len(content) <= max_chars:
        return [(content, page)]

    print(f"      Splitting large chunk ({len(content)} chars)...")

    # ── Strategy 1: split at numbered items "1.", "2.", "3." ──────────────
    numbered = re.compile(r'(?=\n\s*\d+\.\s+[A-Z])', re.MULTILINE)
    parts = numbered.split(content)
    if len(parts) > 1:
        chunks = _group_parts(parts, page, max_chars)
        if all(len(c[0]) <= max_chars for c in chunks):
            return chunks

    # ── Strategy 2: split at lettered items "a.", "b.", "c." ─────────────
    lettered = re.compile(r'(?=\n\s*[a-z]\.\s+[A-Z])', re.MULTILINE)
    parts = lettered.split(content)
    if len(parts) > 1:
        chunks = _group_parts(parts, page, max_chars)
        if all(len(c[0]) <= max_chars for c in chunks):
            return chunks

    # ── Strategy 3: split at double newlines (paragraphs) ────────────────
    parts = re.split(r'\n\n+', content)
    if len(parts) > 1:
        chunks = _group_parts(parts, page, max_chars)
        if all(len(c[0]) <= max_chars for c in chunks):
            return chunks

    # ── Strategy 4: hard split at max_chars boundary (last resort) ────────
    print(f"      Using hard split (no natural boundary found)")
    chunks = []
    while len(content) > max_chars:
        split_at = content.rfind("\n", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append((content[:split_at].strip(), page))
        content = content[split_at:].strip()
    if content:
        chunks.append((content, page))
    return chunks

def _group_parts(
    parts: list[str],
    page: int,
    max_chars: int
) -> list[tuple[str, int]]:
    """
    Group split parts into chunks that stay under max_chars.
    Accumulates parts until the budget is reached, then starts a new chunk.
    """
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append((current.strip(), page))
            current = part
        else:
            current += "\n\n" + part if current else part

    if current.strip():
        chunks.append((current.strip(), page))

    return chunks

# ============================================================
# MAIN PIPELINE
# ============================================================

def split_into_parent_chunks(
    pages: list[dict],
    doc_type: str
) -> list[ParentChunk]:

    config = ITF_CONFIG if doc_type == "ITF" else GRAND_SLAM_CONFIG
    max_chars = config["max_parent_chars"]

    full_text, page_boundaries = _build_text_with_page_map(
        pages, skip_pages=config["skip_pages"]
    )

    if not page_boundaries:
        raise RuntimeError(
            f"No pages left after skipping {config['skip_pages']} pages!"
        )

    main_chunks = _split_by_pattern(
        full_text, config["parent_pattern"], page_boundaries
    )

    if not main_chunks:
        print(f"   No headers found for {doc_type}. Using page fallback.")
        return [
            ParentChunk(
                content=p["text"],
                page=p["page"],
                metadata={"doc_type": doc_type, "type": "page_fallback"}
            )
            for p in pages if p["page"] > config["skip_pages"]
        ]

    parents = []

    # ── Grand Slam: Article → Sub-sections ───────────────────────────────
    if config["sub_pattern"]:
        for parent_idx, (main_header, main_content, main_page) in enumerate(main_chunks):

            sub_chunks = _split_by_pattern(
                main_content,
                config["sub_pattern"],
                [(0, main_page)]
            )

            if sub_chunks:
                for sub_idx, (sub_header, sub_content, _) in enumerate(sub_chunks):

                    # Split further if still too big
                    pieces = _split_large_chunk(sub_content, main_page, max_chars)

                    for piece_idx, (piece_content, piece_page) in enumerate(pieces):
                        parents.append(ParentChunk(
                            content=piece_content,
                            page=piece_page,
                            metadata={
                                "doc_type": doc_type,
                                "article": main_header,
                                "section": sub_header,
                                "parent_index": f"{parent_idx}.{sub_idx}.{piece_idx}",
                            }
                        ))
            else:
                # No sub-sections — split the whole article if needed
                pieces = _split_large_chunk(main_content, main_page, max_chars)
                for piece_idx, (piece_content, piece_page) in enumerate(pieces):
                    parents.append(ParentChunk(
                        content=piece_content,
                        page=piece_page,
                        metadata={
                            "doc_type": doc_type,
                            "article": main_header,
                            "parent_index": f"{parent_idx}.{piece_idx}",
                        }
                    ))

    # ── ITF: each numbered rule is one parent ─────────────────────────────
    else:
        for parent_idx, (header, content, page) in enumerate(main_chunks):
            pieces = _split_large_chunk(content, page, max_chars)
            for piece_idx, (piece_content, piece_page) in enumerate(pieces):
                parents.append(ParentChunk(
                    content=piece_content,
                    page=piece_page,
                    metadata={
                        "doc_type": doc_type,
                        "rule": header,
                        "parent_index": f"{parent_idx}.{piece_idx}",
                    }
                ))

    print(f"   Created {len(parents)} parent chunks for {doc_type}")
    return parents


def split_parent_into_children(
    parent: ParentChunk,
    doc_type: str
) -> list[ChildChunk]:
    config = ITF_CONFIG if doc_type == "ITF" else GRAND_SLAM_CONFIG

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["child_size"],
        chunk_overlap=config["child_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    children = []
    for i, text in enumerate(splitter.split_text(parent.content)):
        if not text.strip():
            continue
        children.append(ChildChunk(
            content=text.strip(),
            page=parent.page,
            metadata={**parent.metadata, "child_index": i}
        ))
    return children


def process_document(
    pages: list[dict],
    doc_type: str
) -> list[ParentChunk]:
    parents = split_into_parent_chunks(pages, doc_type)
    total_children = 0
    for parent in parents:
        parent.children = split_parent_into_children(parent, doc_type)
        total_children += len(parent.children)
    print(f"   Total: {len(parents)} parents, {total_children} children")
    return parents