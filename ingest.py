"""One-time ingestion: PDF -> page-aware chunks -> embeddings -> Qdrant Cloud.

Run locally ONCE before submitting. The chatbot never runs this; it only reads
the collection this script populates.

    python ingest.py

Design decisions (justified in the README):
  * Page-aware splitting: text is chunked WITHIN each page, never across pages,
    so every chunk maps to exactly one page number -> unambiguous citations.
  * Front matter (title page + table of contents) is detected and skipped,
    because TOC lines match queries but contain no real answers.
  * Each chunk carries {source, page, section} metadata. `page` is the 1-based
    PDF index, which for this document equals the printed page number, so a
    cited page can be opened and verified directly.
  * Deterministic chunk IDs (uuid5) make re-ingestion an idempotent upsert
    instead of creating duplicates.
"""

from __future__ import annotations

import re
import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client.models import Distance

from src.config import get_settings
from src.vectorstore import get_embeddings

# Stable namespace so re-runs regenerate identical IDs for identical chunks.
_ID_NAMESPACE = uuid.UUID("a3f4c1e2-0b6d-4e7a-9c1f-2d8b6e5a4c30")

# This PDF's text extraction splits multi-digit numbers with spaces
# ("Chapter  1 2    Camera 79", page "13 0"), so digit groups below tolerate
# internal spaces and the id/title boundary is the run of 2+ spaces.
# Running header on content pages, e.g. "Chapter  5    Phone 45"
_HEADER_RE = re.compile(r"^\s*(Chapter|Appendix)\s+([0-9A-Z][0-9 ]*?)\s{2,}(.+?)\s+[0-9 ]+$")
# Chapter/appendix start page begins with a lone id, e.g. "1", "1 2", or "A"
_CHAPTER_ID_RE = re.compile(r"^([0-9][0-9 ]*|[A-Z])$")


def _collapse_digits(s: str) -> str:
    """'1 2' -> '12' (undo the PDF's intra-number spacing)."""
    return re.sub(r"\s+", "", s)


def _looks_like_toc(text: str) -> bool:
    """True if most non-empty lines start with a number (table-of-contents shape)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    numbered = sum(1 for ln in lines if re.match(r"^\s*\d", ln))
    return numbered / len(lines) > 0.6


def _parse_section(text: str) -> str | None:
    """Extract a human-readable section like 'Chapter 5: Phone' from page text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    # Case 1: standard running header on a normal content page.
    m = _HEADER_RE.match(lines[0])
    if m:
        return f"{m.group(1)} {_collapse_digits(m.group(2))}: {m.group(3).strip()}"
    # Case 2: chapter/appendix start page — lone id, then printed page no, then title.
    if _CHAPTER_ID_RE.match(lines[0]):
        cid = _collapse_digits(lines[0]) if lines[0][0].isdigit() else lines[0]
        kind = "Appendix" if cid.isalpha() else "Chapter"
        for ln in lines[1:]:
            if not re.match(r"^[0-9 ]+$", ln):  # first non-numeric line is the title
                return f"{kind} {cid}: {ln}"
    return None


def _strip_running_header(text: str) -> str:
    """Drop the repeated running-header line so it doesn't pollute embeddings."""
    lines = text.splitlines()
    if lines and _HEADER_RE.match(lines[0].strip()):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def load_pages(pdf_path: str) -> list[tuple[int, str, str | None]]:
    """Return [(page_number, body_text, section)] for content pages only.

    Front matter (short title page + TOC) at the start is skipped. page_number
    is 1-based and equals the printed page number for this document.
    """
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str, str | None]] = []
    in_content = False
    current_section: str | None = None

    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        raw = page.extract_text() or ""

        # Skip leading front matter: short title page and TOC pages.
        if not in_content:
            if len(raw.strip()) < 100 or _looks_like_toc(raw):
                continue
            in_content = True  # first real content page reached

        section = _parse_section(raw) or current_section
        current_section = section
        body = _strip_running_header(raw)
        if body:
            pages.append((page_number, body, section))

    return pages


def build_chunks(pages, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split each page independently so no chunk ever crosses a page boundary."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph/line/sentence breaks
    )
    docs: list[Document] = []
    for page_number, body, section in pages:
        for i, piece in enumerate(splitter.split_text(body)):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": "iPhone User Guide (iOS 7.1)",
                        "page": page_number,
                        "section": section,
                        "chunk": i,
                    },
                )
            )
    return docs


def _deterministic_id(doc: Document) -> str:
    key = f"{doc.metadata['page']}:{doc.metadata['chunk']}:{doc.page_content}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def main() -> None:
    settings = get_settings()

    print(f"Loading PDF: {settings.pdf_path}")
    pages = load_pages(settings.pdf_path)
    print(f"  content pages kept: {len(pages)}")

    chunks = build_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"  chunks produced:    {len(chunks)}")
    print(
        f"  avg chars/chunk:    "
        f"{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)}"
    )

    ids = [_deterministic_id(c) for c in chunks]

    print(
        f"Embedding with {settings.embedding_model} and upserting into "
        f"Qdrant collection '{settings.qdrant_collection}' (force_recreate=True)…"
    )
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        ids=ids,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        distance=Distance.COSINE,
        force_recreate=True,  # clean, reproducible index on every run
        batch_size=64,
    )
    print(f"Done. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    main()
