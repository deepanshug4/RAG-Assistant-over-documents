"""
Document ingestion: load files (PDF / TXT) and split into overlapping
chunks. Chunking matters because embedding models have token limits, and
smaller chunks give more precise retrieval. Overlap prevents losing
context that straddles a chunk boundary.
"""

from pypdf import PdfReader


def load_text_from_pdf(file) -> str:
    """Extract text from a PDF file-like object."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_text_from_txt(file) -> str:
    """Read a plain-text file-like object."""
    raw = file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return raw


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size / overlap are in characters (not tokens) for simplicity and
    predictability. ~800 chars ≈ a couple of paragraphs — a good retrieval
    unit. Overlap of ~150 keeps continuity across boundaries.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, leaving overlap
    return chunks


def build_chunks(files) -> list[dict]:
    """
    Turn uploaded files into a list of chunk records:
      {"source": filename, "chunk_id": i, "text": "..."}
    Keeping the source + id lets us CITE where each answer came from.
    """
    records = []
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            text = load_text_from_pdf(f)
        else:
            text = load_text_from_txt(f)

        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": name, "chunk_id": i, "text": chunk})
    return records