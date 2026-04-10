import os
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def _is_heading(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("--- START OF PAPER") or t.startswith("--- Page"):
        return True
    if re.match(r'^(abstract|introduction|background|related work|method|methods|results|discussion|conclusion|references|appendix)\b', t, re.IGNORECASE):
        return True
    if re.match(r'^(\d+(\.\d+)*|[IVXLCDM]+)\.?\s+[A-Za-z]', t, re.IGNORECASE):
        return True
    # Be conservative here: short caption/legend fragments from figures are common.
    return False


def _is_strong_heading(text: str) -> bool:
    """Detect heading lines that should start a fresh chunk."""
    t = (text or "").strip()
    if not t:
        return False

    if t.startswith("--- START OF PAPER") or t.startswith("--- Page"):
        return True

    if re.match(r'^(abstract|introduction|background|related work|method|methods|materials|results|discussion|conclusion|future work|references|appendix)\b', t, re.IGNORECASE):
        return True

    if re.match(r'^(\d+(\.\d+)*|[IVXLCDM]+)\.?\s+[A-Za-z]', t, re.IGNORECASE):
        return True

    words = t.split()
    if 2 <= len(words) <= 10 and t.isupper() and re.search(r'[A-Z]', t):
        return True

    return False


def _is_formula_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    symbols = len(re.findall(r'[=+\-*/^_∑Σ∫√≤≥≈≠∞]', t))
    brackets = len(re.findall(r'[()\[\]{}]', t))
    has_digit = any(ch.isdigit() for ch in t)
    if symbols >= 2:
        return True
    if has_digit and ("=" in t or brackets >= 2):
        return True
    return bool(re.search(r'\b(?:sin|cos|tan|log|min|max|arg)\b', t, re.IGNORECASE))


def _split_to_paragraphs(text: str) -> list:
    """Split extracted paper text into paragraph units while preserving visual blocks."""
    raw = (text or "").replace("\r\n", "\n")

    # Normalize known legacy placeholder outputs so retrieval quality does not degrade.
    raw = raw.replace(
        "[Vision Model Comprehensive Analysis: None]",
        "[Vision Model Comprehensive Analysis: [Image transcription failed: empty response from model]]",
    )

    # Keep visual annotation blocks as single units.
    marker = "<<VISUAL_BLOCK_SPLIT>>"
    raw = raw.replace("=========================================", marker)

    parts = re.split(r'\n\s*\n+', raw)
    paragraphs = []
    for p in parts:
        cleaned = re.sub(r'\s+', ' ', p).strip()
        if not cleaned:
            continue
        if marker in cleaned:
            visual_segments = [seg.strip() for seg in cleaned.split(marker) if seg.strip()]
            paragraphs.extend(visual_segments)
        else:
            paragraphs.append(cleaned)
    return paragraphs


def _merge_short_paragraphs(paragraphs: list) -> list:
    """Keep paragraph units intact and preserve order (no short-line concatenation)."""
    cleaned = []
    for p in paragraphs:
        text = (p or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _build_chunks_from_paragraphs(paragraphs: list, chunk_size: int = 1400, overlap_paragraphs: int = 0, min_chunk_chars: int = 380) -> list:
    """Pack paragraphs into chunks with paragraph-level overlap."""
    if not paragraphs:
        return []

    chunks = []
    i = 0
    while i < len(paragraphs):
        current = []
        current_len = 0
        j = i

        while j < len(paragraphs):
            p = paragraphs[j]
            if current and _is_strong_heading(p):
                # Keep section titles/titles at the start of their own chunks.
                break
            extra = len(p) + (2 if current else 0)
            # If the current chunk is still very short, allow one more paragraph
            # even if we cross target size slightly.
            if current and current_len + extra > chunk_size and current_len >= min_chunk_chars:
                break
            current.append(p)
            current_len += extra
            j += 1

        if not current:
            # Hard fallback when one paragraph is itself longer than chunk_size.
            p = paragraphs[i]
            chunks.append(p[:chunk_size])
            i += 1
            continue

        chunk_text = "\n\n".join(current)
        if chunks and len(chunk_text) < 120 and j < len(paragraphs):
            # Avoid emitting overlap-only or noise-sized chunks in the middle.
            i += 1
            continue

        chunks.append(chunk_text)
        i = max(i + 1, j - overlap_paragraphs)

    return chunks


def _save_chunk_preview(preview_dir: str, filename: str, paragraphs: list, chunks: list) -> None:
    os.makedirs(preview_dir, exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', filename)
    preview_path = os.path.join(preview_dir, f"{safe_name}_chunks_preview.txt")

    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"File: {filename}\n")
        f.write(f"Paragraph count after merge: {len(paragraphs)}\n")
        f.write(f"Chunk count: {len(chunks)}\n\n")
        for idx, chunk in enumerate(chunks, start=1):
            f.write(f"===== CHUNK {idx} =====\n")
            f.write(chunk)
            f.write("\n\n")

def initialize_vector_store(documents_data: dict, progress_callback=None):
    """
    Chunks the combined text and captions from multiple papers 
    and embeds them into a FAISS local vector store.
    """
    if progress_callback:
        progress_callback("Splitting documents paragraph-by-paragraph...")

    chunks = []
    metadatas = []
    preview_dir = "chunk_previews"

    for filename, text in documents_data.items():
        paragraphs = _split_to_paragraphs(text)
        merged_paragraphs = _merge_short_paragraphs(paragraphs)
        doc_chunks = _build_chunks_from_paragraphs(merged_paragraphs, chunk_size=1400, overlap_paragraphs=0)

        for c in doc_chunks:
            chunks.append(c)
            metadatas.append({"source": filename})

        try:
            _save_chunk_preview(preview_dir, filename, merged_paragraphs, doc_chunks)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Could not save chunk preview for {filename}: {e}")
    
    if not chunks:
        if progress_callback:
            progress_callback("No valid text found to chunk.")
        return None
        
    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks locally using HuggingFaceEmbeddings...")
        
    # Using local embedding model so we don't rely on remote embedding APIs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    
    if progress_callback:
        progress_callback("Vector store initialized successfully. Chunk previews saved in chunk_previews/.")
        
    return vector_store

def retrieve_context(vector_store, query: str, k: int = 5) -> str:
    """Helper to retrieve 'k' similar chunks."""
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
