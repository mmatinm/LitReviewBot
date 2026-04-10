import os
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which",
    "into", "about", "have", "has", "had", "were", "was", "are", "you", "your", "paper",
    "papers", "give", "show", "tell", "please", "using", "used", "than", "then", "them",
}


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


def _is_reference_query(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(r"\b(reference|references|bibliograph|citation|cited works|works cited)\b", q))


def _unique_doc_key(doc) -> tuple:
    meta = doc.metadata or {}
    source = meta.get("source", "")
    chunk_id = meta.get("chunk_id", None)
    # Keep a content fallback so dedupe still works if chunk_id is missing.
    content_head = (doc.page_content or "")[:120]
    return (source, chunk_id, content_head)


def _query_terms(query: str) -> list:
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", (query or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _term_hit_count(query: str, text: str) -> int:
    low = (text or "").lower()
    terms = _query_terms(query)
    if not terms:
        return 0
    return sum(1 for t in terms if t in low)


def _boilerplate_penalty(text: str) -> float:
    t = (text or "").strip().lower()
    penalty = 0.0
    if t.startswith("--- start of paper") or t.startswith("--- page"):
        penalty += 2.0
    if "image transcription failed" in t:
        penalty += 1.5
    if len(t) < 90:
        penalty += 0.8
    return penalty


def _dense_component(doc) -> float:
    score = (doc.metadata or {}).get("_dense_score", None)
    if score is None:
        return 0.0
    try:
        s = float(score)
    except Exception:
        return 0.0
    # FAISS returns distance-like values where smaller is better.
    return 1.0 / (1.0 + max(0.0, s))


def _rank_docs_for_query(query: str, docs: list) -> list:
    if not docs:
        return []

    is_ref = _is_reference_query(query)
    terms = _query_terms(query)

    def score(doc):
        txt = (doc.page_content or "").lower()
        bonus = 0.0
        bonus += _dense_component(doc)

        if terms:
            hits = sum(1 for t in terms if t in txt)
            bonus += min(4.0, 0.7 * hits)

        # Boost exact short phrase containment for intent-like questions.
        query_norm = " ".join(re.findall(r"[a-zA-Z0-9]+", (query or "").lower()))
        if query_norm and query_norm in " ".join(re.findall(r"[a-zA-Z0-9]+", txt)):
            bonus += 1.0

        if "references" in txt or "bibliography" in txt or "works cited" in txt:
            bonus += 3.0 if is_ref else -0.6
        if re.search(r"\[[0-9]+\]", txt):
            bonus += 0.8 if not is_ref else 1.0
        if re.search(r"\([12][0-9]{3}\)", txt):
            bonus += 0.6 if not is_ref else 1.0

        bonus -= _boilerplate_penalty(txt)
        return bonus

    return sorted(docs, key=score, reverse=True)


def _reference_lexical_scan(vector_store, source_filter: str = None, limit: int = 30) -> list:
    """Fallback lexical scan for references-style content in stored chunks."""
    results = []
    try:
        all_docs = list(vector_store.docstore._dict.values())
    except Exception:
        return results

    for d in all_docs:
        meta = d.metadata or {}
        if source_filter and meta.get("source") != source_filter:
            continue

        txt = (d.page_content or "")
        low = txt.lower()
        score = 0

        if "references" in low or "bibliography" in low or "works cited" in low:
            score += 4
        if re.search(r"\[[0-9]{1,3}\]", txt):
            score += 2
        if re.search(r"\([12][0-9]{3}\)", txt):
            score += 1
        if re.search(r"\bdoi\b|arxiv|ieee|springer|acm", low):
            score += 1

        if score > 0:
            results.append((score, d))

    results.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in results[:limit]]


def _query_lexical_scan(vector_store, query: str, source_filter: str = None, limit: int = 30) -> list:
    """Fallback lexical scan for generic questions when dense retrieval misses obvious term hits."""
    results = []
    try:
        all_docs = list(vector_store.docstore._dict.values())
    except Exception:
        return results

    for d in all_docs:
        meta = d.metadata or {}
        if source_filter and meta.get("source") != source_filter:
            continue

        txt = d.page_content or ""
        hits = _term_hit_count(query, txt)
        if hits <= 0:
            continue

        # Penalize boilerplate-like chunks even if they contain generic query terms.
        score = float(hits) - _boilerplate_penalty(txt)
        if score > 0:
            results.append((score, d))

    results.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in results[:limit]]


def retrieve_docs(vector_store, query: str, k: int = 8, source_filter: str = None) -> list:
    """Retrieve relevant docs with optional source filter and references-aware fallback."""
    if not vector_store:
        return []

    filter_dict = {"source": source_filter} if source_filter else None
    is_ref_query = _is_reference_query(query)
    primary_k = max(k, 20) if is_ref_query else k

    docs = []
    try:
        scored = vector_store.similarity_search_with_score(query, k=primary_k, filter=filter_dict)
        for d, score in scored:
            meta = dict(d.metadata or {})
            meta["_dense_score"] = float(score)
            d.metadata = meta
            docs.append(d)
    except Exception:
        docs = vector_store.similarity_search(query, k=primary_k, filter=filter_dict)

    if is_ref_query:
        # Add targeted fallback retrieval focused on reference sections.
        ref_queries = [
            f"{query} references bibliography",
            "references bibliography works cited",
        ]
        merged = {}
        for d in docs:
            merged[_unique_doc_key(d)] = d

        for rq in ref_queries:
            try:
                extra_scored = vector_store.similarity_search_with_score(rq, k=primary_k, filter=filter_dict)
                for d, score in extra_scored:
                    meta = dict(d.metadata or {})
                    meta["_dense_score"] = float(score)
                    d.metadata = meta
                    merged[_unique_doc_key(d)] = d
            except Exception:
                extra = vector_store.similarity_search(rq, k=primary_k, filter=filter_dict)
                for d in extra:
                    merged[_unique_doc_key(d)] = d

        lexical = _reference_lexical_scan(vector_store, source_filter=source_filter, limit=max(20, k * 4))
        for d in lexical:
            merged[_unique_doc_key(d)] = d

        docs = list(merged.values())

    docs = _rank_docs_for_query(query, docs)

    # For non-reference questions, prefer chunks with explicit lexical overlap.
    if not is_ref_query:
        with_hits = [d for d in docs if _term_hit_count(query, d.page_content) > 0]
        if with_hits:
            return with_hits[:k]

        lexical = _query_lexical_scan(vector_store, query=query, source_filter=source_filter, limit=max(20, k * 4))
        if lexical:
            merged = {}
            for d in docs:
                merged[_unique_doc_key(d)] = d
            for d in lexical:
                merged[_unique_doc_key(d)] = d
            reranked = _rank_docs_for_query(query, list(merged.values()))
            with_hits = [d for d in reranked if _term_hit_count(query, d.page_content) > 0]
            if with_hits:
                return with_hits[:k]

    return docs[:k]


def retrieve_context(vector_store, query: str, k: int = 5, source_filter: str = None) -> str:
    docs = retrieve_docs(vector_store, query=query, k=k, source_filter=source_filter)
    return "\n\n".join([doc.page_content for doc in docs])

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

        for chunk_id, c in enumerate(doc_chunks):
            chunks.append(c)
            metadatas.append({"source": filename, "chunk_id": chunk_id})

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
