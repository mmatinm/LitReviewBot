import os
import re
import math
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_TARGET_WORDS = 520
CHUNK_MAX_WORDS = 700
CHUNK_OVERLAP_WORDS = 80
DEFAULT_CANDIDATE_K = 30
DEFAULT_RERANK_K = 10

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which",
    "into", "about", "have", "has", "had", "were", "was", "are", "you", "your", "paper",
    "papers", "give", "show", "tell", "please", "using", "used", "than", "then", "them",
}


def _is_heading(text: str) -> bool:
    t = re.sub(r"^#{1,6}\s+", "", (text or "").strip())
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
    t = re.sub(r"^#{1,6}\s+", "", (text or "").strip())
    if not t:
        return False

    if t.startswith("--- START OF PAPER") or t.startswith("--- Page"):
        return True

    if re.match(r'^(abstract|introduction|background|related work|method|methods|materials|results|discussion|conclusion|conclusions|future work|references|appendix)\b', t, re.IGNORECASE):
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
    if len(t.split()) > 80 or re.search(r"#page-\d+-\d+|\\\[[0-9]", t):
        return False
    symbols = len(re.findall(r'[=+\-*/^_∑Σ∫√≤≥≈≠∞]', t))
    brackets = len(re.findall(r'[()\[\]{}]', t))
    has_digit = any(ch.isdigit() for ch in t)
    if symbols >= 3 and symbols / max(1, len(t)) > 0.015:
        return True
    if has_digit and ("=" in t or brackets >= 2):
        return True
    return bool(re.search(r'\b(?:sin|cos|tan|log|min|max|arg)\b', t, re.IGNORECASE))


def _heading_label(text: str) -> str:
    """Return a stable section label from a Markdown or extracted heading."""
    return re.sub(r"^#{1,6}\s+", "", (text or "").strip()).strip()


def _split_to_paragraphs(text: str) -> list:
    """Split extracted paper text into paragraph units while preserving visual blocks."""
    raw = (text or "").replace("\r\n", "\n")
    # Preserve Markdown heading boundaries before whitespace normalization.
    raw = re.sub(r"(?m)^(#{1,6}\s+.+?)\s*$", r"\n\n\1\n\n", raw)
    # Marker can concatenate terminal section headings with the preceding
    # paragraph (for example, "... coverage. V. CONCLUSIONS REFERENCES").
    raw = re.sub(
        r"(?m)^((?:[IVXLCM]+\.\s+)?(?:CONCLUSIONS?|REFERENCES?))\s+",
        r"\1\n\n",
        raw,
    )
    # Marker may place terminal headings and the bibliography in one block.
    raw = re.sub(r"(?<!#)\s+((?:[IVXLCM]+\.\s+)?(?:CONCLUSIONS?|REFERENCES?|BIBLIOGRAPHY|WORKS CITED))\s+",
                 r"\n\n\1\n\n", raw)
    raw = re.sub(r"(?m)^\s*-\s*(?=<span[^>]*page-\d+-\d+[^>]*></span>)", "", raw)

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
    """Coalesce extraction fragments without combining normal paragraphs."""
    cleaned = []
    pending_prefix = []
    in_references = False

    def is_page_marker(text: str) -> bool:
        return text.startswith("--- START OF PAPER") or text.startswith("--- Page")

    def is_reference_heading(text: str) -> bool:
        return bool(re.match(
            r"^(?:#+\s*)?(?:references|bibliography|works cited)\b",
            text,
            re.IGNORECASE,
        ))

    for p in paragraphs:
        text = (p or "").strip()
        if not text:
            continue

        if is_reference_heading(text):
            if pending_prefix:
                cleaned.append(" ".join(pending_prefix))
                pending_prefix = []
            cleaned.append(text)
            in_references = True
            continue

        # Bibliography entries are intentionally kept independent.
        if in_references:
            cleaned.append(text)
            continue

        if is_page_marker(text):
            pending_prefix.append(text)
            continue

        if _is_strong_heading(text):
            if pending_prefix:
                cleaned.append(" ".join(pending_prefix))
                pending_prefix = []
            cleaned.append(text)
            continue

        # Short lines at the start of a PDF are usually title/authors/metadata.
        # Keep them together, but do not absorb the first real paragraph.
        if len(text) < 180:
            if cleaned and not _is_strong_heading(cleaned[-1]):
                cleaned[-1] = f"{cleaned[-1]}\n\n{text}"
            else:
                pending_prefix.append(text)
            continue

        if pending_prefix:
            cleaned.append(" ".join(pending_prefix))
            pending_prefix = []
        cleaned.append(text)

    if pending_prefix:
        if cleaned and not _is_strong_heading(cleaned[-1]):
            cleaned[-1] = f"{cleaned[-1]}\n\n{' '.join(pending_prefix)}"
        else:
            cleaned.append(" ".join(pending_prefix))
    return cleaned


def _split_long_paragraph(paragraph: str, max_chars: int = 2400) -> list:
    """Split only oversized paragraphs at sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
    pieces = []
    current = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > max_chars:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence) + (1 if current_len else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces or [paragraph]


def _page_label(text: str) -> str:
    match = re.search(
        r"(?:\{(\d+)\}|---\s*Page\s+|<!--\s*page\s*[:\-]\s*|page\s+)(\d+)?",
        text or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    marker_page = match.group(1) or match.group(2)
    return str(int(marker_page) + 1) if match.group(1) else marker_page


def _split_reference_entries(text: str) -> list[str]:
    """Split a bibliography block only at numbered-entry boundaries."""
    normalized = re.sub(r"(?<!\d)(\d)\s+(\d)\s+(\d)\s+(\d)(?=[.,)\s]|$)", r"\1\2\3\4", text or "")
    matches = list(re.finditer(
        r"(?=(?<!\d)(?:\\?\[\d{1,3}\\?\]|\d{1,3}[.)])\s+)",
        normalized,
    ))
    if not matches:
        return [normalized]
    entries = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        entry = normalized[match.start():end].strip()
        if entry:
            entries.append(entry)
    return entries


def _section_aware_chunks(text: str) -> list[dict]:
    """Build bounded windows while keeping headings, pages, and academic blocks intact."""
    paragraphs = _merge_short_paragraphs(_split_to_paragraphs(text))
    units = []
    section = ""
    # Marker anchors use zero-based internal page IDs; expose a human-readable
    # page value and carry the latest known page until the next anchor appears.
    page = "1"
    for paragraph in paragraphs:
        page_match = re.search(
            r"\{(\d+)\}|---\s*Page\s+(\d+)\s*(?:---)?",
            paragraph,
            re.IGNORECASE,
        )
        if page_match:
            page = str(int(page_match.group(1)) + 1) if page_match.group(1) else page_match.group(2)
            paragraph = f"{paragraph[:page_match.start()]} {paragraph[page_match.end():]}".strip()
            if not paragraph or re.fullmatch(r"[-=*_ ]+", paragraph):
                continue
        structural_text = re.sub(
            r"^(?:-\s*)?(?:<span\b[^>]*page-\d+-\d+[^>]*></span>\s*)+",
            "",
            paragraph,
            flags=re.IGNORECASE,
        ).strip()
        if re.match(r"^---\s*START OF PAPER", structural_text, re.IGNORECASE):
            paragraph = re.sub(r"^---\s*START OF PAPER[^-]*---\s*", "", paragraph, flags=re.IGNORECASE).strip()
            if not paragraph:
                continue
            structural_text = paragraph
        if _is_strong_heading(structural_text):
            heading = _heading_label(structural_text.splitlines()[0])
            section = heading
            if structural_text == heading or structural_text.lstrip("#").strip() == heading:
                continue
            paragraph = structural_text
        in_reference_section = bool(
            section and re.search(r"\b(references?|bibliography|works cited)\b", section, re.I)
        )
        if in_reference_section:
            reference_entries = _split_reference_entries(paragraph)
            if len(reference_entries) > 1 or (
                reference_entries and re.match(r"^(?:\\?\[\d{1,3}\\?\]|\d{1,3}[.)])\s+", reference_entries[0])
            ):
                for entry in reference_entries:
                    units.append({
                        "text": entry,
                        "section": section,
                        "page": page,
                        "content_type": "reference",
                    })
                continue
        is_reference = in_reference_section and bool(
            re.match(r"^\s*(?:\\?\[\d{1,3}\\?\]|\d{1,3}[.)])\s+", paragraph)
        )
        if in_reference_section:
            content_type = "reference"
        elif is_reference:
            content_type = "reference"
        elif re.search(r"<table\b|^\s*\|.*\|\s*$", paragraph, re.I | re.M) and (
            paragraph.count("|") >= 4 or paragraph.lower().count("<tr") >= 2
        ):
            content_type = "table"
        elif re.match(r"^\s*(?:figure|fig\.|table|diagram|picture)\s*\d*", paragraph, re.I):
            content_type = "caption"
        elif _is_formula_like(paragraph):
            content_type = "formula"
        else:
            content_type = "text"
        units.append({"text": paragraph, "section": section, "page": page, "content_type": content_type})

    chunks = []
    current = []
    current_words = 0
    chunk_index = 0

    def emit(items):
        nonlocal chunk_index
        if not items:
            return
        raw_text = "\n\n".join(item["text"] for item in items).strip()
        first = items[0]
        retrieval_prefix = " | ".join(
            value for value in [
                f"Section: {first['section']}" if first["section"] else "",
                f"Page: {first['page']}" if first["page"] else "",
                f"Type: {first['content_type']}",
            ] if value
        )
        types = {item["content_type"] for item in items}
        content_type = next(
            (candidate for candidate in ("reference", "table", "formula", "caption") if candidate in types),
            "text",
        )
        chunks.append({
            "text": raw_text,
            "embedding_text": f"{retrieval_prefix}\n\n{raw_text}",
            "section": first["section"],
            "page": first["page"],
            "content_type": content_type,
            "chunk_index": chunk_index,
        })
        chunk_index += 1

    for unit in units:
        words = unit["text"].split()
        unit_words = len(words)
        protected = unit["content_type"] in {"formula", "table", "caption", "reference"}
        section_changed = current and unit["section"] != current[-1]["section"]
        if section_changed:
            emit(current)
            current = []
            current_words = 0
        if current and (current_words + unit_words > CHUNK_TARGET_WORDS or
                        (current_words + unit_words > CHUNK_MAX_WORDS and protected)):
            emit(current)
            tail_words = [] if unit["content_type"] == "reference" or current[-1]["content_type"] == "reference" else current[-1]["text"].split()[-CHUNK_OVERLAP_WORDS:]
            overlap = [{**current[-1], "text": " ".join(tail_words)}] if tail_words else []
            overlap_words = len(tail_words)
            current = overlap
            current_words = overlap_words
        if unit_words > CHUNK_MAX_WORDS and not protected:
            for part in _split_long_paragraph(unit["text"], max_chars=CHUNK_MAX_WORDS * 5):
                part_unit = {**unit, "text": part}
                if current and current_words + len(part.split()) > CHUNK_MAX_WORDS:
                    emit(current)
                    current = []
                    current_words = 0
                current.append(part_unit)
                current_words += len(part.split())
        else:
            current.append(unit)
            current_words += unit_words
    emit(current)
    return chunks


def _combine_heading_units(paragraphs: list) -> list:
    """Attach standalone headings to the following paragraph for better embeddings."""
    combined = []
    index = 0
    while index < len(paragraphs):
        current = paragraphs[index]
        if (
            _is_strong_heading(current)
            and index + 1 < len(paragraphs)
            and not _is_strong_heading(paragraphs[index + 1])
        ):
            combined.append(f"{current}\n\n{paragraphs[index + 1]}")
            index += 2
            continue
        combined.append(current)
        index += 1
    return combined


def _extract_citation_ids(text: str) -> set[str]:
    """Extract numeric bracket citations and numeric reference labels."""
    cleaned = (text or "").replace("\\[", "[").replace("\\]", "]")
    return set(re.findall(r"(?<!\d)\[(\d{1,3})\](?!\()", cleaned))


def _build_reference_index(paragraphs: list) -> dict[str, str]:
    references = {}
    in_references = False
    for paragraph in paragraphs:
        if re.match(r"^(?:#+\s*)?(?:references|bibliography|works cited)\b", paragraph, re.IGNORECASE):
            in_references = True
            continue
        if not in_references:
            continue
        matches = list(re.finditer(
            r"(?<!\d)(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+",
            paragraph,
        ))
        for index, match in enumerate(matches):
            reference_id = match.group(1) or match.group(2)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(paragraph)
            references[reference_id] = paragraph[match.start():end].strip()
    return references


def _save_chunk_debug(
    debug_dir: str,
    filename: str,
    paragraphs: list,
    chunk_records: list,
) -> None:
    """Write the exact paragraph-to-embedding mapping for visual inspection."""
    os.makedirs(debug_dir, exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', filename)
    debug_path = os.path.join(debug_dir, f"{safe_name}_chunk_debug.txt")

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"File: {filename}\n")
        f.write(f"Paragraph count: {len(paragraphs)}\n")
        f.write(f"Embedding unit count: {len(chunk_records)}\n")
        f.write("Each EMBEDDING UNIT below is sent to the embedding model exactly as shown.\n\n")
        for record_index, record in enumerate(chunk_records, start=1):
            f.write(f"{'=' * 20} EMBEDDING UNIT {record_index} {'=' * 20}\n")
            f.write(f"Paragraph ID: {record['paragraph_id']}\n")
            f.write(f"Part: {record['part'] + 1}/{record['part_count']}\n")
            f.write(f"Page: {record.get('page') or '[unknown]'}\n")
            f.write(f"Section: {record['section'] or '[none]'}\n")
            f.write(f"Content type: {record.get('content_type', 'text')}\n")
            f.write(f"Citations: {', '.join(record['citations']) or '[none]'}\n")
            f.write(f"Reference paragraph: {record['is_reference']}\n\n")
            f.write(record.get("embedding_text", record["text"]))
            f.write("\n\n")


def _is_reference_query(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.search(
        r"\b(refs?|refrences?|references?|bibliograph|citation|cited works|works cited)\b",
        q,
    ))


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", (text or "").lower())


def _bm25_scores(query: str, docs: list) -> dict[int, float]:
    """Small in-memory BM25 implementation for the current FAISS document set."""
    query_terms = set(_query_terms(query))
    if not query_terms or not docs:
        return {}
    tokenized = [_tokenize(doc.page_content) for doc in docs]
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    document_frequency = {term: sum(term in tokens for tokens in tokenized) for term in query_terms}
    scores = {}
    for index, tokens in enumerate(tokenized):
        frequencies = {term: tokens.count(term) for term in query_terms}
        length_norm = len(tokens) / max(1.0, avg_len)
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            idf = math.log(1 + (len(docs) - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            score += idf * (frequency * 2.0) / (frequency + 1.2 * (0.75 + 0.25 * length_norm))
        scores[index] = score
    return scores


def _rerank_candidates(query: str, docs: list, limit: int) -> list:
    """Use the BGE cross-encoder on CPU when available; otherwise preserve hybrid order."""
    if not docs:
        return []
    try:
        from sentence_transformers import CrossEncoder
        model = getattr(_rerank_candidates, "_model", None)
        if model is False:
            return docs[:limit]
        if model is None:
            model = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                device="cpu",
                max_length=512,
                local_files_only=True,
            )
            _rerank_candidates._model = model
        scores = model.predict([(query, doc.page_content) for doc in docs], show_progress_bar=False)
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda item: float(item[0]), reverse=True)[:limit]]
    except (ImportError, OSError, RuntimeError, ValueError):
        _rerank_candidates._model = False
        return docs[:limit]


def _term_hit_count(query: str, text: str) -> int:
    low = (text or "").lower()
    terms = _query_terms(query)
    if not terms:
        return 0
    return sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", low))


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

    terms = _query_terms(query)

    def score(doc):
        txt = (doc.page_content or "").lower()
        bonus = 0.0
        bonus += _dense_component(doc)

        if terms:
            hits = sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", txt))
            bonus += min(4.0, 0.7 * hits)

        # Boost exact short phrase containment for intent-like questions.
        query_norm = " ".join(re.findall(r"[a-zA-Z0-9]+", (query or "").lower()))
        if query_norm and query_norm in " ".join(re.findall(r"[a-zA-Z0-9]+", txt)):
            bonus += 1.0

        if "references" in txt or "bibliography" in txt or "works cited" in txt:
            bonus += 0.4
        if re.search(r"\[[0-9]+\]", txt):
            bonus += 0.2
        if re.search(r"\([12][0-9]{3}\)", txt):
            bonus += 0.2

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


def _expand_citations(vector_store, selected: list, k: int) -> list:
    reference_index = getattr(vector_store, "_litreview_reference_index", {})
    if not reference_index:
        return selected
    citation_docs = []
    seen = set()
    for doc in selected:
        source = (doc.metadata or {}).get("source")
        for citation_id in _extract_citation_ids(doc.page_content):
            reference = reference_index.get(source, {}).get(citation_id)
            if reference and (source, citation_id) not in seen:
                citation_docs.append(
                    Document(
                        page_content=reference,
                        metadata={
                            "source": source,
                            "chunk_id": f"reference-{citation_id}",
                            "citation_expansion": True,
                            "citation_id": citation_id,
                        },
                    )
                )
                seen.add((source, citation_id))
    return selected + citation_docs[: min(6, k)]


def _include_neighbor_chunks(vector_store, selected: list, limit: int) -> list:
    """Add immediate same-paper neighbors without exceeding the requested context size."""
    by_source = {}
    try:
        all_docs = vector_store.docstore._dict.values()
    except AttributeError:
        return selected
    for doc in all_docs:
        meta = doc.metadata or {}
        by_source.setdefault(meta.get("source"), {})[meta.get("chunk_index")] = doc
    result = list(selected)
    seen = {_unique_doc_key(doc) for doc in result}
    for doc in selected:
        meta = doc.metadata or {}
        siblings = by_source.get(meta.get("source"), {})
        index = meta.get("chunk_index")
        if not isinstance(index, int):
            continue
        for neighbor_index in (index - 1, index + 1):
            neighbor = siblings.get(neighbor_index)
            if neighbor and _unique_doc_key(neighbor) not in seen and len(result) < limit:
                result.append(neighbor)
                seen.add(_unique_doc_key(neighbor))
    return result


def retrieve_docs(
    vector_store,
    query: str,
    k: int = DEFAULT_RERANK_K,
    source_filter: str = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    include_neighbors: bool = False,
) -> list:
    """Hybrid dense/BM25 retrieval followed by optional CPU cross-encoder reranking."""
    if not vector_store:
        return []

    filter_dict = {"source": source_filter} if source_filter else None
    primary_k = max(k, candidate_k)
    dense_docs = []
    try:
        scored = vector_store.similarity_search_with_score(query, k=primary_k, filter=filter_dict)
        for d, score in scored:
            meta = dict(d.metadata or {})
            meta["_dense_score"] = float(score)
            d.metadata = meta
            dense_docs.append(d)
    except Exception:
        dense_docs = vector_store.similarity_search(query, k=primary_k, filter=filter_dict)

    try:
        all_docs = list(vector_store.docstore._dict.values())
    except AttributeError:
        all_docs = dense_docs
    if source_filter:
        all_docs = [d for d in all_docs if (d.metadata or {}).get("source") == source_filter]
    bm25 = _bm25_scores(query, all_docs)
    lexical_docs = [doc for _, doc in sorted(
        ((score, doc) for index, score in bm25.items() for doc in [all_docs[index]] if score > 0),
        key=lambda item: item[0], reverse=True,
    )[:candidate_k]]
    merged = {}
    for doc in dense_docs + lexical_docs:
        merged[_unique_doc_key(doc)] = doc
    docs = _rank_docs_for_query(query, list(merged.values()))
    selected = _rerank_candidates(query, docs[:candidate_k], k)
    if include_neighbors:
        selected = _include_neighbor_chunks(vector_store, selected, k)
    return selected


def retrieve_context(vector_store, query: str, k: int = 5, source_filter: str = None) -> str:
    docs = retrieve_docs(vector_store, query=query, k=k, source_filter=source_filter)
    return "\n\n".join(_format_doc_for_context(doc) for doc in docs)


def _format_doc_for_context(doc) -> str:
    meta = doc.metadata or {}
    trace = " | ".join(
        value for value in [
            str(meta.get("source", "")),
            f"page {meta['page']}" if meta.get("page") else "",
            meta.get("section", ""),
            f"chunk {meta.get('chunk_id')}" if meta.get("chunk_id") is not None else "",
        ] if value
    )
    source_text = meta.get("raw_text", doc.page_content)
    return f"[Source: {trace}]\n{source_text}" if trace else source_text


def extract_global_paper_briefs(documents_data: dict, max_chars_per_paper: int = 2500) -> str:
    """
    Extract a high-level executive profile for every uploaded paper from its initial text.
    Captures the title, abstract, and core problem formulation so the LLM retains global
    awareness of all papers during synthesis.
    """
    if not documents_data:
        return "No paper texts available."
    briefs = []
    for filename, full_text in documents_data.items():
        cleaned = re.sub(r"^---\s*START OF PAPER[^-]*---\s*", "", full_text, flags=re.IGNORECASE).strip()
        intro_slice = cleaned[:max_chars_per_paper].strip()
        briefs.append(f"### Paper: {filename}\n{intro_slice}")
    return "\n\n".join(briefs)


def retrieve_balanced_review_context(
    vector_store,
    paper_names: list[str],
    query: str,
    target_k: int = 36,
    max_chars: int = 250000,
    prioritize_visuals: bool = False,
) -> str:
    """
    Balanced multi-paper retrieval for literature review sections.
    Guarantees that every uploaded paper contributes top relevant chunks,
    boosts visual evidence (tables/captions) if requested,
    and includes top global salient chunks.
    """
    if not vector_store or not paper_names:
        return ""

    num_papers = max(1, len(paper_names))
    per_paper_quota = max(3, min(8, target_k // num_papers))

    gathered_docs = []
    seen_keys = set()

    # 1. Balanced per-paper retrieval
    for paper in paper_names:
        paper_docs = retrieve_docs(
            vector_store,
            query=query,
            k=per_paper_quota,
            source_filter=paper,
            candidate_k=max(15, per_paper_quota * 3),
        )
        for d in paper_docs:
            k = _unique_doc_key(d)
            if k not in seen_keys:
                seen_keys.add(k)
                gathered_docs.append(d)

    # 2. Global salience retrieval across all papers
    global_docs = retrieve_docs(
        vector_store,
        query=query,
        k=min(12, target_k),
        source_filter=None,
        candidate_k=30,
    )
    for d in global_docs:
        k = _unique_doc_key(d)
        if k not in seen_keys:
            seen_keys.add(k)
            gathered_docs.append(d)

    # 3. If prioritize_visuals is enabled, prioritize tables, formulas, and visual captions
    if prioritize_visuals:
        def visual_sort_key(d):
            ctype = (d.metadata or {}).get("content_type", "text")
            return 0 if ctype in {"table", "caption", "formula"} else 1
        gathered_docs.sort(key=visual_sort_key)

    # 4. Format chunks with source provenance and enforce max_chars ceiling
    formatted_chunks = []
    total_chars = 0
    for doc in gathered_docs:
        formatted = _format_doc_for_context(doc)
        if total_chars + len(formatted) > max_chars:
            break
        formatted_chunks.append(formatted)
        total_chars += len(formatted)

    return "\n\n".join(formatted_chunks)

def initialize_vector_store(documents_data: dict, progress_callback=None):
    """
    Chunks the combined text and captions from multiple papers 
    and embeds them into a FAISS local vector store.
    """
    if progress_callback:
        progress_callback("Splitting documents paragraph-by-paragraph...")

    chunks = []
    metadatas = []
    reference_indexes = {}
    debug_dir = "extracted_texts"

    for filename, text in documents_data.items():
        chunks_for_paper = _section_aware_chunks(text)
        reference_indexes[filename] = _build_reference_index(_merge_short_paragraphs(_split_to_paragraphs(text)))
        chunk_records = []
        for chunk_info in chunks_for_paper:
            chunk = chunk_info["text"]
            citations = sorted(_extract_citation_ids(chunk))
            chunk_id = f"{filename}:{chunk_info['chunk_index']}"
            chunks.append(chunk_info["embedding_text"])
            metadatas.append({
                "source": filename,
                "title": filename,
                "chunk_id": chunk_id,
                "chunk_index": chunk_info["chunk_index"],
                "section": chunk_info["section"],
                "page": chunk_info["page"],
                "content_type": chunk_info["content_type"],
                "raw_text": chunk,
                "citations": citations,
                "is_reference": chunk_info["content_type"] == "reference",
            })
            chunk_records.append({
                "paragraph_id": chunk_id,
                "part": 0,
                "part_count": 1,
                "section": chunk_info["section"],
                "page": chunk_info["page"],
                "content_type": chunk_info["content_type"],
                "citations": citations,
                "is_reference": chunk_info["content_type"] == "reference",
                "text": chunk,
                "embedding_text": chunk_info["embedding_text"],
            })
        _save_chunk_debug(debug_dir, filename, [item["text"] for item in chunks_for_paper], chunk_records)
    
    if not chunks:
        if progress_callback:
            progress_callback("No valid text found to chunk.")
        return None
        
    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks locally using HuggingFaceEmbeddings...")
        
    # Using local embedding model so we don't rely on remote embedding APIs
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    vector_store._litreview_reference_index = reference_indexes
    
    if progress_callback:
        progress_callback("Vector store initialized successfully.")
        
    return vector_store
