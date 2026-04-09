import os
import fitz  # PyMuPDF
import re
from PIL import Image
import io
from openai import OpenAI
from api_client import generate_image_caption

def _extract_label_components(label: str):
    """Extract normalized visual type and id from a label string."""
    lower = label.lower() if label else ""
    visual_type = "figure" if "fig" in lower else ("table" if "tab" in lower or "tbl" in lower else "")

    # Prefer ids immediately following visual label words to avoid false matches
    # like the "i" in the word "Figure".
    scoped_match = re.search(
        r'(?i)\b(?:fig(?:ure)?|tab(?:le)?|tbl)\.?\s*([0-9]+(?:\.[0-9]+)?[A-Za-z]?|[IVXLCDM]+)\b',
        label or ""
    )
    if scoped_match:
        return visual_type, scoped_match.group(1).rstrip(').:-')

    # Fallbacks for noisy label strings.
    digit_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?[A-Za-z]?)\b', label or "")
    if digit_match:
        return visual_type, digit_match.group(1).rstrip(').:-')

    roman_match = re.search(r'\b([IVXLCDM]+)\b', label or "")
    number = roman_match.group(1).rstrip(').:-') if roman_match else ""
    return visual_type, number

def normalize_label(label: str) -> list:
    """Takes a label like 'Figure 1' or 'Table II' and returns regex variations."""
    visual_type, num = _extract_label_components(label)
    if not num:
        return [label.lower()]
    num_esc = re.escape(num)

    # Allow optional spaces/punctuation: Fig 3, Fig.3, Figure 3, Table II, Tbl. 1, etc.
    if visual_type == "figure":
        return [
            rf"\bfigure\s*{num_esc}(?:\b|\))",
            rf"\bfig(?:ure)?s?\s*\.?\s*{num_esc}(?:\b|\))",
        ]
    elif visual_type == "table":
        return [
            rf"\btable\s*{num_esc}(?:\b|\))",
            rf"\btab(?:le)?s?\s*\.?\s*{num_esc}(?:\b|\))",
            rf"\btbls?\s*\.?\s*{num_esc}(?:\b|\))",
        ]
    return [label.lower()]

def search_relevant_paragraphs(all_blocks: list, label: str, caption_body: str = "") -> str:
    if not label or label == "Unknown":
        return "No specific label identified."
    patterns = normalize_label(label)
    combined_regex = re.compile('|'.join(patterns), re.IGNORECASE)
    visual_type, num = _extract_label_components(label)
    loose_regex = None
    if visual_type and num:
        # Fallback: type token appears close to number token (handles noisy extraction variants).
        num_esc = re.escape(num)
        if visual_type == "figure":
            loose_regex = re.compile(rf"\bfig(?:ure)?s?\b[^\n]{{0,20}}\b{num_esc}(?:\b|\))", re.IGNORECASE)
        elif visual_type == "table":
            loose_regex = re.compile(rf"\b(?:table|tab|tbl)s?\b[^\n]{{0,20}}\b{num_esc}(?:\b|\))", re.IGNORECASE)

    relevant = []
    caption_clean = re.sub(r'\s+', ' ', (caption_body or "").strip()).lower()
    for text in all_blocks:
        clean_text = text.replace('\n', ' ').replace('\xa0', ' ')
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text.split()) <= 2:
            continue

        # Skip the caption line itself when searching for contextual references.
        if caption_clean and clean_text.lower() == caption_clean:
            continue

        if combined_regex.search(clean_text) or (loose_regex and loose_regex.search(clean_text)):
            relevant.append(clean_text)

    # De-duplicate while preserving order.
    relevant = list(dict.fromkeys(relevant))

    if not relevant:
        return f"No deep paragraphs explicitly mentioning '{label}' found."
    return "\n...\n".join(relevant)

def _is_heading_like(line: str) -> bool:
    """Heuristic heading detector so titles/subtitles are kept as boundaries."""
    stripped = line.strip()
    if not stripped:
        return False

    # Common academic section titles
    if re.match(r'^(abstract|introduction|background|related work|method|methods|materials|results|discussion|conclusion|references|appendix)\b', stripped, re.IGNORECASE):
        return True

    # Numbered headings (e.g., "2", "2.1", "III.")
    if re.match(r'^(\d+(\.\d+)*|[IVXLCDMivxlcdm]+)\.?\s+[A-Za-z]', stripped):
        return True

    words = stripped.split()
    if len(words) <= 14:
        if stripped.isupper():
            return True
        # Heading-like line usually does not end with sentence punctuation
        if not re.search(r'[\.!?]$', stripped):
            capitalized_ratio = sum(1 for w in words if w[:1].isupper()) / max(1, len(words))
            if capitalized_ratio >= 0.6:
                return True

    return False

def extract_paragraphs_from_blocks(sorted_blocks: list) -> list:
    """
    Reconstruct paragraph-like units from reading-order blocks while preserving
    headings and avoiding cross-block merges.
    """
    paragraphs = []

    for block in sorted_blocks:
        display_text = block.get("text", "")
        if "[Visual Element:" in display_text:
            # Keep visual reports as dedicated chunkable paragraphs.
            visual_paragraph = re.sub(r'\s+', ' ', display_text).strip()
            if visual_paragraph:
                paragraphs.append(visual_paragraph)
            continue

        raw_text = block.get("raw_text", display_text)
        if not raw_text:
            continue

        normalized = raw_text.replace('\ufffd', '').replace('(cid:10)', ' ').replace('(cid:13)', ' ')
        
        # Never merge across distinct PyMuPDF blocks. Split only inside each block.
        # The intelligent \n\n breaks and bullet point logic will take it from here.
        block_parts = re.split(r'\n\s*\n+', normalized)

        for part in block_parts:
            lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
            if not lines:
                continue

            # If the first line is a heading, keep it as its own paragraph.
            first_line = re.sub(r'\s+', ' ', lines[0]).strip()
            if _is_heading_like(first_line):
                paragraphs.append(first_line)
                lines = lines[1:]
                if not lines:
                    continue

            cleaned = re.sub(r'\s+', ' ', " ".join(lines)).strip()
            if not cleaned:
                continue

            # Split inline section patterns such as:
            # "B. Fixed-Wing UAV The UAV flies ..."
            inline_heading_match = re.match(
                r'^((?:[A-Z]|[IVXLCDM]+)\.\s+[A-Za-z][A-Za-z0-9\-/]*(?:\s+[A-Za-z][A-Za-z0-9\-/]*){0,10})\s+(.+)$',
                cleaned
            )
            if inline_heading_match and _is_heading_like(inline_heading_match.group(1)):
                paragraphs.append(inline_heading_match.group(1).strip())
                cleaned = inline_heading_match.group(2).strip()
                if not cleaned:
                    continue

            # Keep bullets as separate chunkable paragraph units.
            bullet_parts = re.split(r'\s(?=•\s)', cleaned)
            for bp in bullet_parts:
                bp_clean = re.sub(r'\s+', ' ', bp).strip()
                if bp_clean:
                    paragraphs.append(bp_clean)

    # Pass everything through a global stitcher to glue cross-column and cross-page breaks
    return _post_process_and_stitch_paragraphs([p for p in paragraphs if p])

def _post_process_and_stitch_paragraphs(paragraphs: list) -> list:
    """Stitches cross-column/cross-page paragraph breaks safely."""
    if not paragraphs: return []
    merged = []
    
    for p in paragraphs:
        if not merged:
            merged.append(p)
            continue
            
        prev = merged[-1]
        
        if "[Visual Element:" in prev or "[Visual Element:" in p:
            merged.append(p)
            continue
            
        # Academic hyphenation recovery across columns: e.g. "com-\nputer"
        if re.search(r'[A-Za-z]-$', prev) and re.match(r'^[a-z]', p):
            merged[-1] = prev[:-1] + p
        # Broken mid-sentence across columns: previous doesn't end with sentence terminator, next starts with lowercase
        elif re.search(r'[^.!?\]\)"\']$', prev) and re.match(r'^[a-z0-9]', p):
            merged[-1] = prev + " " + p
        else:
            merged.append(p)
            
    return merged

def merge_visual_rects(page_rects, margin=15, vertical_margin=10):
    """Merges overlapping or nearby geometric bounding boxes to form single coherent visual areas."""
    merged = []
    for r in page_rects:
        if r.is_empty: continue
        r_exp = r + (-margin, -vertical_margin, margin, vertical_margin)
        intersected = []
        for i, m in enumerate(merged):
            m_exp = m + (-margin, -vertical_margin, margin, vertical_margin)
            if r_exp.intersects(m_exp):
                intersected.append(i)
        
        if not intersected:
            merged.append(r)
        else:
            new_r = r
            for i in sorted(intersected, reverse=True):
                new_r = new_r | merged.pop(i)
            merged.append(new_r)
    return merged

def sort_text_blocks(blocks, page_width):
    """
    Topological block sort handling pure 1-column vs 2-column layouts seamlessly.
    It reads full-width abstracts first, then correctly sequences the left column followed by the right column.
    """
    def get_col(bbox):
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2
        # If block spans > 55% of the page width, it's a full-width section
        is_wide = (x1 - x0) > (page_width * 0.55)
        if is_wide: return 0 
        return 1 if cx < page_width / 2 else 2
        
    blocks = sorted(blocks, key=lambda b: b['bbox'][1])
    regions = []
    current_region = []
    
    for b in blocks:
        c = get_col(b['bbox'])
        if c == 0:
            if current_region:
                regions.append(current_region)
                current_region = []
            regions.append([b])
        else:
            current_region.append(b)
    if current_region:
        regions.append(current_region)
        
    sorted_blocks = []
    for region in regions:
        if not region: continue
        if len(region) == 1 and get_col(region[0]['bbox']) == 0:
            sorted_blocks.append(region[0])
        else:
            left = sorted([b for b in region if get_col(b['bbox']) == 1], key=lambda b: b['bbox'][1])
            right = sorted([b for b in region if get_col(b['bbox']) == 2], key=lambda b: b['bbox'][1])
            sorted_blocks.extend(left)
            sorted_blocks.extend(right)
            
    return sorted_blocks

def get_closest_visual(caption_bbox, visual_rects, label_type):
    """Pairs a caption to its structurally corresponding visual bounding box."""
    best_rect = None
    min_dist = float('inf')
    cx = (caption_bbox[0] + caption_bbox[2]) / 2
    
    for r in visual_rects:
        rcx = (r.x0 + r.x1) / 2
        # Ensure it's roughly in the same vertical column band, OR the visual is full width
        if abs(rcx - cx) < 250 or r.width > 350:
            # Both Figures and Tables are typically ABOVE their captions in this format.
            # Skip visuals that are entirely below the caption.
            if r.y0 > caption_bbox[3] + 10:
                continue
            dist = abs(caption_bbox[1] - r.y1)
                
            if dist < min_dist:
                min_dist = dist
                best_rect = r
                
    if min_dist > 150:
        return None
        
    return best_rect

def extract_pdf_data(upload_files, client: OpenAI, vision_model: str, progress_callback=None, process_visuals=False):
    """
    Advanced PDF Processor strictly rolled back to the stable topological sort.
    We dropped pdfplumber entirely for extreme speed, utilizing PyMuPDF table geometry.
    Fixed the sub-label text cutoff by precisely linking figure boundaries to their captions.
    """
    documents_data = {}  
    total_files = len(upload_files)
    
    save_dir = "extracted_visuals"
    os.makedirs(save_dir, exist_ok=True)
    save_text_dir = "extracted_texts"
    os.makedirs(save_text_dir, exist_ok=True)
    
    label_pattern = re.compile(r'(?i)^\s*(figure|fig\.?|table|tab\.?)\s+(\d+(?:\.\d+)?[A-Za-z]?|[IVXLCDMivxlcdm]+)(?=[\s\)\]:\.-]|$)(?::|\.|-|\))?\s*(.*)')

    for file_idx, uploaded_file in enumerate(upload_files):
        filename = uploaded_file.name
        
        if progress_callback:
            progress_callback(f"Processing '{filename}' ({file_idx + 1}/{total_files})")
            
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        page_payloads = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # --- 1. Gather all geometric visual regions with massive pure C speedups! ---
            raw_rects = []
            
            # 1a. Rasp Images
            for img in page.get_image_info(xrefs=True):
                raw_rects.append(fitz.Rect(img["bbox"]))
                
            # 1b. Vector Graphics (charts/graphs/schematics)
            for d in page.get_drawings():
                r = d["rect"]
                # Prevent massive invisible page-borders from mapping as an image
                if 10 < r.width < page.rect.width * 0.90 and 10 < r.height < page.rect.height * 0.40:
                    raw_rects.append(fitz.Rect(r))
                    
            # 1c. Native Fast Tables
            tabs = page.find_tables()
            if tabs.tables:
                for t in tabs.tables:
                    raw_rects.append(fitz.Rect(t.bbox))
                
            merged_visuals = merge_visual_rects(raw_rects, margin=15, vertical_margin=10)
            
            # --- 2. Extract Text & Clean Pollutant Labels ---
            # We use "dict" to preserve geometry accurately.
            page_dict = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)
            text_blocks = []
            
            for b in page_dict.get("blocks", []):
                if b["type"] == 0:  # Text block
                    raw_text = ""
                    lines = b.get("lines", [])
                    if lines:
                        block_x1 = max((l["bbox"][2] for l in lines), default=b["bbox"][2])
                        prev_line_bbox = None
                        prev_line_text = ""
                        for line in lines:
                            spans = line.get("spans", [])
                            if not spans: continue
                            
                            line_text = "".join(span["text"] for span in spans)
                            bbox = line["bbox"]
                            
                            if prev_line_bbox:
                                # Only a new line if it moved down (prevents superscripts/accents from splitting)
                                moved_down = bbox[1] > prev_line_bbox[1] + 5.0
                                
                                if moved_down:
                                    true_indent = bbox[0] > prev_line_bbox[0] + 5.0
                                    gap = bbox[1] - prev_line_bbox[3]
                                    line_height = prev_line_bbox[3] - prev_line_bbox[1]
                                    prev_short = prev_line_bbox[2] < block_x1 - 15.0
                                    
                                    is_para = False
                                    
                                    # 1. Definite large gap between paragraphs
                                    if gap > line_height * 0.5:
                                        is_para = True
                                        
                                    # 2. True indentation relative to the line before it
                                    if true_indent:
                                        pt = prev_line_text.strip()
                                        # Protect hanging indents from bullet lists/numbered lists
                                        if not re.match(r"^(?:\?|•|-|\d+\.|[IVXLCDMivxlcdm]+\.)", pt):
                                            is_para = True
                                            
                                    # 3. Previous line was noticeably short, implying sentence/paragraph end
                                    if prev_short and gap > 0 and bbox[0] >= prev_line_bbox[0] - 2.0:
                                        is_para = True
                                        
                                    if is_para:
                                        if not raw_text.endswith("\n\n"): raw_text += "\n\n"
                                    else:
                                        if not raw_text.endswith("\n"): raw_text += "\n"
                            
                                # If it didn't move down, it's part of the same physical line.
                            
                            raw_text += line_text
                            prev_line_bbox = bbox
                            prev_line_text = line_text
                    if raw_text:
                        if not raw_text.endswith("\n"):
                            raw_text += "\n"
                    text_blocks.append({"bbox": b["bbox"], "text": raw_text})

            clean_text_blocks = []
            for b in text_blocks:
                b_rect = fitz.Rect(b["bbox"])
                polluted = False
                for v_rect in merged_visuals:
                    intersect = b_rect.intersect(v_rect)
                    if not intersect.is_empty:
                        if intersect.get_area() > b_rect.get_area() * 0.5:
                            polluted = True
                            break

                if not polluted:
                    raw_text = b["text"].replace('\ufffd', '').replace('(cid:10)', ' ').replace('(cid:13)', ' ')
                    # Keep a cleaned one-line display text while preserving raw text for paragraph reconstruction.
                    text = re.sub(r'\s+', ' ', raw_text.replace('\n', ' ')).strip()
                    if text:
                        clean_text_blocks.append({
                            "bbox": b["bbox"], 
                            "text": text,
                            "raw_text": raw_text,
                            "is_caption": False
                        })
            
            page_payloads.append((page, clean_text_blocks, merged_visuals))

        # Reconstruct logical paragraphs once from ordered page blocks.
        all_paragraphs = []
        for page, text_blocks, _ in page_payloads:
            ordered_blocks = sort_text_blocks(text_blocks, page.rect.width)
            all_paragraphs.extend(extract_paragraphs_from_blocks(ordered_blocks))

        # --- 3. Link Captions, Sweep Orphaned Labels, Query Vision ---
        full_paper_formatted_text = f"--- START OF PAPER: {filename} ---\n\n"
        
        for page_num, (page, text_blocks, merged_visuals) in enumerate(page_payloads):
            
            for b in text_blocks:
                text_clean = b["text"].replace('\n', ' ')
                match = label_pattern.search(text_clean)
                
                if match:
                    label_type = match.group(1).lower()
                    label_num = match.group(2).rstrip(').:-')
                    caption_body = match.group(3).strip()
                    
                    canonical_label = f"Figure {label_num}" if "fig" in label_type else f"Table {label_num}"
                    
                    caption_rect = fitz.Rect(b["bbox"])
                    matched_visual = get_closest_visual(caption_rect, merged_visuals, label_type)
                    
                    # 1. Fallback for tables missed by native extraction
                    if not matched_visual and "tab" in label_type:
                        table_blocks_rect = fitz.Rect()
                        cx = (caption_rect.x0 + caption_rect.x1) / 2
                        
                        for tb in text_blocks:
                            if tb == b: continue
                            tb_rect = fitz.Rect(tb["bbox"])
                            tb_cx = (tb_rect.x0 + tb_rect.x1) / 2
                            # If text is ABOVE the caption and in same column natively
                            if tb_rect.y1 <= caption_rect.y0 + 10 and abs(tb_cx - cx) < 200:
                                # Stop at 150px to prevent capturing massive chunks of text (vertical ballooning)
                                if caption_rect.y0 - tb_rect.y1 > 150:
                                    continue
                                table_blocks_rect = table_blocks_rect | tb_rect
                                
                        if not table_blocks_rect.is_empty:
                            matched_visual = table_blocks_rect
                        else:
                            # Absolute geometry fallback (modest box directly above the caption instead of huge)
                            matched_visual = fitz.Rect(max(30, caption_rect.x0 - 20), max(30, caption_rect.y0 - 100), min(page.rect.width - 30, caption_rect.x1 + 20), caption_rect.y0)
                    
                    if matched_visual:
                        try:
                            # 10px boundary pad
                            final_rect = matched_visual + (-10, -10, 10, 10) 
                            
                            # CRITICAL FIX for the "CNN/LSTM text cutoff":
                            # We stretch the bounding box perfectly flush to the caption header 
                            # capturing all un-boxed sub-labels naturally sitting between.
                            # Both tables and figures are above their captions.
                            final_rect.y1 = max(final_rect.y1, caption_rect.y0 - 2)

                            # Problem 1: Crop full width pictures wider BUT keep tables constrained
                            # If a FIGURE spans a large segment horizontally, stretch it to the page margins
                            if "fig" in label_type and final_rect.width > page.rect.width * 0.65:
                                final_rect.x0 = 35 
                                final_rect.x1 = page.rect.width - 35
                            elif "tab" in label_type:
                                # Provide gentle padding for tables. Avoid blowing out horizontally to caption bounds
                                # if the visual is naturally narrower than the caption.
                                final_rect.x0 = final_rect.x0 - 10
                                final_rect.x1 = final_rect.x1 + 10
                                # Re-adjust just to make sure we at least cover the caption safely
                                if caption_rect.x0 < final_rect.x0:
                                    final_rect.x0 = caption_rect.x0 - 5
                                if caption_rect.x1 > final_rect.x1:
                                    final_rect.x1 = caption_rect.x1 + 5
                            else:
                                final_rect.x0 = min(final_rect.x0, caption_rect.x0 - 10)
                                final_rect.x1 = max(final_rect.x1, caption_rect.x1 + 10)

                            final_rect = final_rect.intersect(page.rect)
                            context_paragraphs = search_relevant_paragraphs(all_paragraphs, canonical_label, caption_body)
                            if context_paragraphs.startswith("No deep paragraphs explicitly mentioning"):
                                # Fallback to page-local paragraphs when global matching misses.
                                page_ordered_blocks = sort_text_blocks(text_blocks, page.rect.width)
                                page_paragraphs = extract_paragraphs_from_blocks(page_ordered_blocks)
                                page_context = search_relevant_paragraphs(page_paragraphs, canonical_label, caption_body)
                                if not page_context.startswith("No deep paragraphs explicitly mentioning"):
                                    context_paragraphs = page_context
                            
                            safe_filename = filename.replace(".pdf", "")
                            base_name = f"{safe_filename}_Page{page_num+1}_{canonical_label.replace(' ', '_')}"

                            if process_visuals:
                                pix = page.get_pixmap(clip=final_rect, dpi=200) 
                                image_bytes = pix.tobytes("jpeg")

                                with open(os.path.join(save_dir, f"{base_name}.jpg"), "wb") as f:
                                    f.write(image_bytes)

                                vision_prompt_instruction = f"The provided image may be imperfectly cropped and contain multiple figures, tables, or irrelevant text. Your STRICT objective is to locate and analyze ONLY the visual named '{canonical_label}' which matches this Original Caption: '{caption_body}'. Completely IGNORE any other graphs, tables, or text surrounding it. Provide a comprehensive summary of ONLY the target visual.\n\nContextual paragraphs from the paper for reference:\n{context_paragraphs}"
                                vision_description = generate_image_caption(client, vision_model, image_bytes, vision_prompt_instruction)

                                with open(os.path.join(save_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                                    f.write(f"--- Detected Label ---\n{canonical_label}: {caption_body}\n\n")
                                    f.write(f"--- Relevant Text Paragraphs ---\n{context_paragraphs}\n\n")
                                    f.write(f"--- Extracted Model Description ---\n{vision_description}\n")
                            else:
                                vision_description = "Visual AI processing disabled by user to conserve OpenRouter API limit. Visual files were not saved."
                            
                            b["text"] = (
                                f"\n\n=========================================\n"
                                f"[Visual Element: {canonical_label}]\n"
                                f"[Caption Extracted: {caption_body}]\n"
                                f"[Relevant Paragraphs: {context_paragraphs}]\n"
                                f"[Vision Model Comprehensive Analysis: {vision_description}]\n"
                                f"=========================================\n\n"
                            )
                            b["is_caption"] = True
                            if matched_visual in merged_visuals:
                                merged_visuals.remove(matched_visual)
                        except Exception as e:
                            print(f"[Warn] Render failed for {canonical_label}: {e}")

            # Sort intelligently using topological column algorithm (Fixes messy text)
            sorted_blocks = sort_text_blocks(text_blocks, page.rect.width)

            # Build page text as paragraph-separated units for better chunking quality.
            page_paragraphs = extract_paragraphs_from_blocks(sorted_blocks)
            page_text = "\n\n".join(page_paragraphs)
            full_paper_formatted_text += f"\n\n--- Page {page_num + 1} ---\n{page_text}\n\n"

        documents_data[filename] = full_paper_formatted_text
        
        safe_filename = filename.replace(".pdf", "")
        text_dump_path = os.path.join(save_text_dir, f"{safe_filename}_extracted_text.txt")
        try:
            with open(text_dump_path, "w", encoding="utf-8") as dump_f:
                dump_f.write(full_paper_formatted_text)
        except Exception as e:
            print(f"Skipped saving text dump: {e}")
            
    return documents_data