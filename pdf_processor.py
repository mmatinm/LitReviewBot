import os
import fitz  # PyMuPDF
import re
from PIL import Image
import io
from openai import OpenAI
from api_client import generate_image_caption

def normalize_label(label: str) -> list:
    """Takes a label like 'Figure 1' or 'Table II' and returns regex variations."""
    num_match = re.search(r'\d+[\w\.]*|[IVXLCDMivxlcdm]+', label)
    if not num_match:
        return [label.lower()]
    num = num_match.group()
    if "fig" in label.lower():
        return [rf"\bfigure\s+{num}\b", rf"\bfig\.\s*{num}\b", rf"\bfig\s+{num}\b"]
    elif "tab" in label.lower():
        return [rf"\btable\s+{num}\b", rf"\btbl\.\s*{num}\b", rf"\btbl\s+{num}\b", rf"\btab\.\s*{num}\b", rf"\btab\s+{num}\b"]
    return [label.lower()]

def search_relevant_paragraphs(all_blocks: list, label: str) -> str:
    if not label or label == "Unknown":
        return "No specific label identified."
    patterns = normalize_label(label)
    combined_regex = re.compile('|'.join(patterns), re.IGNORECASE)
    relevant = []
    for text in all_blocks:
        if len(text.split()) > 5:
            clean_text = text.replace('\n', ' ')
            if combined_regex.search(clean_text):
                relevant.append(clean_text.strip())
    if not relevant:
        return f"No deep paragraphs explicitly mentioning '{label}' found."
    return "\n...\n".join(relevant)

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
    
    label_pattern = re.compile(r'(?i)^(figure|fig\.?|table|tab\.?)\s+(\d+[\w\.]*|[IVXLCDMivxlcdm]+)(?::|\.|-)?\s*(.*)')

    for file_idx, uploaded_file in enumerate(upload_files):
        filename = uploaded_file.name
        
        if progress_callback:
            progress_callback(f"Processing '{filename}' ({file_idx + 1}/{total_files})")
            
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_blocks_flat = []
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
            blocks = page.get_text("blocks", flags=fitz.TEXT_DEHYPHENATE)
            text_blocks = [{"bbox": b[:4], "text": b[4]} for b in blocks if b[-1] == 0]

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
                    # Clear out unnecessary enters and unknown unicode replacements
                    text = re.sub(r'\s+', ' ', b["text"].replace('\n', ' ').replace('\ufffd', '')).strip()
                    text = text.replace('(cid:10)', ' ').replace('(cid:13)', ' ')
                    if text:
                        clean_text_blocks.append({
                            "bbox": b["bbox"], 
                            "text": text,
                            "is_caption": False
                        })
                        all_blocks_flat.append(text)
            
            page_payloads.append((page, clean_text_blocks, merged_visuals))

        # --- 3. Link Captions, Sweep Orphaned Labels, Query Vision ---
        full_paper_formatted_text = f"--- START OF PAPER: {filename} ---\n\n"
        
        for page_num, (page, text_blocks, merged_visuals) in enumerate(page_payloads):
            
            for b in text_blocks:
                text_clean = b["text"].replace('\n', ' ')
                match = label_pattern.search(text_clean)
                
                if match:
                    label_type = match.group(1).lower()
                    label_num = match.group(2)
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
                            context_paragraphs = search_relevant_paragraphs(all_blocks_flat, canonical_label)
                            
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
            
            page_text = "\n\n".join([b["text"] for b in sorted_blocks])
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