import os
import io
import fitz  # PyMuPDF
import re
from PIL import Image
from openai import OpenAI
from api_client import generate_image_caption

def normalize_label(label: str) -> list:
    """Takes a label like 'Figure 1' and returns regex variations for searching mentions."""
    num_match = re.search(r'\d+[\w\.]*', label)
    if not num_match:
        return [label.lower()]
    
    num = num_match.group()
    if "fig" in label.lower():
        # Match 'figure 1', 'fig. 1', 'fig 1' and even 'figure 1a'
        return [rf"\bfigure\s+{num}\b", rf"\bfig\.\s*{num}\b", rf"\bfig\s+{num}\b"]
    elif "tab" in label.lower():
        # Match 'table 1', 'tbl. 1', 'tbl 1'
        return [rf"\btable\s+{num}\b", rf"\btbl\.\s*{num}\b", rf"\btbl\s+{num}\b"]
    
    return [label.lower()]

def search_relevant_paragraphs(all_blocks: list, label: str) -> str:
    """Searches ALL parsed text blocks across the entire paper for mentions of the figure."""
    if not label or label == "Unknown":
        return "No specific label identified."
        
    patterns = normalize_label(label)
    combined_regex = re.compile('|'.join(patterns), re.IGNORECASE)
    
    relevant_paragraphs = []
    
    for text in all_blocks:
        # Ignore extremely short strings that might just be the bolded figure title itself
        if len(text.split()) > 5:
            # We strip out newlines mathematically inside the block for regex matching
            clean_text = text.replace('\n', ' ')
            if combined_regex.search(clean_text):
                relevant_paragraphs.append(clean_text.strip())
                
    if not relevant_paragraphs:
        return f"No further deep paragraphs explicitly mentioning '{label}' found in the text."
        
    return "\n...\n".join(relevant_paragraphs)

def extract_pdf_data(upload_files, client: OpenAI, vision_model: str, progress_callback=None):
    """
    Parses PDFs using an intelligent layout-based extraction:
    1. Scans text specifically looking for lines starting with 'Figure X' or 'Table X' (i.e. Captions).
    2. Dynamically clips the page bounding box corresponding to that caption natively securing all charts/graphs/plots.
    3. Hunts for all paragraphs mentioning that exact label everywhere in the paper.
    4. Sends it to Vision model, saves the exact match, and neatly integrates the label inline.
    """
    documents_data = {}  
    total_files = len(upload_files)
    
    save_dir = "extracted_visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    # Regex looks for blocks starting with 'Fig. 1', 'Figure 2:', 'Table 1 -' etc.
    label_pattern = re.compile(r'(?i)^(figure|fig\.?|table|tab\.?)\s+(\d+[\w\.]*)(?::|\.|-)?\s*(.*)')

    for file_idx, uploaded_file in enumerate(upload_files):
        filename = uploaded_file.name
        
        if progress_callback:
            progress_callback(f"Processing '{filename}' ({file_idx + 1}/{total_files})")
            
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Pass 1: Extract all text blocks globally & layout maps
        all_blocks = []
        page_blocks_map = {} 
        
        full_paper_text = f"--- START OF PAPER: {filename} ---\n\n"
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if b['type'] == 0]
            
            # Sort blocks top-to-bottom for logical layout parsing
            text_blocks = sorted(text_blocks, key=lambda b: b['bbox'][1])
            
            page_text_blocks = []
            for b in text_blocks:
                # Rebuild text intelligently avoiding shattered spaces
                lines = []
                for l in b["lines"]:
                    lines.append(" ".join([s["text"] for s in l["spans"]]))
                text = " ".join(lines).strip()
                
                if text:
                    page_text_blocks.append({"bbox": b["bbox"], "text": text})
                    all_blocks.append(text)
                    
            page_blocks_map[page_num] = page_text_blocks
            page_full_text = "\n\n".join([b["text"] for b in page_text_blocks])
            full_paper_text += f"\n--- Page {page_num + 1} ---\n" + page_full_text + "\n"

        # Pass 2: Layout Parsing purely for missing/vector graphics matching captions
        visuals_extracted_text = "\n\n--- EXTRACTED FIGURES & TABLES ---\n\n"
        
        visual_count = 0
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            blocks = page_blocks_map[page_num]
            
            for i, b in enumerate(blocks):
                # Clean block mapping
                text_clean = b["text"].replace('\n', ' ')
                match = label_pattern.search(text_clean)
                
                if match:
                    label_type = match.group(1).lower()
                    label_num = match.group(2)
                    caption_body = match.group(3).strip()
                    
                    if "fig" in label_type:
                        canonical_label = f"Figure {label_num}"
                    else:
                        canonical_label = f"Table {label_num}"
                        
                    caption_bbox = b["bbox"] # (x0, y0, x1, y1)
                    clip_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                    
                    render_margin = 20
                    # Usually Figures are placed ABOVE their captions
                    if "fig" in label_type:
                        top_y = 0
                        if i > 0:
                            top_y = blocks[i-1]["bbox"][3] # bottom of previous text block
                            
                        # constrain height realistically so we don't grab half the page randomly
                        top_y = max(top_y, caption_bbox[1] - 450)
                        
                        if top_y >= caption_bbox[1] - render_margin:
                            top_y = max(0, caption_bbox[1] - 300)
                            
                        clip_rect = fitz.Rect(0, top_y, page.rect.width, caption_bbox[1])
                        
                    # Tables are usually placed BELOW their captions (or between)
                    else:
                        bottom_y = page.rect.height
                        if i < len(blocks) - 1:
                            bottom_y = blocks[i+1]["bbox"][1] # top of the subsequent text block
                            
                        bottom_y = min(bottom_y, caption_bbox[3] + 450)
                        
                        if bottom_y <= caption_bbox[3] + render_margin:
                            bottom_y = min(page.rect.height, caption_bbox[3] + 300)
                            
                        clip_rect = fitz.Rect(0, caption_bbox[3], page.rect.width, bottom_y)
                        
                    # CRITICAL FIX: Extract image by clipping the page area natively! 
                    # This captures ALL vectors, plots, lines and maps seamlessly as 1 image.
                    try:
                        pix = page.get_pixmap(clip=clip_rect, dpi=200) # High DPI for clarity
                        image_bytes = pix.tobytes("jpeg")
                        visual_count += 1
                        
                        # 3. Retrieve deep paragraphs referencing this explicit label throughout the paper
                        context_paragraphs = search_relevant_paragraphs(all_blocks, canonical_label)
                        
                        vision_prompt_context = f"Caption: {canonical_label} - {caption_body}\n\nMentions scattered in paper:\n{context_paragraphs}"
                        vision_description = generate_image_caption(client, vision_model, image_bytes, vision_prompt_context)
                        
                        # Save for user debugging
                        safe_filename = filename.replace(".pdf", "")
                        safe_label = canonical_label.replace(" ", "_")
                        base_name = f"{safe_filename}_{safe_label}"
                        
                        img_path = os.path.join(save_dir, f"{base_name}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                            
                        txt_path = os.path.join(save_dir, f"{base_name}.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(f"--- Detected Label ---\n{canonical_label}: {caption_body}\n\n")
                            f.write(f"--- Whole Paper Context Mentions ---\n{context_paragraphs}\n\n")
                            f.write(f"--- AI Vision Output ---\n{vision_description}\n")
                        
                        # Inline Label Injection: Ensuring the exact title flows natively into RAG bounds
                        visual_block = (
                            f"\n\n=========================================\n"
                            f"[Visual Element: {canonical_label}]\n"
                            f"[Caption Extracted: {caption_body}]\n"
                            f"[Vision Model Comprehensive Analysis: {vision_description}]\n"
                            f"=========================================\n\n"
                        )
                        visuals_extracted_text += visual_block
                        
                    except Exception as e:
                        print(f"Skipped rendering rect for {canonical_label}: {e}")

        # Stitch visuals to the end so chunking parses them flawlessly
        documents_data[filename] = full_paper_text + visuals_extracted_text
        
    return documents_data
