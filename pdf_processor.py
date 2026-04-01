import io
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from api_client import generate_image_caption

def find_image_context(page_text: str, radius: int = 500) -> str:
    """
    Naive extraction of context for an image. 
    It searches for indications of Figures/Tables in the current page.
    """
    # Simple heuristic: Return the entire page text as context to the vision model.
    # A more advanced version would look for regex patterns like "Figure \d" or "Table \d".
    return page_text

def extract_pdf_data(upload_files, client: OpenAI, vision_model: str, progress_callback=None):
    """
    Parses PDFs, extracts text, images, and injects captions into the text flow.
    """
    documents_data = {}  # {filename: formatted_text}
    
    total_files = len(upload_files)
    
    for file_idx, uploaded_file in enumerate(upload_files):
        filename = uploaded_file.name
        
        if progress_callback:
            progress_callback(f"Processing '{filename}' ({file_idx + 1}/{total_files})")
            
        # Read the PDF into PyMuPDF
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        full_document_text = f"--- START OF PAPER: {filename} ---\n\n"
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            
            # Context around images
            context = find_image_context(page_text)
            
            # Extract images
            image_list = page.get_images(full=True)
            
            # First, append the raw text of the page
            full_document_text += page_text + "\n"
            
            # Process and caption the images directly after the page text
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Verify format
                img_ext = base_image["ext"]
                if img_ext not in ["jpeg", "jpg", "png", "webp"]:
                    # Convert to JPG in memory
                    try:
                        temp_image = Image.open(io.BytesIO(image_bytes))
                        rgb_im = temp_image.convert('RGB')
                        b_io = io.BytesIO()
                        rgb_im.save(b_io, format='JPEG')
                        image_bytes = b_io.getvalue()
                    except Exception:
                        continue # Skip unprocessable image
                
                # Generate Context-Aware Caption
                caption = generate_image_caption(client, vision_model, image_bytes, context)
                
                # Inject caption into the flow
                caption_block = (
                    f"\n\n[IMAGE/TABLE EXTRACTED (Page {page_num + 1}, Item {img_index + 1})]\n"
                    f"[Model Generated Caption]: {caption}\n\n"
                )
                full_document_text += caption_block
        
        documents_data[filename] = full_document_text
        
    return documents_data
