from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_vector_store(documents_data: dict, progress_callback=None):
    """
    Chunks the combined text and captions from multiple papers 
    and embeds them into a FAISS local vector store.
    """
    all_text = ""
    for filename, text in documents_data.items():
        all_text += text + "\n\n"
        
    if progress_callback:
        progress_callback("Splitting documents into chunks...")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
    
    if not chunks:
        if progress_callback:
            progress_callback("No valid text found to chunk.")
        return None
        
    if progress_callback:
        progress_callback(f"Embedding {len(chunks)} chunks locally using HuggingFaceEmbeddings...")
        
    # Using local embedding model so we don't rely on remote embedding APIs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    if progress_callback:
        progress_callback("Vector store initialized successfully.")
        
    return vector_store

def retrieve_context(vector_store, query: str, k: int = 5) -> str:
    """Helper to retrieve 'k' similar chunks."""
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
