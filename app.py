import streamlit as st
from config import VISION_MODELS, TEXT_MODELS
from api_client import get_openrouter_client, call_openrouter
from pdf_processor import extract_pdf_data
from vector_store import initialize_vector_store

# ==========================================
# Configuration & Setup
# ==========================================
st.set_page_config(page_title="Literature Review Bot", page_icon="📚", layout="wide")

# ==========================================
# UI Layout & Interactions
# ==========================================
def main():
    st.title("📚 Literature Review Bot")
    st.markdown("Upload research papers (PDFs) and let the bot extract insights, answer questions, and even draft literature reviews for you!")
    
    # ------------------
    # Sidebar
    # ------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("OpenRouter API Key", type="password")
        if not api_key:
            st.warning("Please enter your OpenRouter API Key to proceed.")
            st.markdown("[Get one here](https://openrouter.ai/keys)")
        
        st.subheader("Model Selection")
        # Allow typing custom or selecting from preset
        vision_model_input = st.selectbox(
            "Vision Model (For Images/Tables)",
            options=VISION_MODELS,
            index=0,
            help="Select a vision-capable model or type a valid OpenRouter Vision model ID."
        )
        custom_vision = st.text_input("Or type custom Vision Model ID:")
        vision_model = custom_vision if custom_vision else vision_model_input

        text_model_input = st.selectbox(
            "Main Text Model (For Q&A/Reviews)",
            options=TEXT_MODELS,
            index=0,
            help="Select a text-capable model or type a valid OpenRouter Text model ID."
        )
        custom_text = st.text_input("Or type custom Text Model ID:")
        text_model = custom_text if custom_text else text_model_input
        
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        
        process_btn = st.button("process Papers")

    # State variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_data" not in st.session_state:
        st.session_state.documents_data = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # ------------------
    # Processing Phase
    # ------------------
    if process_btn:
        if not api_key:
            st.sidebar.error("API Key missing!")
        elif not uploaded_files:
            st.sidebar.error("Please upload at least one PDF.")
        else:
            client = get_openrouter_client(api_key)
            status_text = st.empty()
            
            def update_progress(msg: str):
                status_text.text(f"⏳ {msg}")
            
            with st.spinner("Step 1: Extracting text and identifying figures using Vision Model..."):
                # 1. Parse texts & images
                st.session_state.documents_data = extract_pdf_data(
                    uploaded_files, 
                    client, 
                    vision_model, 
                    progress_callback=update_progress
                )
                
            with st.spinner("Step 2: Chunking and Embedding into Vector Store..."):
                # 2. Chunk & Vectorize
                st.session_state.vector_store = initialize_vector_store(
                    st.session_state.documents_data,
                    progress_callback=update_progress
                )
                
            status_text.empty()
            st.sidebar.success("✅ PDFs Processed and Indexed successfully!")

    # ------------------
    # Main Tabs
    # ------------------
    tab1, tab2, tab3 = st.tabs(["Chat", "Summaries", "Literature Review Builder"])
    
    # ------------------
    # TAB 1: Chat 
    # ------------------
    with tab1:
        st.header("Chat with your Papers")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
            
        user_query = st.chat_input("Ask a question about the uploaded papers...")
        
        if user_query:
            if not api_key:
                st.error("Please configure your API Key in the sidebar.")
            elif st.session_state.vector_store is None:
                st.warning("Please upload and process PDFs first.")
            else:
                # Add human message 
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user").write(user_query)
                
                # Retrieval
                docs = st.session_state.vector_store.similarity_search(user_query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""
                You are a helpful research assistant. Answer the user's question using ONLY the provided context from the research papers.
                If the answer isn't in the context, say "I don't know based on the provided papers."
                
                Context:
                {context}
                
                Question:
                {user_query}
                """
                
                client = get_openrouter_client(api_key)
                with st.spinner("Thinking..."):
                    answer = call_openrouter(client, text_model, prompt)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)

    # ------------------
    # TAB 2: Summaries
    # ------------------
    with tab2:
        st.header("Paper Summaries")
        if st.session_state.documents_data:
            paper_names = list(st.session_state.documents_data.keys())
            selected_paper = st.selectbox("Select a paper to summarize", options=paper_names)
            
            if st.button("Generate Summary"):
                paper_text = st.session_state.documents_data[selected_paper]
                
                # Simple Map-Reduce style summarization approximation to handle long documents cleanly
                truncated_text = paper_text[:30000] # Safe 8k token limit heuristic (can be expanded)
                
                prompt = f"""
                You are an expert researcher. Please provide a structured summary of the following research paper.
                Include:
                - Core Objective
                - Methodology
                - Key Findings (Include data from figures/tables if present in text)
                - Conclusion
                
                Paper Text:
                {truncated_text}
                """
                client = get_openrouter_client(api_key)
                with st.spinner(f"Summarizing {selected_paper}..."):
                    summary = call_openrouter(client, text_model, prompt, temperature=0.3)
                    st.markdown(summary)
        else:
            st.info("Upload and process some PDFs first.")

    # ------------------
    # TAB 3: Lit Review
    # ------------------
    with tab3:
        st.header("Literature Review Builder")
        
        review_type = st.radio("Review Detail Level", ["Short", "Detailed/Long"])
        include_visuals = st.checkbox("Include references to graphs/tables", value=True)
        
        if st.button("Generate Literature Review"):
            if not api_key:
                st.error("Please configure your API Key.")
            elif not st.session_state.documents_data:
                st.error("Please upload and process papers first.")
            else:
                client = get_openrouter_client(api_key)
                
                # Fetching broad themes by searching keywords representing distinct sections
                search_terms = "objective methodology findings conclusion overview summary figures tables literature"
                docs = st.session_state.vector_store.similarity_search(search_terms, k=15)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                visuals_prompt = "Make sure to explicitly mention insights derived from tables and graphs." if include_visuals else "Do not focus heavily on specific tables or graphs."
                
                prompt = f"""
                You are a senior academic researcher writing a cross-study Literature Review based on the following extracted chunks from multiple papers.
                Format the output as a cohesive {review_type.lower()} literature review. 
                {visuals_prompt}
                
                Structure it with:
                1. Introduction / Thematic Overview
                2. Comparing Methodologies
                3. Synthesis of Results
                4. Conclusion
                
                Extracted Data Context:
                {context}
                """
                
                with st.spinner(f"Synthesizing {review_type} Literature Review..."):
                    lit_review = call_openrouter(client, text_model, prompt, temperature=0.4)
                    st.markdown(lit_review)

if __name__ == "__main__":
    main()