import streamlit as st
import os
from config import VISION_MODELS, TEXT_MODELS
from api_client import get_openrouter_client, call_openrouter
from pdf_processor import extract_pdf_data
from vector_store import initialize_vector_store, retrieve_context, retrieve_docs


def _load_uploaded_text_documents(uploaded_text_files, progress_callback=None):
    """Load user-provided TXT documents directly into documents_data."""
    docs = {}
    total = len(uploaded_text_files)
    for idx, txt_file in enumerate(uploaded_text_files):
        if progress_callback:
            progress_callback(f"Loading text file '{txt_file.name}' ({idx + 1}/{total})")

        raw = txt_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")

        name = txt_file.name
        if name in docs:
            base, ext = os.path.splitext(name)
            name = f"{base}_uploaded_{idx + 1}{ext}"
        docs[name] = text
    return docs

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
        uploaded_documents = st.file_uploader(
            "Upload documents (.pdf or .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload new papers as .pdf. If a paper was already processed before, upload its extracted .txt instead to skip PDF/vision processing.",
        )

        process_visuals_ui = st.checkbox("Process tables and figures with AI for uploaded PDFs (High Usage)", value=False, help="Applies only to uploaded PDFs. WARNING: This consumes a considerable amount of your OpenRouter usage limit and increases processing time. TXT inputs skip this step.")
        process_btn = st.button("process Papers")

    # State variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_data" not in st.session_state:
        st.session_state.documents_data = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "last_retrieval_debug" not in st.session_state:
        st.session_state.last_retrieval_debug = None
        
    # ------------------
    # Processing Phase
    # ------------------
    if process_btn:
        uploaded_documents = uploaded_documents or []
        uploaded_files = [f for f in uploaded_documents if f.name.lower().endswith(".pdf")]
        uploaded_text_files = [f for f in uploaded_documents if f.name.lower().endswith(".txt")]

        has_pdf = bool(uploaded_files)
        has_uploaded_txt = bool(uploaded_text_files)

        if not (has_pdf or has_uploaded_txt):
            st.sidebar.error("Please upload PDFs and/or TXT files.")
        elif has_pdf and process_visuals_ui and not api_key:
            st.sidebar.error("API Key missing! Required when visual processing is enabled.")
        else:
            client = get_openrouter_client(api_key) if (api_key and process_visuals_ui) else None
            status_text = st.empty()
            
            def update_progress(msg: str):
                status_text.text(f"⏳ {msg}")
            
            docs_from_pdf = {}
            docs_from_uploaded_txt = {}

            if has_pdf:
                with st.spinner("Extracting text from PDFs..."):
                    docs_from_pdf = extract_pdf_data(
                        uploaded_files,
                        client,
                        vision_model,
                        progress_callback=update_progress,
                        process_visuals=process_visuals_ui,
                    )

            if has_uploaded_txt:
                with st.spinner("Loading uploaded TXT files..."):
                    docs_from_uploaded_txt = _load_uploaded_text_documents(
                        uploaded_text_files,
                        progress_callback=update_progress,
                    )

            combined_docs = {}
            combined_docs.update(docs_from_uploaded_txt)
            combined_docs.update(docs_from_pdf)
            st.session_state.documents_data = combined_docs

            with st.spinner("Building vector store..."):
                st.session_state.vector_store = initialize_vector_store(
                    st.session_state.documents_data,
                    progress_callback=update_progress
                )
    tab1, tab2, tab3 = st.tabs(["Chat", "Summaries", "Literature Review Builder"])
    
    # ------------------
    # TAB 1: Chat 
    # ------------------
    with tab1:
        st.header("Chat with your Papers")
        
        # Streamlit uses SVGs or basic emojis for avatars.
        # This SVG natively replicates the 'default user icon' Streamlit theme (rounded square) but draws a boy figure instead.
        boy_avatar_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='4' fill='%23ff4b4b'/%3E%3Cpath d='M12 12A3.5 3.5 0 0 0 12 5A3.5 3.5 0 0 0 12 12zM7 21v-1.5A4.5 4.5 0 0 1 11.5 15h1A4.5 4.5 0 0 1 17 19.5V21zM11.5 4a3.5 3.5 0 0 0-1.5 1.5C9.5 5 10.5 4 11.5 4zM12 4.5c.5-1 2-1 2 0 0 1-1 1-2 0z' fill='white'/%3E%3C/svg%3E"
        
        chat_focus_paper = None
        if st.session_state.documents_data:
            paper_options = ["All papers"] + list(st.session_state.documents_data.keys())
            chosen = st.selectbox(
                "Paper scope for retrieval",
                options=paper_options,
                index=0,
                help="Use a specific paper when asking questions like 'this paper' or asking for references.",
            )
            chat_focus_paper = None if chosen == "All papers" else chosen

        show_retrieval_debug = st.checkbox(
            "Show retrieval debug",
            value=False,
            key="show_retrieval_debug",
            help="Displays the chunks retrieved for the latest Chat question.",
        )

        # Display chat history
        for msg in st.session_state.chat_history:
            avatar = boy_avatar_svg if msg["role"] == "user" else None
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])
            
        user_query = st.chat_input("Ask a question about the uploaded papers...")
        
        if user_query:
            if not api_key:
                st.error("Please configure your API Key in the sidebar.")
            elif st.session_state.vector_store is None:
                st.warning("Please upload and process PDFs first.")
            else:
                # Add human message 
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user", avatar=boy_avatar_svg).write(user_query)
                
                # Retrieval (references-aware + optional paper filter)
                retrieved_docs = retrieve_docs(
                    st.session_state.vector_store,
                    query=user_query,
                    k=6,
                    source_filter=chat_focus_paper,
                )
                context = "\n\n".join([d.page_content for d in retrieved_docs])
                # Keep prompts cost-safe while preserving enough evidence for grounded answers.
                context = context[:12000]

                st.session_state.last_retrieval_debug = {
                    "query": user_query,
                    "source_filter": chat_focus_paper,
                    "doc_count": len(retrieved_docs),
                    "context_chars": len(context),
                    "docs": [
                        {
                            "source": (d.metadata or {}).get("source", "unknown"),
                            "chunk_id": (d.metadata or {}).get("chunk_id", "?"),
                            "char_len": len(d.page_content or ""),
                            "preview": (d.page_content or "")[:280],
                        }
                        for d in retrieved_docs
                    ],
                }
                
                scope_hint = f"Retrieval scope is only this paper: {chat_focus_paper}." if chat_focus_paper else "Retrieval scope includes all uploaded papers."
                prompt = f"""
                You are a helpful research assistant. Answer the user's question using ONLY the provided context from the research papers.
                If the answer isn't in the context, say "I don't know based on the provided papers."
                {scope_hint}
                
                Context:
                {context}
                
                Question:
                {user_query}
                """
                
                client = get_openrouter_client(api_key)
                with st.spinner("Thinking..."):
                    answer = call_openrouter(client, text_model, prompt, max_tokens=900)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

        if show_retrieval_debug and st.session_state.last_retrieval_debug:
            dbg = st.session_state.last_retrieval_debug
            with st.expander("Retrieval Debug", expanded=False):
                st.caption(
                    f"Query: {dbg['query']} | Scope: {dbg['source_filter'] or 'All papers'} | Retrieved: {dbg['doc_count']} chunks | Context chars sent: {dbg['context_chars']}"
                )
                for i, item in enumerate(dbg["docs"], start=1):
                    st.markdown(
                        f"**{i}. {item['source']} | chunk_id={item['chunk_id']} | chars={item['char_len']}**"
                    )
                    st.text(item["preview"])

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
                truncated_text = paper_text[:180000]
                
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
                    summary = call_openrouter(client, text_model, prompt, temperature=0.3, max_tokens=1800)
                    st.session_state.summaries[selected_paper] = summary
                    st.markdown(summary)
            elif selected_paper in st.session_state.summaries:
                st.markdown(st.session_state.summaries[selected_paper])
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
                context = retrieve_context(st.session_state.vector_store, query=search_terms, k=20, source_filter=None)
                
                visuals_prompt = "Make sure to explicitly mention insights derived from tables and graphs." if include_visuals else "Do not focus heavily on specific tables or graphs."
                
                prompt = f"""
                You are a senior academic researcher writing a cross-study Literature Review based on the following extracted chunks from multiple papers.
                Format the output as a cohesive {review_type.lower()} literature review. 
                {visuals_prompt}

                CRITICAL WRITING CONSTRAINTS:
                - Write as a scientific story, not a list of isolated summaries.
                - Every paragraph must have one clear message/claim.
                - Every paragraph's claim must be explicitly supported by one or more papers from the provided context.
                - Synthesize across studies (agreements, tensions, and gaps), not just restate them.
                - Keep a professional academic tone and use precise language.
                
                Structure it with:
                1. Introduction / Thematic Overview
                2. Comparing Methodologies
                3. Synthesis of Results
                4. Conclusion
                
                Extracted Data Context:
                {context}
                """
                
                with st.spinner(f"Synthesizing {review_type} Literature Review..."):
                    lit_review = call_openrouter(client, text_model, prompt, temperature=0.4, max_tokens=2200)
                    st.markdown(lit_review)

if __name__ == "__main__":
    main()