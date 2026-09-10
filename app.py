import streamlit as st
import os
from types import SimpleNamespace
from config import IMAGE_MODELS, PROVIDERS, TEXT_MODELS
from api_client import (
    get_llm_client,
    get_openrouter_client,
    call_openrouter,
    condense_query_with_history,
)
from marker_processor import extract_pdf_data_with_marker
from vector_store import (
    initialize_vector_store,
    retrieve_context,
    retrieve_docs,
    _format_doc_for_context,
    _is_reference_query,
    extract_global_paper_briefs,
    retrieve_balanced_review_context,
)


def _prepare_reusable_uploaded_pdfs(uploaded_pdf_files):
    """Cache PDF bytes once so multiple parser backends can read safely."""
    reusable = []
    for f in uploaded_pdf_files:
        data = f.read()
        reusable.append(SimpleNamespace(name=f.name, read=lambda d=data: d))
    return reusable


def _load_uploaded_text_documents(uploaded_text_files, progress_callback=None):
    """Load user-provided TXT or Markdown documents directly into documents_data."""
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

        provider_name = st.selectbox(
            "LLM Provider",
            options=list(PROVIDERS.keys()),
            index=0,
            help="Select your LLM service provider.",
        )
        provider_config = PROVIDERS[provider_name]

        api_key = st.text_input(f"{provider_name} API Key", type="password")
        if not api_key:
            st.warning(f"Please enter your {provider_name} API Key to proceed.")
            if provider_config.get("key_url"):
                st.markdown(f"[Get one here]({provider_config['key_url']})")
        
        st.subheader("Model Selection")
        text_model_input = st.selectbox(
            "Main Text Model (For Q&A/Reviews)",
            options=provider_config.get("text_models", []),
            index=0,
            help=f"Select a text-capable model or type a valid {provider_name} model ID.",
        )
        custom_text = st.text_input("Or type custom Text Model ID:")
        text_model = custom_text if custom_text else text_model_input

        image_model_input = st.selectbox(
            "Image Model (For Captions)",
            options=provider_config.get("image_models", []),
            index=0,
            help=f"Select the {provider_name} vision-capable model used to caption extracted images.",
        )
        custom_image = st.text_input("Or type custom Image Model ID:")
        image_model = custom_image if custom_image else image_model_input
        
        st.subheader("Document Upload")
        uploaded_documents = st.file_uploader(
            "Upload documents (.pdf, .txt, or .md)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload new papers as .pdf, or upload processed .txt/.md files to skip PDF and vision processing.",
        )

        st.info("PDFs are parsed with Marker. If Marker fails, processing stops so the error can be debugged directly.")

        st.caption("Marker extracts Markdown, formulas, tables, and saves images with caption sidecars in extracted_visuals.")
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
    if "literature_review" not in st.session_state:
        st.session_state.literature_review = None
        
    # ------------------
    # Processing Phase
    # ------------------
    if process_btn:
        uploaded_documents = uploaded_documents or []
        uploaded_files = [f for f in uploaded_documents if f.name.lower().endswith(".pdf")]
        uploaded_text_files = [
            f for f in uploaded_documents
            if f.name.lower().endswith((".txt", ".md"))
        ]
        reusable_uploaded_files = _prepare_reusable_uploaded_pdfs(uploaded_files) if uploaded_files else []

        has_pdf = bool(reusable_uploaded_files)
        has_uploaded_txt = bool(uploaded_text_files)

        if not (has_pdf or has_uploaded_txt):
            st.sidebar.error("Please upload PDFs, TXT files, and/or Markdown files.")
        else:
            status_text = st.empty()
            
            def update_progress(msg: str):
                status_text.text(f"⏳ {msg}")
            
            docs_from_pdf = {}
            docs_from_uploaded_txt = {}
            extraction_failed = False

            if has_pdf:
                with st.spinner("Extracting text from PDFs with Marker..."):
                    try:
                        docs_from_pdf = extract_pdf_data_with_marker(
                            reusable_uploaded_files,
                            progress_callback=update_progress,
                            mode="fast",
                            api_key=api_key,
                            image_model=image_model,
                            base_url=provider_config["base_url"],
                            extra_headers=provider_config.get("headers"),
                        )
                    except Exception as e:
                        st.sidebar.error(f"Marker extraction failed: {e}")
                        extraction_failed = True

            if has_uploaded_txt:
                with st.spinner("Loading uploaded TXT/Markdown files..."):
                    docs_from_uploaded_txt = _load_uploaded_text_documents(
                        uploaded_text_files,
                        progress_callback=update_progress,
                    )

            if extraction_failed:
                st.stop()

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
        chat_header_col1, chat_header_col2 = st.columns([5, 1])
        with chat_header_col1:
            st.header("Chat with your Papers")
        with chat_header_col2:
            if st.button("🗑️ Clear Chat", help="Reset conversational chat history"):
                st.session_state.chat_history = []
                st.session_state.last_retrieval_debug = None
                st.rerun()
        
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
                client = get_llm_client(api_key, base_url=provider_config["base_url"])
                extra_headers = provider_config.get("headers")

                # Step 1: Condense follow-up question using conversation history for accurate retrieval
                search_query = user_query
                if st.session_state.chat_history:
                    with st.spinner("Refining search query from chat history..."):
                        search_query = condense_query_with_history(
                            client=client,
                            model=text_model,
                            chat_history=st.session_state.chat_history,
                            latest_query=user_query,
                            extra_headers=extra_headers,
                        )

                # Add user message to display and history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user", avatar=boy_avatar_svg).write(user_query)
                
                # Step 2: Retrieval (references-aware + optional paper filter) using reformulated search_query
                retrieved_docs = retrieve_docs(
                    st.session_state.vector_store,
                    query=search_query,
                    k=10,
                    source_filter=chat_focus_paper,
                    candidate_k=30,
                )
                context = "\n\n".join(_format_doc_for_context(d) for d in retrieved_docs)
                # Keep prompts cost-safe while preserving enough evidence for grounded answers.
                # context = context[:12000]

                st.session_state.last_retrieval_debug = {
                    "query": user_query,
                    "search_query": search_query,
                    "source_filter": chat_focus_paper,
                    "doc_count": len(retrieved_docs),
                    "context_chars": len(context),
                    "docs": [
                        {
                            "source": (d.metadata or {}).get("source", "unknown"),
                            "chunk_id": (d.metadata or {}).get("chunk_id", "?"),
                            "page": (d.metadata or {}).get("page", ""),
                            "section": (d.metadata or {}).get("section", ""),
                            "content_type": (d.metadata or {}).get("content_type", "text"),
                            "char_len": len((d.metadata or {}).get("raw_text", d.page_content) or ""),
                            "preview": ((d.metadata or {}).get("raw_text", d.page_content) or "")[:280],
                        }
                        for d in retrieved_docs
                    ],
                }

                # Step 3: Format sliding window of conversation history (up to last 3 prior turns / 6 messages)
                past_turns = st.session_state.chat_history[:-1][-6:]
                history_section = ""
                if past_turns:
                    history_lines = []
                    for m in past_turns:
                        sender = "User" if m["role"] == "user" else "Assistant"
                        content = m["content"].strip()
                        if len(content) > 1200:
                            content = content[:1200] + "..."
                        history_lines.append(f"{sender}: {content}")
                    history_section = "Previous Conversation History:\n" + "\n".join(history_lines) + "\n\n"
                
                scope_hint = f"Retrieval scope is only this paper: {chat_focus_paper}." if chat_focus_paper else "Retrieval scope includes all uploaded papers."
                prompt = f"""
                You are an expert academic research assistant. Answer the user's question using ONLY the provided context from the research papers and the ongoing conversation history.
                If the answer isn't in the context, say "I don't know based on the provided papers."
                {scope_hint}
                Cite supporting evidence naturally using the paper name, page number, or section when useful.
                Do not invent page numbers or citations.
                Give a complete, well-structured answer. Do not stop early, omit requested items,
                or truncate a list. For reference-list questions, include every reference
                present in the supplied context and preserve its numbering. Use clear Markdown
                headings and numbered lists when useful.
                Return only the final answer for the user. Do not describe your reasoning or
                retrieval process. Never say "the user is asking", "I need to", "looking at
                the context", "chunk", "retrieved chunks", "context", "metadata", or "debug".
                
                {history_section}Context:
                {context}
                
                Question:
                {user_query}
                """
                
                with st.chat_message("assistant"):
                    thinking = st.empty()
                    thinking.markdown("_Thinking..._")
                    try:
                        answer = call_openrouter(
                            client,
                            text_model,
                            prompt,
                            temperature=0.25,
                            max_tokens=6000,
                            extra_headers=extra_headers,
                        )
                    except Exception as error:
                        answer = f"Unable to answer the question: {error}"
                    if not answer or not answer.strip():
                        answer = "The text model returned an empty response. Please try again."
                    thinking.markdown(answer)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

        if show_retrieval_debug and st.session_state.last_retrieval_debug:
            dbg = st.session_state.last_retrieval_debug
            with st.expander("Retrieval Debug", expanded=True):
                query_caption = f"Query: {dbg['query']}"
                if dbg.get("search_query") and dbg["search_query"] != dbg["query"]:
                    query_caption += f" | Rewritten for retrieval: '{dbg['search_query']}'"
                st.caption(
                    f"{query_caption} | Scope: {dbg['source_filter'] or 'All papers'} | "
                    f"Retrieved: {dbg['doc_count']} chunks | Context chars sent: {dbg['context_chars']}"
                )
                for i, item in enumerate(dbg["docs"], start=1):
                    st.markdown(
                        f"**{i}. {item['source']} | page={item['page'] or '?'} | "
                        f"section={item['section'] or '?'} | chunk_id={item['chunk_id']} | "
                        f"type={item['content_type']} | chars={item['char_len']}**"
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
                
                # Slicing up to 250,000 characters (~60,000 tokens), safe for all modern models
                truncated_text = paper_text[:250000]
                
                prompt = f"""
                You are an expert researcher. Please provide a detailed, structured academic summary of the following research paper.
                Include:
                - Core Objective & Problem Formulation
                - Methodology & System Architecture
                - Key Empirical Findings & Metrics (Include specific data from figures/tables if present in text)
                - Critical Limitations & Conclusion
                
                Paper Text:
                {truncated_text}
                """
                client = get_llm_client(api_key, base_url=provider_config["base_url"])
                extra_headers = provider_config.get("headers")
                with st.spinner(f"Summarizing {selected_paper}..."):
                    summary = call_openrouter(
                        client,
                        text_model,
                        prompt,
                        temperature=0.3,
                        max_tokens=6000,
                        extra_headers=extra_headers,
                    )
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
        
        review_type = st.radio("Review Detail Level", ["Detailed/Long", "Short"], index=0)
        include_visuals = st.checkbox("Include references to graphs/tables", value=True)
        
        sec_max_tokens = 6000 if review_type == "Detailed/Long" else 3000
        sec_temp = 0.35

        generate_btn = st.button("Generate Literature Review", type="primary")
        
        if generate_btn:
            if not api_key:
                st.error(f"Please configure your {provider_name} API Key.")
            elif not st.session_state.documents_data:
                st.error("Please upload and process papers first.")
            elif st.session_state.vector_store is None:
                st.warning("Please upload and process papers to build the vector store first.")
            else:
                client = get_llm_client(api_key, base_url=provider_config["base_url"])
                extra_headers = provider_config.get("headers")
                paper_names = list(st.session_state.documents_data.keys())

                status = st.status("🔬 Synthesizing Literature Review...", expanded=True)

                # Pre-computation: Global Paper Briefs (provides 10,000-ft view of all papers to every section)
                status.write("📋 Extracting global paper profiles across all uploaded studies...")
                global_briefs = extract_global_paper_briefs(st.session_state.documents_data, max_chars_per_paper=3000)

                # =========================================================================
                # SECTION 1: Introduction & Thematic Landscape
                # =========================================================================
                status.write("📖 Stage 1/4: Retrieving evidence & synthesizing Thematic Landscape...")
                sec1_query = "research background motivation core problem objective research questions theoretical framework thematic overview scope"
                sec1_context = retrieve_balanced_review_context(
                    st.session_state.vector_store,
                    paper_names=paper_names,
                    query=sec1_query,
                    target_k=36,
                    max_chars=250000,
                    prioritize_visuals=include_visuals,
                )

                sec1_prompt = f"""
                You are a senior academic researcher writing Section 1 of a publication-grade cross-study Literature Review synthesizing {len(paper_names)} papers: {", ".join(paper_names)}.

                SECTION TO WRITE:
                ## 1. Thematic Landscape & Problem Formulation

                OBJECTIVES:
                - Synthesize the overarching research domain, historical background, and motivation shared across these studies.
                - Formulate the core research questions and challenges that the literature attempts to resolve.
                - Cluster the papers thematically: what are the schools of thought or paradigms represented?
                - Contrast the foundational goals and scope of each paper without writing disjointed serial summaries.
                - Set an authoritative academic tone and establish common terminology.

                GLOBAL PAPER PROFILES:
                {global_briefs}

                TARGETED EXTRACTED EVIDENCE:
                {sec1_context}

                CRITICAL WRITING INSTRUCTIONS:
                - Write a rich, coherent narrative synthesis. Every paragraph must advance a clear synthetic claim supported by citations.
                - Cite papers naturally using paper names (e.g., `[Chen et al., 2025]` or `[Filename]`).
                - Detail level: {review_type.lower()}. Do not truncate or omit any paper.
                - Output ONLY Section 1 in Markdown, starting with `## 1. Thematic Landscape & Problem Formulation`.
                """

                sec1_text = call_openrouter(
                    client,
                    text_model,
                    sec1_prompt,
                    temperature=sec_temp,
                    max_tokens=sec_max_tokens,
                    extra_headers=extra_headers,
                )

                # =========================================================================
                # SECTION 2: Comparative Methodologies & Architectural Paradigms
                # =========================================================================
                status.write("⚙️ Stage 2/4: Retrieving evidence & comparing Methodologies & Architectures...")
                sec2_query = "methodology system architecture algorithms experimental design datasets benchmarks baseline implementation pipeline"
                sec2_context = retrieve_balanced_review_context(
                    st.session_state.vector_store,
                    paper_names=paper_names,
                    query=sec2_query,
                    target_k=36,
                    max_chars=250000,
                    prioritize_visuals=include_visuals,
                )

                sec2_prompt = f"""
                You are a senior academic researcher writing Section 2 of a publication-grade cross-study Literature Review synthesizing {len(paper_names)} papers: {", ".join(paper_names)}.
                This section builds directly upon Section 1. Do NOT repeat general background or re-introduce the papers. Focus directly on technical and methodological comparison.

                SECTION TO WRITE:
                ## 2. Comparative Analysis of Methodologies & Architectural Paradigms

                OBJECTIVES:
                - Thoroughly compare the technical architectures, models, algorithmic frameworks, pipelines, and reasoning/agentic mechanisms across all studies.
                - Compare datasets, training paradigms, evaluation protocols, and baselines used by each study.
                - Analyze design assumptions, trade-offs, and computational complexity.
                - MANDATORY: Include a detailed, well-formatted Markdown Comparison Table comparing ALL papers:
                  | Paper | Core Architecture / Method | Key Technical Innovations | Datasets & Benchmarks | Primary Assumptions & Baselines |
                - Accompany the table with in-depth analytical paragraphs synthesizing why authors chose differing designs and where methodological paradigms diverge.

                GLOBAL PAPER PROFILES:
                {global_briefs}

                TARGETED EXTRACTED EVIDENCE:
                {sec2_context}

                PREVIOUS SECTION CONTEXT (FOR CONTINUITY):
                {sec1_text[:2000]}...

                CRITICAL WRITING INSTRUCTIONS:
                - Synthesize across studies (e.g., compare multi-agent consensus vs single-agent self-debugging, parameter size vs prompt optimization).
                - Cite papers explicitly. Ensure the Markdown table is complete and includes all papers.
                - Output ONLY Section 2 in Markdown, starting with `## 2. Comparative Analysis of Methodologies & Architectural Paradigms`.
                """

                sec2_text = call_openrouter(
                    client,
                    text_model,
                    sec2_prompt,
                    temperature=sec_temp,
                    max_tokens=sec_max_tokens,
                    extra_headers=extra_headers,
                )

                # =========================================================================
                # SECTION 3: Empirical Synthesis & Quantitative Benchmarks
                # =========================================================================
                status.write("📊 Stage 3/4: Retrieving evidence & synthesizing Quantitative Benchmarks & Results...")
                sec3_query = "experimental results empirical findings evaluation metrics performance benchmarks comparison tables figures data trade-offs ablation"
                sec3_context = retrieve_balanced_review_context(
                    st.session_state.vector_store,
                    paper_names=paper_names,
                    query=sec3_query,
                    target_k=38,
                    max_chars=250000,
                    prioritize_visuals=include_visuals,
                )

                sec3_prompt = f"""
                You are a senior academic researcher writing Section 3 of a publication-grade cross-study Literature Review synthesizing {len(paper_names)} papers: {", ".join(paper_names)}.
                This section synthesizes the empirical evidence, quantitative metrics, and benchmark results supporting the claims.

                SECTION TO WRITE:
                ## 3. Empirical Synthesis & Quantitative Benchmarks

                OBJECTIVES:
                - Synthesize experimental findings, benchmark performances, and quantitative metrics across all studies.
                - Report exact numbers, percentages, ablation results, and trends extracted from text, tables, and figures.
                - Compare performance trade-offs (e.g., accuracy vs latency, duplicate rate vs coverage, parameter scale vs reasoning depth).
                - MANDATORY: Include a comprehensive Markdown Performance Matrix Table:
                  | Paper | Evaluated Models / Configurations | Key Benchmarks & Tasks | Reported Metrics & Results | Observed Trade-offs / Failure Modes |
                - Accompany the table with rigorous analytical commentary on why certain models or methods prevailed.

                GLOBAL PAPER PROFILES:
                {global_briefs}

                TARGETED EXTRACTED EVIDENCE:
                {sec3_context}

                CRITICAL WRITING INSTRUCTIONS:
                - Use precise quantitative figures and statistics wherever present in the extracted evidence.
                - Connect empirical results directly back to the architectural choices analyzed in Section 2.
                - Output ONLY Section 3 in Markdown, starting with `## 3. Empirical Synthesis & Quantitative Benchmarks`.
                """

                sec3_text = call_openrouter(
                    client,
                    text_model,
                    sec3_prompt,
                    temperature=sec_temp,
                    max_tokens=sec_max_tokens,
                    extra_headers=extra_headers,
                )

                # =========================================================================
                # SECTION 4: Critical Discussion, Research Gaps & Future Directions
                # =========================================================================
                status.write("🔭 Stage 4/4: Retrieving evidence & synthesizing Research Gaps & Future Directions...")
                sec4_query = "conclusions critical discussion limitations open challenges gaps tensions future research directions recommendations"
                sec4_context = retrieve_balanced_review_context(
                    st.session_state.vector_store,
                    paper_names=paper_names,
                    query=sec4_query,
                    target_k=36,
                    max_chars=250000,
                    prioritize_visuals=include_visuals,
                )

                sec4_prompt = f"""
                You are a senior academic researcher writing the concluding Section 4 of a publication-grade cross-study Literature Review synthesizing {len(paper_names)} papers: {", ".join(paper_names)}.
                This section provides critical evaluation, synthesizes consensus versus contradictions, and provides a forward-looking research agenda.

                SECTION TO WRITE:
                ## 4. Critical Discussion, Unresolved Research Gaps & Future Directions

                MANDATORY SUBSECTIONS REQUIRED:
                ### 4.1 Cross-Study Agreements, Tensions & Contradictions
                - Where do the findings across these papers reinforce each other?
                - Where do their conclusions, empirical claims, or theoretical assumptions clash or contradict?

                ### 4.2 Critical Methodological, Computational & Dataset Limitations
                - What common limitations, evaluation blind spots, or compute/prompt constraints characterize the current state of the art?

                ### 4.3 Unresolved Research Gaps in the Literature
                - Identify and analyze at least 3-4 concrete, substantive research gaps that none of the reviewed papers have adequately resolved.

                ### 4.4 Promising Directions & Open Problems for Future Research
                - Provide concrete, actionable roadmaps and hypotheses for future researchers seeking to advance the field beyond these works.

                GLOBAL PAPER PROFILES:
                {global_briefs}

                TARGETED EXTRACTED EVIDENCE:
                {sec4_context}

                CRITICAL WRITING INSTRUCTIONS:
                - Be rigorous, analytical, and visionary. Avoid generic platitudes; tie research gaps to specific technical problems identified in the papers.
                - Ensure the "Research Gaps" and "Future Research" subsections are rich, detailed, and substantive.
                - Output ONLY Section 4 in Markdown, starting with `## 4. Critical Discussion, Unresolved Research Gaps & Future Directions`.
                """

                sec4_text = call_openrouter(
                    client,
                    text_model,
                    sec4_prompt,
                    temperature=sec_temp,
                    max_tokens=sec_max_tokens,
                    extra_headers=extra_headers,
                )

                # =========================================================================
                # Assemble Full Literature Review
                # =========================================================================
                paper_list_md = ", ".join(f"`{p}`" for p in paper_names)
                full_review = (
                    f"# Comprehensive Cross-Study Literature Review\n\n"
                    f"**Synthesized Studies ({len(paper_names)} Papers):** {paper_list_md}\n\n"
                    f"---\n\n"
                    f"{sec1_text.strip()}\n\n"
                    f"---\n\n"
                    f"{sec2_text.strip()}\n\n"
                    f"---\n\n"
                    f"{sec3_text.strip()}\n\n"
                    f"---\n\n"
                    f"{sec4_text.strip()}\n"
                )

                st.session_state.literature_review = full_review
                status.update(label="✅ Literature Review Synthesis Complete!", state="complete", expanded=False)

        # Display cached or newly generated literature review
        if st.session_state.literature_review:
            st.markdown(st.session_state.literature_review)
            st.download_button(
                label="📥 Download Literature Review (.md)",
                data=st.session_state.literature_review,
                file_name="literature_review.md",
                mime="text/markdown",
            )

if __name__ == "__main__":
    main()