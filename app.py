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
    """Cache PDF bytes in memory, skipping empty files."""
    reusable = []
    for f in uploaded_pdf_files:
        data = f.read()
        if not data or len(data) == 0:
            st.sidebar.warning(f"Skipping '{f.name}': file is empty.")
            continue
        reusable.append(SimpleNamespace(name=f.name, read=lambda d=data: d))
    return reusable


def _load_uploaded_text_documents(uploaded_text_files, progress_callback=None):
    """Load text and Markdown files into document dictionary."""
    docs = {}
    total = len(uploaded_text_files)
    for idx, txt_file in enumerate(uploaded_text_files):
        if progress_callback:
            progress_callback(f"Loading '{txt_file.name}' ({idx + 1}/{total})")

        raw = txt_file.read()
        if not raw or len(raw) == 0:
            st.sidebar.warning(f"Skipping '{txt_file.name}': file is empty.")
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")

        if not text.strip():
            st.sidebar.warning(f"Skipping '{txt_file.name}': file contains no readable text.")
            continue

        name = txt_file.name
        if name in docs:
            base, ext = os.path.splitext(name)
            name = f"{base}_uploaded_{idx + 1}{ext}"
        docs[name] = text
    return docs


def _is_call_error(res: str) -> bool:
    """Check if model output indicates an error response."""
    if not res or not isinstance(res, str):
        return True
    s = res.strip()
    return (
        not s
        or s.startswith("Error calling Main Text Model")
        or s.startswith("Unable to answer")
    )


# Configuration
st.set_page_config(page_title="Literature Review Bot", page_icon="📚", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');

/* Color variables & foundation */
:root {
    --bg-canvas: #060A12;
    --bg-surface: rgba(14, 23, 42, 0.72);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(37, 99, 235, 0.35);
    --primary: #2563EB;
    --primary-gradient: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #0284C7 100%);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
}

html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8FAFC;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Chrome clean up */
header[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden !important;
}

/* Atmospheric canvas */
.stApp {
    background-color: #060A12 !important;
    background-image: 
        radial-gradient(circle at 15% 5%, rgba(37, 99, 235, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 12%, rgba(14, 165, 233, 0.10) 0%, transparent 45%),
        radial-gradient(circle at 50% 95%, rgba(29, 78, 216, 0.08) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
}

/* Custom scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.14);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(37, 99, 235, 0.45);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #090E1A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

.sidebar-brand-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 1.4rem;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.14) 0%, rgba(14, 23, 42, 0.6) 100%);
    border: 1px solid rgba(37, 99, 235, 0.28);
    border-radius: 12px;
}

.sidebar-section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 1.35rem 0 0.65rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #60A5FA;
}

.sidebar-section-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2563EB;
    box-shadow: 0 0 8px #2563EB;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(14, 23, 42, 0.7) !important;
    border: 1.5px dashed rgba(37, 99, 235, 0.35) !important;
    border-radius: 14px !important;
    padding: 12px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(37, 99, 235, 0.8) !important;
    background: rgba(18, 30, 54, 0.85) !important;
    box-shadow: 0 0 20px rgba(37, 99, 235, 0.2) !important;
}

/* Inputs & Selectboxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: rgba(14, 23, 42, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
}

/* Executive Cards */
.executive-card {
    background: rgba(14, 23, 42, 0.72);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
}

.executive-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
}

/* Hero Header */
.hero-wrapper {
    position: relative;
    padding: 1.6rem 1.85rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.7) 50%, rgba(14, 165, 233, 0.1) 100%);
    border: 1px solid rgba(37, 99, 235, 0.3);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.35);
    overflow: hidden;
}

.hero-wrapper::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 240px;
    height: 100%;
    background: radial-gradient(circle at top right, rgba(37, 99, 235, 0.22), transparent 70%);
    pointer-events: none;
}

.hero-top-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 0.6rem;
}

.hero-badge-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: rgba(37, 99, 235, 0.15);
    color: #93C5FD;
    border: 1px solid rgba(37, 99, 235, 0.3);
}

.pulse-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.28);
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse 2s infinite ease-in-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

.hero-title-text {
    font-family: 'Outfit', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, #FFFFFF 20%, #E2E8F0 60%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle-text {
    font-size: 0.94rem;
    line-height: 1.55;
    color: #94A3B8;
    max-width: 820px;
    margin: 0;
}

/* Status Metrics Strip */
.status-pill-grid {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 9px;
    margin-top: 1.1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 13px;
    border-radius: 9px;
    font-size: 0.8rem;
    font-weight: 500;
    background: rgba(14, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #CBD5E1;
    transition: all 0.2s ease;
}

.metric-pill-active {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.35);
    color: #34D399;
}

/* Floating Segmented Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.8) !important;
    padding: 5px !important;
    border-radius: 13px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    gap: 4px !important;
    display: inline-flex !important;
    margin-bottom: 1.25rem !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 8px 18px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Buttons */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 50%, #2563EB 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(37, 99, 235, 0.55) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(14, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #E2E8F0 !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: rgba(20, 32, 58, 0.9) !important;
    border-color: rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* 4-Stage Literature Review Stepper */
.pipeline-stepper {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 0.85rem 0 1.4rem 0;
}

@media (max-width: 960px) {
    .pipeline-stepper {
        grid-template-columns: 1fr;
    }
}

.step-card {
    position: relative;
    background: rgba(14, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    transition: all 0.25s ease;
}

.step-card:hover {
    border-color: rgba(37, 99, 235, 0.35);
    background: rgba(18, 30, 54, 0.8);
}

.step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.step-badge {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #60A5FA;
}

.step-number {
    font-size: 0.75rem;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
}

.step-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #F8FAFC;
    margin: 0 0 3px 0;
}

.step-desc {
    font-size: 0.74rem;
    color: #94A3B8;
    line-height: 1.35;
    margin: 0;
}
</style>
"""


def main():
    # Inject design system stylesheet immediately
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # State initialization
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

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-card">
                <svg width="26" height="26" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="32" height="32" rx="8" fill="url(#sideBrandGrad)"/>
                    <path d="M9 11L16 7L23 11L16 15L9 11Z" stroke="white" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9 16L16 20L23 16" stroke="white" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9 21L16 25L23 21" stroke="white" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                    <defs>
                        <linearGradient id="sideBrandGrad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#1E40AF"/>
                            <stop offset="1" stop-color="#0284C7"/>
                        </linearGradient>
                    </defs>
                </svg>
                <div>
                    <div style="font-weight: 700; font-size: 0.96rem; letter-spacing: -0.02em; color: #F8FAFC;">ScholarLens</div>
                    <div style="font-size: 0.70rem; color: #60A5FA; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Research Workbench</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-header"><span class="sidebar-section-dot"></span>01 · Provider & Auth</div>', unsafe_allow_html=True)

        provider_name = st.selectbox(
            "LLM Provider",
            options=list(PROVIDERS.keys()),
            index=0,
            help="Select your LLM service provider.",
        )
        provider_config = PROVIDERS[provider_name]

        raw_api_key = st.text_input(f"{provider_name} API Key", type="password")
        api_key = raw_api_key.strip() if raw_api_key else ""
        if not api_key:
            st.warning(f"Please enter your {provider_name} API Key to proceed.")
            if provider_config.get("key_url"):
                st.markdown(f"[Get an API Key here]({provider_config['key_url']})")

        st.markdown('<div class="sidebar-section-header"><span class="sidebar-section-dot"></span>02 · Model Configuration</div>', unsafe_allow_html=True)
        text_model_input = st.selectbox(
            "Main Text Model (Q&A & Review)",
            options=provider_config.get("text_models", []),
            index=0,
            help="Select a text-capable model or enter a custom model ID below.",
        )
        image_model_input = st.selectbox(
            "Image Model (Visual Captions)",
            options=provider_config.get("image_models", []),
            index=0,
            help="Select a vision model used to caption extracted figures and tables.",
        )

        with st.expander("🛠️ Custom Model Overrides", expanded=False):
            custom_text = st.text_input("Custom Text Model ID:")
            custom_image = st.text_input("Custom Image Model ID:")

        text_model = custom_text.strip() if (custom_text and custom_text.strip()) else text_model_input
        image_model = custom_image.strip() if (custom_image and custom_image.strip()) else image_model_input

        st.markdown('<div class="sidebar-section-header"><span class="sidebar-section-dot"></span>03 · Document Ingestion</div>', unsafe_allow_html=True)
        uploaded_documents = st.file_uploader(
            "Upload research papers (.pdf, .txt, .md)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF papers, or upload pre-extracted .txt/.md files.",
        )

        process_btn = st.button("⚡ Index Documents", type="primary", use_container_width=True)

        if st.session_state.documents_data:
            with st.expander(f"📚 Indexed Papers ({len(st.session_state.documents_data)})", expanded=False):
                for p_name in st.session_state.documents_data.keys():
                    st.caption(f"✓ `{p_name}`")
        
    # Document ingestion and indexing
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
                        st.sidebar.error(
                            f"❌ Marker extraction failed: {e}\n\n"
                            "**Troubleshooting:**\n"
                            "- Ensure PDFs are not encrypted or password-protected.\n"
                            "- Alternatively, upload the paper as a `.txt` or `.md` file."
                        )
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

            if not combined_docs:
                st.sidebar.error("❌ No readable text could be extracted from the uploaded files.")
                st.stop()

            st.session_state.documents_data = combined_docs

            with st.spinner("Building vector store..."):
                st.session_state.vector_store = initialize_vector_store(
                    st.session_state.documents_data,
                    progress_callback=update_progress
                )
                if st.session_state.vector_store is None:
                    st.sidebar.error("❌ Failed to initialize vector store: no valid text chunks found.")
                else:
                    st.sidebar.success(f"✅ Successfully indexed {len(combined_docs)} document(s)!")

    # Main header and status display
    paper_count = len(st.session_state.documents_data) if st.session_state.documents_data else 0
    has_index = st.session_state.vector_store is not None

    st.markdown(
        f"""
        <div class="hero-wrapper">
            <div class="hero-top-row">
                <div class="hero-badge-group">
                    <span class="hero-badge">ScholarLens · Multi-Paper Research Workbench</span>
                    <span class="pulse-indicator"><span class="pulse-dot"></span>Core Engine Online</span>
                </div>
            </div>
            <h1 class="hero-title-text">ScholarLens AI</h1>
            <p class="hero-subtitle-text">
                Autonomous academic literature synthesis, cross-study comparative matrices, and grounded conversational inquiry across full-text papers.
            </p>
            <div class="status-pill-grid">
                <span class="metric-pill {'metric-pill-active' if paper_count > 0 else ''}">
                    📄 <strong>{paper_count}</strong> {'Papers Loaded' if paper_count != 1 else 'Paper Loaded'}
                </span>
                <span class="metric-pill {'metric-pill-active' if has_index else ''}">
                    🧠 Vector Index: <strong>{'FAISS Engine Active' if has_index else 'Awaiting Documents'}</strong>
                </span>
                <span class="metric-pill">
                    ⚡ Text Model: <code>{text_model}</code>
                </span>
                <span class="metric-pill">
                    🌐 Provider: <strong>{provider_name}</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📑 Paper Summaries", "🔬 Literature Review Builder"])
    
    # Tab 1: Interactive chat
    with tab1:
        chat_header_col1, chat_header_col2, chat_header_col3 = st.columns([3, 2, 1])
        with chat_header_col1:
            st.markdown("### 💬 Chat with your Papers")
        with chat_header_col2:
            chat_focus_paper = None
            if st.session_state.documents_data:
                paper_options = ["All papers"] + list(st.session_state.documents_data.keys())
                chosen = st.selectbox(
                    "Paper Scope",
                    options=paper_options,
                    index=0,
                    help="Limit retrieval to a single paper or search across all papers.",
                    label_visibility="collapsed",
                )
                chat_focus_paper = None if chosen == "All papers" else chosen
        with chat_header_col3:
            if st.button("🗑️ Clear", help="Reset conversational chat history", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_retrieval_debug = None
        # Retrieval debug traces toggle (commented out per user request)
        # show_retrieval_debug = st.checkbox(
        #     "Show retrieval debug traces",
        #     value=False,
        #     key="show_retrieval_debug",
        #     help="Displays the chunks retrieved for the latest Chat question.",
        # )

        # Starter inquiry suggestions when conversation is fresh
        starter_query = None
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="executive-card">
                    <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.25rem; color: #F8FAFC;">💡 Research Inquiry Starters</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.75rem;">Launch a multi-paper synthesis with one click or ask a custom question below:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                if st.button("🔍 Compare Core Methodologies & Paradigms", use_container_width=True):
                    starter_query = "Compare the core methodologies, architectures, and algorithmic frameworks proposed across these papers."
                if st.button("📊 Synthesize Benchmark Results & Metrics", use_container_width=True):
                    starter_query = "What are the primary datasets, evaluation benchmarks, and quantitative results reported?"
            with col_q2:
                if st.button("🎯 Formulate Overarching Research Questions", use_container_width=True):
                    starter_query = "What overarching research questions and problem formulations unite these studies?"
                if st.button("🔭 Identify Critical Limitations & Open Gaps", use_container_width=True):
                    starter_query = "What unresolved research gaps, compute limitations, and future directions do the authors highlight?"

        user_avatar = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%231D4ED8'/%3E%3Ccircle cx='16' cy='12' r='4' fill='white'/%3E%3Cpath d='M8 24c0-4.4 3.6-8 8-8s8 3.6 8 8' stroke='white' stroke-width='2' stroke-linecap='round' fill='none'/%3E%3C/svg%3E"
        assistant_avatar = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230B132B' stroke='%232563EB' stroke-width='1.5'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%2338BDF8'/%3E%3Ccircle cx='10' cy='10' r='2' fill='%2360A5FA'/%3E%3Ccircle cx='22' cy='10' r='2' fill='%2360A5FA'/%3E%3Ccircle cx='10' cy='22' r='2' fill='%2360A5FA'/%3E%3Ccircle cx='22' cy='22' r='2' fill='%2360A5FA'/%3E%3Cline x1='10' y1='10' x2='16' y2='16' stroke='%232563EB' stroke-width='1.2'/%3E%3Cline x1='22' y1='10' x2='16' y2='16' stroke='%232563EB' stroke-width='1.2'/%3E%3Cline x1='10' y1='22' x2='16' y2='16' stroke='%232563EB' stroke-width='1.2'/%3E%3Cline x1='22' y1='22' x2='16' y2='16' stroke='%232563EB' stroke-width='1.2'/%3E%3C/svg%3E"

        # Render conversation history
        for msg in st.session_state.chat_history:
            avatar = user_avatar if msg["role"] == "user" else assistant_avatar
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

        input_query = st.chat_input("Ask a question about the uploaded papers...")
        user_query = input_query or starter_query
        
        if user_query:
            if not api_key:
                st.error("Please configure your API Key in the sidebar.")
            elif st.session_state.vector_store is None:
                st.warning("Please upload and process PDFs first.")
            else:
                client = get_llm_client(api_key, base_url=provider_config["base_url"])
                extra_headers = provider_config.get("headers")

                # Rewrite follow-up query using chat context
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

                # Record user query
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.chat_message("user", avatar=user_avatar).write(user_query)
                
                # Retrieve relevant chunks
                retrieved_docs = retrieve_docs(
                    st.session_state.vector_store,
                    query=search_query,
                    k=10,
                    source_filter=chat_focus_paper,
                    candidate_k=30,
                )
                context = "\n\n".join(_format_doc_for_context(d) for d in retrieved_docs)

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

                # Include recent conversation turns in prompt
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
                
                with st.chat_message("assistant", avatar=assistant_avatar):
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

                    if _is_call_error(answer):
                        thinking.empty()
                        st.error(
                            f"❌ **Failed to generate answer:**\n\n{answer}\n\n"
                            "**Troubleshooting:**\n"
                            "- Verify your API key is correct and has active credits/quota.\n"
                            "- If you hit a rate limit (HTTP 429), please wait 15–30 seconds.\n"
                            "- Try selecting a different model or provider in the sidebar."
                        )
                        if st.session_state.chat_history and st.session_state.chat_history[-1].get("content") == user_query:
                            st.session_state.chat_history.pop()
                    else:
                        thinking.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()

        # Retrieval debug traces view (commented out per user request)
        # if st.session_state.last_retrieval_debug:
        #     dbg = st.session_state.last_retrieval_debug
        #     with st.expander("Retrieval Debug", expanded=True):
        #         query_caption = f"Query: {dbg['query']}"
        #         if dbg.get("search_query") and dbg["search_query"] != dbg["query"]:
        #             query_caption += f" | Rewritten for retrieval: '{dbg['search_query']}'"
        #         st.caption(
        #             f"{query_caption} | Scope: {dbg['source_filter'] or 'All papers'} | "
        #             f"Retrieved: {dbg['doc_count']} chunks | Context chars sent: {dbg['context_chars']}"
        #         )
        #         for i, item in enumerate(dbg["docs"], start=1):
        #             st.markdown(
        #                 f"**{i}. {item['source']} | page={item['page'] or '?'} | "
        #                 f"section={item['section'] or '?'} | chunk_id={item['chunk_id']} | "
        #                 f"type={item['content_type']} | chars={item['char_len']}**"
        #             )
        #             st.text(item["preview"])

    # Tab 2: Structured summaries
    with tab2:
        st.markdown("### 📑 Structured Paper Summaries")
        if st.session_state.documents_data:
            paper_names = list(st.session_state.documents_data.keys())

            with st.container(border=True):
                sum_col1, sum_col2 = st.columns([3, 1])
                with sum_col1:
                    selected_paper = st.selectbox("Select a paper to summarize", options=paper_names)
                    if selected_paper:
                        char_count = len(st.session_state.documents_data[selected_paper])
                        st.caption(f"Document Size: {char_count:,} characters · ~{char_count // 4:,} tokens")
                with sum_col2:
                    st.write("")
                    gen_sum_btn = st.button("⚡ Generate Summary", type="primary", use_container_width=True)

            if gen_sum_btn:
                paper_text = st.session_state.documents_data[selected_paper]

                # Cap input length to fit within model context limits
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
                with st.spinner(f"Synthesizing structured summary for {selected_paper}..."):
                    summary = call_openrouter(
                        client,
                        text_model,
                        prompt,
                        temperature=0.3,
                        max_tokens=6000,
                        extra_headers=extra_headers,
                    )
                    if _is_call_error(summary):
                        st.error(
                            f"❌ **Failed to generate summary for '{selected_paper}':**\n\n{summary}\n\n"
                            "**Troubleshooting:**\n"
                            "- Verify your API key has available credits or quota.\n"
                            "- If you hit a rate limit (HTTP 429), please wait 15–30 seconds before retrying.\n"
                            "- Try switching to another model (e.g. AvalAI `gemini-2.5-flash-lite` or `gpt-4.1-nano`)."
                        )
                    else:
                        st.session_state.summaries[selected_paper] = summary

            if selected_paper in st.session_state.summaries:
                current_summary = st.session_state.summaries[selected_paper]
                st.markdown("---")
                sum_bar1, sum_bar2 = st.columns([3, 1])
                with sum_bar1:
                    st.caption(f"Structured Academic Summary for **{selected_paper}** ({len(current_summary.split()):,} words)")
                with sum_bar2:
                    st.download_button(
                        label="📥 Download Summary (.md)",
                        data=current_summary,
                        file_name=f"{selected_paper}_summary.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                st.markdown(current_summary)
        else:
            st.info("Upload and index documents in the sidebar to view or generate structured summaries.")

    # Tab 3: Literature review builder
    with tab3:
        st.markdown("### 🔬 Publication-Grade Literature Review")

        st.markdown(
            """
            <div class="pipeline-stepper">
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-badge">Stage 1</span>
                        <span class="step-number">01</span>
                    </div>
                    <div class="step-title">Thematic Landscape</div>
                    <p class="step-desc">Scope, problem formulation, and overarching research paradigms across papers.</p>
                </div>
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-badge">Stage 2</span>
                        <span class="step-number">02</span>
                    </div>
                    <div class="step-title">Methodologies & Matrix</div>
                    <p class="step-desc">Architectural comparisons, algorithmic trade-offs, and comparative table.</p>
                </div>
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-badge">Stage 3</span>
                        <span class="step-number">03</span>
                    </div>
                    <div class="step-title">Empirical Benchmarks</div>
                    <p class="step-desc">Quantitative datasets, evaluation metrics, and cross-study findings.</p>
                </div>
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-badge">Stage 4</span>
                        <span class="step-number">04</span>
                    </div>
                    <div class="step-title">Discussion & Gaps</div>
                    <p class="step-desc">Critical contradictions, compute bottlenecks, and future research agenda.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                review_type = st.radio("Review Detail Level", ["Detailed/Long", "Short"], index=0, horizontal=True)
            with p_col2:
                include_visuals = st.checkbox("Prioritize extracted figures, diagrams & tables", value=True)

            sec_max_tokens = 6000 if review_type == "Detailed/Long" else 3000
            sec_temp = 0.35

            generate_btn = st.button("🔬 Synthesize Cross-Study Literature Review", type="primary", use_container_width=True)
        
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

                status = st.status("Synthesizing Literature Review...", expanded=True)

                # Extract global document summaries
                status.write("Extracting paper profiles across all uploaded studies...")
                global_briefs = extract_global_paper_briefs(st.session_state.documents_data, max_chars_per_paper=3000)

                # Section 1: Thematic Landscape
                status.write("Stage 1/4: Retrieving evidence & synthesizing Thematic Landscape...")
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
                if _is_call_error(sec1_text):
                    status.update(label="Failed generating Section 1: Thematic Landscape", state="error")
                    st.error(
                        f"**Failed generating Section 1 (Thematic Landscape):**\n\n{sec1_text}\n\n"
                        "**Troubleshooting:**\n"
                        "- Verify your API key and account credit balance/quota.\n"
                        "- Upstream model server or reverse proxy timed out or rate-limited. Wait a few moments and retry.\n"
                        "- Consider choosing an alternate model in the sidebar (e.g., `gemini-2.5-flash-lite`, `gpt-4.1-nano`, or an OpenRouter model)."
                    )
                    st.stop()

                # Section 2: Comparative Methodologies
                status.write("Stage 2/4: Retrieving evidence & comparing Methodologies & Architectures...")
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
                if _is_call_error(sec2_text):
                    status.update(label="Failed generating Section 2: Methodologies & Architectures", state="error")
                    st.error(
                        f"**Failed generating Section 2 (Methodologies & Architectures):**\n\n{sec2_text}\n\n"
                        "**Troubleshooting:**\n"
                        "- Verify your API key and quota.\n"
                        "- Upstream model server timed out or failed. Click 'Generate Literature Review' to retry."
                    )
                    st.stop()

                # Section 3: Empirical Synthesis & Benchmarks
                status.write("Stage 3/4: Retrieving evidence & synthesizing Quantitative Benchmarks & Results...")
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

                PREVIOUS METHODOLOGY CONTEXT (FOR CONTINUITY):
                {sec2_text[:2000]}...

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
                if _is_call_error(sec3_text):
                    status.update(label="Failed generating Section 3: Empirical Synthesis & Benchmarks", state="error")
                    st.error(
                        f"**Failed generating Section 3 (Empirical Synthesis & Benchmarks):**\n\n{sec3_text}\n\n"
                        "**Troubleshooting:**\n"
                        "- Upstream server timed out or returned an empty response. Click 'Generate Literature Review' to retry."
                    )
                    st.stop()

                # Section 4: Critical Discussion & Research Gaps
                status.write("Stage 4/4: Retrieving evidence & synthesizing Research Gaps & Future Directions...")
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

                PREVIOUS SECTIONS SUMMARY (FOR CONTINUITY):
                - Methodology Highlights:
                {sec2_text[:1200]}...

                - Empirical Findings Highlights:
                {sec3_text[:1200]}...

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
                if _is_call_error(sec4_text):
                    status.update(label="Failed generating Section 4: Critical Discussion & Future Directions", state="error")
                    st.error(
                        f"**Failed generating Section 4 (Critical Discussion & Future Directions):**\n\n{sec4_text}\n\n"
                        "**Troubleshooting:**\n"
                        "- Upstream server timed out or returned an empty response. Click 'Generate Literature Review' to retry."
                    )
                    st.stop()

                # Assemble final review document
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
                status.update(label="Literature Review Synthesis Complete", state="complete", expanded=False)

        # Display cached or newly generated literature review
        if st.session_state.literature_review:
            review_text = st.session_state.literature_review
            word_count = len(review_text.split())
            paper_count = len(st.session_state.documents_data) if st.session_state.documents_data else 0

            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns([1, 1, 2])
            with m_col1:
                st.metric("Synthesized Papers", f"{paper_count} Studies")
            with m_col2:
                st.metric("Total Word Count", f"{word_count:,} words")
            with m_col3:
                st.write("")
                st.download_button(
                    label="📥 Download Literature Review (.md)",
                    data=review_text,
                    file_name="literature_review.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.markdown(review_text)

if __name__ == "__main__":
    main()