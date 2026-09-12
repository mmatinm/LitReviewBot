# LitReviewBot: Academic Literature Review Assistant

LitReviewBot is a local, multi-paper research assistant and literature review generator. It extracts structured content from academic PDFs, indexes text and visual elements into a hybrid vector store, and synthesizes cross-study literature reviews using state-of-the-art language models.

[![Launch App](https://img.shields.io/badge/Launch_App-LitReviewBot-2563EB?style=for-the-badge&logo=streamlit&logoColor=white)](https://litreviewbot.streamlit.app/)

---

## Key Features

- **High-Fidelity PDF Parsing**: Extracts text, LaTeX equations, tables, and figures using [Marker](https://github.com/VikParuchuri/marker) in fast, CPU-friendly mode.
- **Hybrid Retrieval Engine**: Combines dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and BM25 lexical search with CPU cross-encoder reranking (`bge-reranker-base`).
- **Multi-Turn Conversational Q&A**: Maintains conversation history, automatically reformulates follow-up queries, allows paper-specific scoping, and cites exact source chunks with page and section metadata.
- **Structured Paper Summaries**: Automatically generates comprehensive summaries covering objectives, methodology, empirical findings, and limitations.
- **4-Stage Review Synthesis**: Generates publication-grade literature reviews across multiple papers:
  1. *Thematic Landscape & Problem Formulation*
  2. *Comparative Methodologies & Architectural Matrix*
  3. *Empirical Synthesis & Benchmark Comparison*
  4. *Critical Discussion, Future Directions & Research Gaps*
- **Multi-Provider LLM Support**: Built-in support for **OpenRouter**, **AvalAI**, and **Hormouz AI**, with custom model IDs and automatic exponential backoff retries.
- **Export & Reuse Extracted Markdown**: Download extracted `.md` files individually or as a bulk ZIP archive to reuse in future sessions without re-running PDF extraction.

---

## Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/LitReviewBot.git
cd LitReviewBot
```

### 2. Set Up a Virtual Environment
```bash
# Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS:
# python3 -m venv venv
# source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Supported Providers & Models

LitReviewBot supports multiple OpenAI-compatible API providers configured directly from the sidebar:

| Provider | Supported Models | Default Vision / Image Model |
| :--- | :--- | :--- |
| **OpenRouter** | `google/gemini-2.5-flash`, `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.3-70b-instruct`, Custom IDs | `google/gemini-2.5-flash` |
| **AvalAI** | `gemini-2.5-flash-lite`, `gpt-4.1-nano`, Custom IDs | `gemini-2.5-flash-lite`, `gpt-4.1-nano` |
| **Hormouz AI** | `nemotron-3-ultra-550b-a55b-free`, `gemma-4-31b-it-free`, Custom IDs | `gemma-4-31b-it-free` |

---

## Usage Workflow

1. **Configure Provider**: Select your LLM provider and enter your API key in the sidebar.
2. **Upload Documents**: Drag and drop research papers (`.pdf`, `.txt`, `.md`).
3. **Index Documents**: Click **Index Documents**. Marker processes the PDFs into structured Markdown with page anchors, table markers, and figure captions.
4. **Explore & Synthesize**:
   - **Chat**: Ask targeted questions across all papers or narrow the scope to an individual paper.
   - **Paper Summaries**: Select any paper to generate a structured academic breakdown.
   - **Literature Review Builder**: Run the 4-stage pipeline to generate a comprehensive, publication-ready cross-study review.

### ⚡ Time-Saving Tip: Reuse Extracted Markdown (.md)
PDF parsing with Marker takes computational time. Once your papers are indexed:
1. Expand **📚 Indexed Papers** in the sidebar.
2. Click **📥 .md** next to any paper, or click **📦 Download All as ZIP**.
3. In future sessions, **upload these `.md` files directly** instead of the original PDFs. LitReviewBot will index them instantly, skipping the PDF parsing step completely.

---

## Project Structure

```text
├── app.py                 # Streamlit UI and orchestration entrypoint
├── src/                   # Core application package
│   ├── __init__.py
│   ├── api_client.py      # Multi-provider client wrapper, retry logic, query reformulation
│   ├── config.py          # Provider configurations and model registries
│   ├── marker_processor.py# Marker PDF parser and visual caption linking
│   ├── pdf_processor.py   # Fallback PyMuPDF extraction engine
│   └── vector_store.py    # Section-aware chunking, FAISS index, BM25, and reranking
├── requirements.txt       # Python dependencies
└── README.md
```

---

## License & Disclaimer

Distributed under the MIT License.

This tool is intended for research assistance and literature synthesis. Always verify critical citations, numerical findings, and empirical claims against original source publications.
