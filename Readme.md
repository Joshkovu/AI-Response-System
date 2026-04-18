# RAG AI System - Ask Your Documents, Privately

A fully local Retrieval-Augmented Generation (RAG) pipeline with a Streamlit chat UI.

Upload any PDF, ask questions in plain English, and get grounded answers from your documents. Retrieval and indexing stay on your machine; generation is handled by Gemini.

## What is this?

Most document Q&A tools send your files to a third-party service for both retrieval and generation. This project keeps the document pipeline local and uses Gemini only for the final answer generation step:

- Embedding - a compact local sentence-transformer converts your document into 384-dimensional semantic vectors stored in a pure-Python in-memory index.
- Retrieval - cosine distance search finds the paragraphs most relevant to your question.
- Generation - Gemini reads only those paragraphs and streams an answer token-by-token directly to your browser.

Your PDFs are processed locally, and only the retrieved context used for generation is sent to Gemini.

## Screenshots

Ready to upload | Active conversation
---|---
Upload screen | Chat screen

The sidebar handles model loading, PDF ingestion, and status. The main area is a persistent chat, so you can ask follow-up questions without re-indexing.

## Architecture

Architecture diagram

Mermaid source

## Data flow in plain English

PDF -> Paragraphs - `PdfReader` reads every page, normalises whitespace, and splits on paragraph breaks.

Paragraphs -> Vectors - `LocalEmbedding.build_index()` batch-embeds all paragraphs in one GPU pass and stores them in `VectorIndex`.

Question -> Context - at query time, the question is embedded and compared against every stored vector by cosine distance; the top-k chunks are returned as a single context string.

Context + Question -> Answer - `AiModel` wraps context and question in a strict RAG prompt, sends the prompt to Gemini, and streams the answer back to the Streamlit UI.

## Key design decisions

Choice | Reason
---|---
`VectorIndex` in pure Python stdlib | No NumPy/FAISS dependency; educational and auditable
L2-normalised vectors + cosine distance | Equivalent to dot-product similarity - fast and well-calibrated
Streaming response handling | Lets Gemini output render progressively without blocking the Streamlit main thread
`@st.cache_resource` for `AiModel` | Model client loads once per server process regardless of how many times Streamlit reruns the script
`st.session_state` for the embedding index | Avoids re-indexing the same PDF on every follow-up question
Strict grounding prompt | The model is instructed to answer only from provided document text

## Getting Started

### Prerequisites

Requirement | Version
---|---
Python | 3.10+
Gemini API access | Google AI Studio API key or Vertex AI access
CUDA-capable GPU | Recommended for faster embedding generation; CPU works but is slower

### 1 - Clone the repository

```bash
git clone https://github.com/Joshkovu/AI-Response-System.git
cd AI-Response-System
```

### 2 - Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3 - Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers streamlit pypdf python-dotenv google-genai
```

GPU note: the cu121 wheel targets CUDA 12.1. Find the right wheel for your CUDA version at pytorch.org/get-started.

CPU-only? Replace the first command with `pip install torch torchvision torchaudio`.

### 4 - Add your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get an API key from Google AI Studio. If you are using Vertex AI instead, configure the matching Google Cloud credentials and project settings in your environment.

### 5 - Run the app

```bash
streamlit run main.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

First run: the embedding model is downloaded once and cached locally. The Gemini client then uses your API key for generation.

## Usage

The sidebar shows `LLM ready` once the generation client is loaded.

Drag and drop any PDF onto the uploader, or click Browse.

Wait for `Document ready - N paragraphs indexed` in the sidebar.

Type your question in the chat input at the bottom.

The answer streams token-by-token. Ask follow-up questions freely - the index is cached.

Upload a new PDF to start a fresh conversation.

## Project Structure

```text
RAG_AI_SYSTEM/
├── main.py                  # Streamlit UI - entry point
├── local_llm.py             # AiModel: loads Gemini, orchestrates RAG, streams output
├── local_embedding.py       # LocalEmbedding: MiniLM wrapper + index interface
├── vector_index.py          # VectorIndex: pure-stdlib cosine/Euclidean vector store
├── pdf_reader.py            # PdfReader: PDF -> clean paragraph list
├── pdfs/                    # Drop your PDFs here (gitignored)
├── application_screenshots/ # UI screenshots used in this README
├── .env                     # GEMINI_API_KEY (gitignored - create this yourself)
└── CLAUDE.md                # Guidance for Claude Code in this repo
```

## Tech Stack

Layer | Library | Role
---|---|---
UI | Streamlit | Chat interface, file upload, live streaming
LLM | Gemini | Answer generation
Embeddings | all-MiniLM-L6-v2 | Semantic search vectors
Inference | Google Gen AI SDK | Model loading and generation
Compute | PyTorch | GPU-accelerated inference
PDF parsing | pypdf | Text extraction
Vector store | Custom (`vector_index.py`) | In-memory cosine search - no external DB

## Extending the Project

Ideas for taking this further:

- Swap the LLM - update the Gemini model name in `AiModel` to a faster or more capable Gemini variant.
- Larger context - increase `k` in `get_context()` to pass more paragraphs to the model, while watching prompt size limits.
- Persistent index - serialize `VectorIndex.vectors` and `.documents` to disk so you do not re-embed on every startup.
- Multi-document - merge indices from several PDFs into one `VectorIndex` for cross-document Q&A.
- Chunking strategy - replace paragraph splits with fixed-token sliding windows for more uniform chunk sizes.
