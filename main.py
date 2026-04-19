"""
Streamlit front-end for the Gemini-powered RAG pipeline.

Lets the user upload a PDF, ask questions about it, and stream answers
in real time using retrieval from local indexed chunks.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from local_embedding import LocalEmbedding
from model import AiModel
from pdf_reader import PdfReader


# ------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

for key, default in [
    ("pdf_name", None),
    ("tmp_pdf_path", None),
    ("local_embedding", None),
    ("chat_history", []),
    ("paragraph_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ai_model() -> AiModel:
    """
    Load the generation model once per Streamlit server process.
    """
    load_dotenv()
    return AiModel()


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Persist uploaded PDF bytes to a temporary file and return its path.
    """
    suffix = f"_{uploaded_file.name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ------------------------------------------------------------------
# Sidebar - model status + document upload
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## RAG Assistant")
    st.markdown("Ask questions about any PDF using Gemini-powered retrieval and generation.")
    st.divider()

    st.markdown("#### Model")
    with st.status("Loading model client...", expanded=True) as model_status:
        ai_model = load_ai_model()
        model_status.update(label="Model ready", state="complete", expanded=False)

    st.divider()

    st.markdown("#### Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Drag and drop a PDF here, or click Browse files.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            if st.session_state.tmp_pdf_path and os.path.exists(st.session_state.tmp_pdf_path):
                os.unlink(st.session_state.tmp_pdf_path)

            st.session_state.pdf_name = uploaded_file.name
            st.session_state.tmp_pdf_path = save_uploaded_pdf(uploaded_file)
            st.session_state.local_embedding = None
            st.session_state.chat_history = []
            st.session_state.paragraph_count = 0

        if st.session_state.local_embedding is None:
            with st.status("Processing document...", expanded=True) as doc_status:
                st.write("Extracting text from PDF...")
                pdf_reader = PdfReader(st.session_state.tmp_pdf_path)
                paragraphs = pdf_reader.get_paragraphs()
                st.session_state.paragraph_count = len(paragraphs)

                st.write(f"Building embedding index for {len(paragraphs)} paragraphs...")
                embedding = LocalEmbedding()
                embedding.build_index(paragraphs)
                st.session_state.local_embedding = embedding

                doc_status.update(
                    label=f"Document ready - {len(paragraphs)} paragraphs indexed",
                    state="complete",
                    expanded=False,
                )
        else:
            st.success(
                f"{st.session_state.pdf_name}\n\n"
                f"{st.session_state.paragraph_count} paragraphs indexed"
            )
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.caption("Powered by Gemini generation + Gemini embeddings")


# ------------------------------------------------------------------
# Main content area - chat interface
# ------------------------------------------------------------------

st.markdown("# Ask Your Document")

if st.session_state.local_embedding is None:
    st.markdown("> Upload a PDF in the sidebar to begin asking questions.")
else:
    st.markdown(
        f"Chatting about {st.session_state.pdf_name} - "
        f"{st.session_state.paragraph_count} paragraphs indexed"
    )

st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    placeholder="Ask a question about your document...",
    disabled=(st.session_state.local_embedding is None),
)

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = ai_model.ask_a_question_from_pdf_stream(
            pdf_path=st.session_state.tmp_pdf_path,
            prompt=prompt,
            local_embedding=st.session_state.local_embedding,
        )
        full_response = st.write_stream(response_stream)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
