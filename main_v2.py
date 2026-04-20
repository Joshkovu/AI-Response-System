"""
Enhanced Streamlit RAG Assistant with multi-document support and persistence.

Features:
- Persistent storage of indexed documents (survives restart)
- Multi-document indexing (add multiple PDFs to same index)
- Document management (list, delete)
- Source citation in answers
- Streaming responses
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from local_embedding import LocalEmbedding
from model import AiModel
from pdf_reader import PdfReader
from video_generator import ExplainerVideoGenerator


# ------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant - Multi-Document",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "local_embedding" not in st.session_state:
    st.session_state.local_embedding = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None

if "latest_video_path" not in st.session_state:
    st.session_state.latest_video_path = None

if "latest_video_summary" not in st.session_state:
    st.session_state.latest_video_summary = None


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ai_model() -> AiModel:
    """Load the generation model once per Streamlit server process."""
    load_dotenv()
    return AiModel()


@st.cache_resource(show_spinner=False)
def load_embedding_index() -> LocalEmbedding:
    """
    Load the persistent embedding index once per Streamlit process.
    This persists across user sessions.
    """
    return LocalEmbedding()


@st.cache_resource(show_spinner=False)
def load_video_generator() -> ExplainerVideoGenerator:
    """Load the video generator once per Streamlit process."""
    return ExplainerVideoGenerator()


def save_uploaded_pdf(uploaded_file) -> str:
    """Persist uploaded PDF bytes to a temporary file and return its path."""
    suffix = f"_{uploaded_file.name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def format_source_info(search_results: list) -> str:
    """Format search results to include source information."""
    if not search_results:
        return ""
    
    context_parts = []
    sources = set()
    
    for doc, distance in search_results:
        content = doc.get("content", "")
        file_name = doc.get("file_name", "Unknown")
        
        if content:
            context_parts.append(content)
            sources.add(file_name)
    
    return "\n\n---\n\n".join(context_parts)


# ------------------------------------------------------------------
# Sidebar - Model + Document Management
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 RAG Assistant")
    st.markdown("Multi-document Q&A with local retrieval and persistent storage.")
    st.divider()

    # Model status
    st.markdown("### Model")
    with st.status("Loading models...", expanded=True) as status:
        ai_model = load_ai_model()
        embedding_index = load_embedding_index()
        video_generator = load_video_generator()
        status.update(label="✅ Models ready", state="complete", expanded=False)

    st.divider()

    # Document management
    st.markdown("### Documents")
    
    # Show index statistics
    stats = embedding_index.get_index_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed Docs", stats["total_documents"])
    with col2:
        st.metric("Chunks", stats["total_vectors"])

    # List indexed documents
    if stats["total_documents"] > 0:
        st.markdown("#### Indexed Documents")
        docs = stats["documents"]
        
        for doc in docs:
            doc_id = doc["doc_id"]
            file_name = doc["file_name"]
            chunk_count = doc["chunk_count"]
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.caption(f"📄 {file_name}\n_{chunk_count} chunks_")
            with col2:
                if st.button("🗑️", key=f"delete_{doc_id}", help="Delete this document"):
                    embedding_index.delete_document(doc_id)
                    st.session_state.chat_history = []
                    st.rerun()
            with col3:
                if st.button("✓", key=f"select_{doc_id}", help="Focus on this document"):
                    st.session_state.current_doc_id = doc_id
                    st.session_state.chat_history = []
    else:
        st.info("No documents indexed yet. Upload one below.")

    st.divider()

    # PDF uploader
    st.markdown("### Add Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF to index",
        type=["pdf"],
        help="Drag and drop a PDF to add it to the knowledge base",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if embedding_index.doc_store.document_exists(uploaded_file.name):
            st.warning(f"📄 {uploaded_file.name} is already indexed.")
        else:
            with st.status("Processing PDF...", expanded=True) as doc_status:
                st.write("📖 Extracting text...")
                tmp_path = save_uploaded_pdf(uploaded_file)
                
                try:
                    pdf_reader = PdfReader(tmp_path)
                    paragraphs = pdf_reader.get_paragraphs()
                    page_count = len(pdf_reader.reader.pages)
                    
                    st.write(f"📊 Embedding {len(paragraphs)} chunks...")
                    doc_id = embedding_index.build_index(
                        chunks=paragraphs,
                        file_name=uploaded_file.name,
                        page_count=page_count,
                        category="uploaded",
                    )
                    
                    st.session_state.chat_history = []
                    doc_status.update(
                        label=f"✅ Indexed: {uploaded_file.name}",
                        state="complete",
                        expanded=False,
                    )
                    st.success(f"Added {uploaded_file.name} ({len(paragraphs)} chunks)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    st.divider()
    st.markdown("### Explainer Video")

    if st.button("Generate 30s explainer", disabled=(stats["total_documents"] == 0)):
        with st.status("Generating explainer video...", expanded=True) as video_status:
            try:
                st.write("Creating a concise summary from indexed documents...")
                summary_context = embedding_index.get_context(
                    "Summarize the key ideas across all indexed documents.",
                    k=12,
                )

                summary_prompt = ai_model.full_prompt_for_rag(
                    relevant_sections=summary_context,
                    question_prompt=(
                        "Create a short explainer script with 6 concise lines. "
                        "Each line should be one key point in plain language."
                    ),
                )
                summary_text = ai_model.ask_a_question(summary_prompt)

                st.write("Rendering MP4 slides from summary...")
                video_path = video_generator.generate_video_from_summary(
                    summary=summary_text,
                    title="Document Explainer",
                    seconds_per_scene=5,
                    max_scenes=6,
                )

                st.session_state.latest_video_path = video_path
                st.session_state.latest_video_summary = summary_text
                video_status.update(
                    label="Explainer video ready",
                    state="complete",
                    expanded=False,
                )
            except Exception as error:
                video_status.update(label="Explainer generation failed", state="error", expanded=True)
                st.error(f"Video generation error: {error}")

    st.divider()
    st.caption("🔒 Private: embeddings local, docs indexed on-device")


# ------------------------------------------------------------------
# Main content area - Chat interface
# ------------------------------------------------------------------

st.markdown("# 💬 Ask Your Documents")

if st.session_state.latest_video_path and os.path.exists(st.session_state.latest_video_path):
    st.markdown("## Document Explainer Video")
    st.video(st.session_state.latest_video_path)
    if st.session_state.latest_video_summary:
        with st.expander("Show generated script"):
            st.write(st.session_state.latest_video_summary)

    with open(st.session_state.latest_video_path, "rb") as video_file:
        st.download_button(
            label="Download MP4",
            data=video_file.read(),
            file_name=os.path.basename(st.session_state.latest_video_path),
            mime="video/mp4",
            key="download_explainer_video",
        )

    st.divider()

if stats["total_documents"] == 0:
    st.markdown(
        """
        Welcome to the RAG Assistant! 
        
        This tool lets you upload PDFs and ask questions about them using AI-powered retrieval.
        
        **Features:**
        -  Persistent storage (documents saved permanently)
        -  Multi-document (index multiple PDFs together)
        -  Local retrieval (your docs don't leave your machine)
        -  Streaming responses (answers appear token-by-token)
        -  Source citation (see where answers come from)
        
        **Getting started:**
        1. Upload a PDF in the sidebar
        2. Wait for indexing to complete
        3. Ask your question below
        
        Upload a PDF to get started! →
        """
    )
else:
    # Show current focus
    if st.session_state.current_doc_id:
        current_doc = embedding_index.doc_store.get_document(st.session_state.current_doc_id)
        if current_doc:
            st.markdown(
                f"📖 **Querying:** {current_doc['file_name']} "
                f"({current_doc['chunk_count']} chunks)"
            )
        else:
            st.markdown(f"📚 **Querying:** All indexed documents")
    else:
        st.markdown(f"📚 **Querying:** All {stats['total_documents']} indexed documents")

    st.divider()

    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"📄 Sources: {', '.join(message['sources'])}")

    # Chat input
    prompt = st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=(stats["total_documents"] == 0),
    )

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write("🔍 Retrieving context...")
            
            # Get relevant chunks
            search_results = embedding_index.search(prompt, k=5)
            
            if not search_results:
                st.warning("No relevant documents found.")
            else:
                # Extract sources
                sources = set()
                for doc, _ in search_results:
                    file_name = doc.get("file_name", "Unknown")
                    if file_name:
                        sources.add(file_name)
                
                # Build context
                context = format_source_info(search_results)
                
                # Generate answer
                st.write("✍️ Generating answer...")
                
                try:
                    response_stream = ai_model.ask_a_question_from_pdf_stream(
                        pdf_path="",  # Not needed - we're using pre-built index
                        prompt=prompt,
                        local_embedding=embedding_index,
                    )
                    full_response = st.write_stream(response_stream)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": list(sources),
                    })
                    
                    # Show sources
                    if sources:
                        st.caption(f"📄 Sources: {', '.join(sorted(sources))}")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {e}")
