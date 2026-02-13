#!/usr/bin/env python3
"""Enhanced Streamlit UI with uploads, citations, and metadata filtering."""

import streamlit as st
import requests
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import InMemoryDocumentLoader
from src.chunking import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore
from src.retrieval import DocumentRetriever
from src.retrieval import DocumentRetriever
from src.generation import CitationAwareGenerator
from src.ui.styles import apply_custom_styles, render_header

# Load environment
load_dotenv()

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Page config
st.set_page_config(
    page_title="AI Leadership Insight Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Adobe Theme
apply_custom_styles()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "in_memory_mode" not in st.session_state:
    st.session_state.in_memory_mode = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = None


# Helper functions
def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_api(question: str, top_k: int = DEFAULT_TOP_K):
    """Query the FastAPI backend."""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None


def query_in_memory(question: str, top_k: int = DEFAULT_TOP_K):
    """Query uploaded documents in-memory."""
    if not st.session_state.uploaded_docs:
        return {"answer": "No documents uploaded.", "sources": [], "context_used": 0}

    try:
        # Retrieve relevant chunks
        documents = st.session_state.retriever.retrieve(question, k=top_k)

        # Generate answer with citations
        result = st.session_state.generator.generate_answer_with_citations(question, documents)

        return result
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None


def process_upload(uploaded_file):
    """Process uploaded file in-memory."""
    with st.spinner(f"📄 Processing {uploaded_file.name}..."):
        try:
            # Load document
            loader = InMemoryDocumentLoader()
            documents = loader.load_from_upload(uploaded_file)

            # Chunk documents
            chunker = TextChunker()
            chunks = chunker.chunk_documents(documents)

            # Add to session
            st.session_state.uploaded_docs.extend(chunks)

            # Initialize in-memory retriever if needed
            
            # Initialize in-memory retriever if needed
            if st.session_state.retriever is None:
                embedding_gen = EmbeddingGenerator()
                vectorstore = ChromaVectorStore(
                    embedding_function=embedding_gen.embeddings,
                    persist_directory=None,  # In-memory only
                )
                
                # Initialize with first batch
                vectorstore.create_index(chunks)
                
                st.session_state.retriever = DocumentRetriever(vectorstore, documents=chunks)
                st.session_state.generator = CitationAwareGenerator()
                
            else:
                # Add chunks to existing vectorstore
                st.session_state.retriever.vectorstore.add_documents(chunks)

            st.success(f"✅ Added {len(chunks)} chunks from {uploaded_file.name}")
            st.session_state.in_memory_mode = True

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


def process_url(url: str):
    """Process document from web URL."""
    with st.spinner(f"🌐 Loading from {url}..."):
        try:
            # Load document from URL
            loader = InMemoryDocumentLoader()
            documents = loader.load_from_url(url)

            # Chunk documents
            chunker = TextChunker()
            chunks = chunker.chunk_documents(documents)

            # Add to session
            st.session_state.uploaded_docs.extend(chunks)

            # Initialize in-memory retriever if needed
            
            # Initialize in-memory retriever if needed
            if st.session_state.retriever is None:
                embedding_gen = EmbeddingGenerator()
                vectorstore = ChromaVectorStore(
                    embedding_function=embedding_gen.embeddings,
                    persist_directory=None,  # In-memory only
                )
                
                # Initialize with first batch
                vectorstore.create_index(chunks)
                
                st.session_state.retriever = DocumentRetriever(vectorstore, documents=chunks)
                st.session_state.generator = CitationAwareGenerator()
                
            else:
                # Add chunks to existing vectorstore
                st.session_state.retriever.vectorstore.add_documents(chunks)

            st.success(f"✅ Added {len(chunks)} chunks from URL")
            st.session_state.in_memory_mode = True

        except Exception as e:
            st.error(f"Error loading from URL: {e}")


def display_citation_answer(result):
    """Display answer with clickable citations."""
    if not result:
        return

    # Display answer
    st.markdown(f"**Answer:**\n\n{result['answer']}")

    # Display citations
    if "citations" in result and result["citations"]:
        st.markdown("---")
        st.markdown("**📚 Citations:**")

        for citation_num, source in sorted(result["citations"].items()):
            with st.expander(f"[{citation_num}] {source['filename']} - Page {source['page']}"):
                st.markdown(f"**Type:** {source['type'].title()}")
                if source.get('section'):
                    st.markdown(f"**Section:** {source['section']}")
                st.markdown(f"**Preview:**\n{source['content_preview']}")

    # Display all sources
    if "sources" in result and result["sources"]:
        st.markdown("---")
        st.markdown(f"**📄 All Sources** ({len(result['sources'])} documents)")

        for source in result["sources"]:
            st.caption(f"[{source['citation_number']}] {source['filename']} - Page {source['page']} ({source['type']})")


# Sidebar
with st.sidebar:
    st.title("🤖 AI Leadership Insight Agent")

    # Mode selector
    st.subheader("📊 Mode")
    mode = st.radio(
        "Select mode:",
        ["Index (Persistent)", "Upload (In-Memory)"],
        index=1 if st.session_state.in_memory_mode else 0
    )

    use_in_memory = mode == "Upload (In-Memory)"

    if not use_in_memory:
        # API Status
        st.subheader("🔌 System Status")
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.caption(f"Make sure FastAPI is running at {FASTAPI_URL}")

    # Upload section
    if use_in_memory:
        st.subheader("📤 Upload Documents")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDFs, DOCX, Images",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Documents will be processed in-memory (not persisted)"
        )

        if uploaded_files:
            if st.button("Process Uploads"):
                for uploaded_file in uploaded_files:
                    process_upload(uploaded_file)

        # Web URL input
        st.markdown("**Or load from web URL:**")
        url_input = st.text_input(
            "Enter web URL",
            placeholder="https://example.com/document.pdf",
            help="Supports PDF, DOCX, or plain text URLs"
        )

        if url_input and st.button("Load from URL"):
            process_url(url_input)

        if st.session_state.uploaded_docs:
            st.info(f"📚 {len(st.session_state.uploaded_docs)} chunks in memory")

            if st.button("Clear Uploads"):
                st.session_state.uploaded_docs = []
                st.session_state.retriever = None
                st.session_state.generator = None
                st.session_state.in_memory_mode = False
                st.rerun()

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of sources", 1, 10, DEFAULT_TOP_K)

    # Metadata filtering
    if use_in_memory and st.session_state.uploaded_docs:
        st.subheader("🔍 Filter Sources")

        # Get available metadata
        available_types = set(doc.metadata.get("type", "text") for doc in st.session_state.uploaded_docs)
        filter_type = st.multiselect("Filter by type:", list(available_types), default=list(available_types))

    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "What was Adobe's total revenue in Q4 FY2025?",
        "How did Digital Media perform?",
        "What are the key growth drivers?",
        "Summarize the financial highlights",
    ]

    for example in examples:
        if st.button(example, key=f"example_{example[:20]}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()

# Main content
render_header()  # Custom header
# st.title("Adobe Leadership Insight Agent")
st.caption("Ask strategic questions about company performance, driven by AI.")

# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "src/ui/adobe-icon.png" # You might need a local file or URL
    # Use standard icon for now to avoid broken image if file missing
    avatar = "👤" if message["role"] == "user" else "🤖"
    
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and "result" in message:
            display_citation_answer(message["result"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            if use_in_memory:
                result = query_in_memory(prompt, top_k)
            else:
                result = query_api(prompt, top_k)

            if result:
                display_citation_answer(result)

                # Store in message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "result": result
                })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📡 Mode: {mode}")
with col2:
    st.caption(f"🔧 Top-K: {top_k}")
with col3:
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
