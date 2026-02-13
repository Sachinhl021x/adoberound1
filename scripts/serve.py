#!/usr/bin/env python3
"""FastAPI server for the RAG system."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore
from src.retrieval import DocumentRetriever
from src.generation import AnswerGenerator, CitationAwareGenerator
from src.agent import AgentGraph

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Leadership Insight Agent",
    description="RAG system for answering leadership questions about company documents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
retriever = None
generator = None
citation_generator = None
agent = None


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str
    top_k: Optional[int] = None
    use_citations: Optional[bool] = False


class Source(BaseModel):
    """Source information."""
    filename: str
    page: Union[str, int]  # Accept both string and integer page numbers
    citation_number: Optional[int] = None
    type: Optional[str] = "text"
    section: Optional[str] = None
    content_preview: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: List[Source]
    context_used: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, generator, citation_generator, agent

    try:
        print("Initializing RAG system...")

        # Initialize embeddings
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.get_embeddings_instance()

        # Load vector store
        vectorstore = ChromaVectorStore(embedding_function=embeddings)
        vectorstore.load_index()

        # Get all documents for Hybrid Search (BM25)
        all_docs = []
        try:
            print("Loading documents for Hybrid Search...")
            # Chroma get() returns dict with 'ids', 'embeddings', 'documents', 'metadatas'
            data = vectorstore.get_vectorstore().get()
            if data and data.get('documents'):
                from langchain_core.documents import Document
                all_docs = [
                    Document(page_content=txt, metadata=meta or {})
                    for txt, meta in zip(data['documents'], data['metadatas'])
                ]
                print(f"✓ Loaded {len(all_docs)} documents for BM25")
        except Exception as e:
            print(f"⚠ Could not load documents for Hybrid Search: {e}")

        # Initialize retriever and generators
        retriever = DocumentRetriever(vectorstore, documents=all_docs)
        generator = AnswerGenerator()
        citation_generator = CitationAwareGenerator()
        
        # Initialize Agent
        print("Initializing Agentic Workflow...")
        agent = AgentGraph(retriever, generator)

        print("✓ RAG system ready!")

    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        sys.exit(1)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="ok",
        message="AI Leadership Insight Agent API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if retriever is None or generator is None:
        return HealthResponse(
            status="error",
            message="System not initialized"
        )

    return HealthResponse(
        status="ok",
        message="System is healthy"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint for asking questions.

    Args:
        request: QueryRequest with question and optional top_k

    Returns:
        QueryResponse with answer, sources, and metadata
    """
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Run Agentic Workflow
        print(f"Processing query via Agent: {request.question}")
        inputs = {"question": request.question}
        
        # Invoke LangGraph app
        final_state = agent.app.invoke(inputs)
        
        # Extract results
        answer = final_state.get("answer", "No answer generated.")
        sources_list = final_state.get("sources", [])
        context_len = final_state.get("context_used", 0)
        
        # Standardize sources
        sources = [Source(**s) for s in sources_list]

        return QueryResponse(
            answer=answer,
            sources=sources,
            context_used=context_len
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def main():
    """Run the FastAPI server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print(f"\nStarting API server at http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs\n")

    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
