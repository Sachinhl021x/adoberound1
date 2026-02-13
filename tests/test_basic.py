"""Basic tests for the RAG system components."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentLoader
from src.chunking import TextChunker
from langchain.schema import Document


def test_document_loader():
    """Test document loader initialization."""
    loader = DocumentLoader("./data/raw")
    assert loader.data_dir.exists()


def test_text_chunker():
    """Test text chunker."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    test_doc = Document(
        page_content="This is a test document. " * 20,
        metadata={"source": "test.pdf"}
    )

    chunks = chunker.chunk_documents([test_doc])

    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_chunk_metadata():
    """Test that chunks preserve metadata."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)

    test_doc = Document(
        page_content="A" * 200,
        metadata={"source": "test.pdf", "page": 1}
    )

    chunks = chunker.chunk_documents([test_doc])

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
