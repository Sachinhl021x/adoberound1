"""Text chunking with overlap for optimal retrieval."""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    """Split documents into chunks with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects with preserved metadata
        """
        chunked_docs = self.splitter.split_documents(documents)

        # Add chunk metadata
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunk_size"] = len(doc.page_content)

        return chunked_docs

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """Split a single text string into chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of Document objects
        """
        chunks = self.splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    **(metadata or {}),
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        return documents
