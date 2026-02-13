"""ChromaDB vector store implementation."""

import os
from pathlib import Path
from typing import List, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class ChromaVectorStore:
    """Vector store using ChromaDB for persistent storage."""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_function: Optional[Any] = None,
    ):
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: Embedding function to use
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "adobe_documents"
        )
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./indexes/chromadb"
        )

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.embedding_function = embedding_function
        self.vectorstore = None

    def create_index(self, documents: List[Document]) -> None:
        """Create a new index from documents.

        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("No documents provided to create index")

        if not self.embedding_function:
            raise ValueError("Embedding function not provided")

        print(f"Creating ChromaDB index with {len(documents)} documents...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        print(f"✓ Index created and persisted to: {self.persist_directory}")

    def load_index(self) -> None:
        """Load an existing index from disk."""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"No index found at: {self.persist_directory}. "
                "Run build_index.py first to create an index."
            )

        if not self.embedding_function:
            raise ValueError("Embedding function not provided")

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )

        print(f"✓ Loaded index from: {self.persist_directory}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to an existing index.

        Args:
            documents: List of Document objects to add
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Load or create an index first.")

        self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents to index")

    def similarity_search(
        self, query: str, k: int = 5, filter: dict = None
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar Document objects
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Load or create an index first.")

        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: dict = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Load or create an index first.")

        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print(f"✓ Deleted collection: {self.collection_name}")

    def get_vectorstore(self) -> Chroma:
        """Get the underlying Chroma vectorstore instance.

        Returns:
            Chroma instance
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Load or create an index first.")
        return self.vectorstore

    def get_count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of documents in the collection
        """
        if not self.vectorstore:
            return 0
        return self.vectorstore._collection.count()

    def persist(self) -> None:
        """Persist the vector store to disk."""
        if self.vectorstore:
            # Note: In newer versions of Chroma, persistence is automatic
            # This method is kept for backward compatibility
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            print(f"✓ Index persisted to: {self.persist_directory}")

    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            # Recreate empty collection
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
            )
            print(f"✓ Cleared all documents from collection")
