"""Document retriever for semantic search."""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever



class DocumentRetriever:
    """Retrieve relevant documents based on queries."""

    def __init__(self, vectorstore, documents: List[Document] = None, top_k: int = None, similarity_threshold: float = None):
        """Initialize the document retriever.

        Args:
            vectorstore: Vector store instance to search
            documents: List of documents for BM25 initialization (optional)
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.vectorstore = vectorstore
        self.top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.similarity_threshold = similarity_threshold or float(
            os.getenv("SIMILARITY_THRESHOLD", "0.7")
        )
        
        
        # Initialize Hybrid Search if documents are provided
        self.bm25_retriever = None
        if documents:
            try:
                print(f"Initializing BM25 retriever with {len(documents)} documents...")
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = self.top_k
                
                # Create Ensemble Retriever (50% Semantic, 50% Keyword)
                # Note: We need to get the native retriever from vectorstore
                # Assuming vectorstore wrapper has a method to get it or we use the underlying object
                vector_retriever = self.vectorstore.get_vectorstore().as_retriever(
                    search_kwargs={"k": self.top_k}
                )
                
                # Use manual ensemble instead of LangChain's EnsembleRetriever due to import issues
                print("✓ Hybrid Search (BM25 + Semantic) initialized with manual RRF")
            except Exception as e:
                print(f"⚠ Failed to initialize Hybrid Search: {e}")
                self.bm25_retriever = None

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)

        Returns:
            List of relevant Document objects
        """
        k = k or self.top_k

        if self.bm25_retriever:
            return self._retrieve_hybrid(query, k)
            
        # Fallback to Semantic Search only
        return self._retrieve_semantic(query, k)

    def _retrieve_semantic(self, query: str, k: int) -> List[Document]:
        """Run semantic search only."""
        # Get documents with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Filter by similarity threshold
        filtered_results = [
            doc for doc, score in results if score >= self.similarity_threshold
        ]

        if not filtered_results:
            print(f"⚠ No results found above similarity threshold ({self.similarity_threshold})")

        return filtered_results

    def _retrieve_hybrid(self, query: str, k: int) -> List[Document]:
        """Run hybrid search with Reciprocal Rank Fusion (RRF)."""
        # 1. Get Keyword Results (BM25)
        bm25_docs = self.bm25_retriever.invoke(query)  # Usually returns top k by default (k=self.top_k)
        
        # 2. Get Semantic Results
        vector_results = self.vectorstore.similarity_search(query, k=k)
        
        # 3. Combine with RRF
        # RRF score = 1 / (c + rank)
        c = 60
        scores = {}
        doc_map = {}
        
        # Process BM25
        for rank, doc in enumerate(bm25_docs):
            # Use content as key (assuming unique content for chunks)
            # Better to use chunk_id if available, but content is safer fallback
            key = doc.page_content
            doc_map[key] = doc
            if key not in scores:
                scores[key] = 0.0
            scores[key] += 1.0 / (c + rank + 1)
            
        # Process Semantic
        for rank, doc in enumerate(vector_results):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
            if key not in scores:
                scores[key] = 0.0
            scores[key] += 1.0 / (c + rank + 1)
        
        # Sort by RRF score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k
        return [doc_map[content] for content, score in sorted_docs[:k]]

    def retrieve_with_scores(
        self, query: str, k: int = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with their similarity scores.

        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)

        Returns:
            List of (Document, score) tuples
        """
        k = k or self.top_k

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Filter by similarity threshold
        filtered_results = [
            (doc, score) for doc, score in results if score >= self.similarity_threshold
        ]

        return filtered_results

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            context_parts.append(
                f"[Source {i}] {source} (Page {page}):\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)
