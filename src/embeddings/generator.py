"""Embedding generator using AWS Bedrock Titan embeddings."""

import os
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.embeddings import BedrockEmbeddings


class ConcurrentEmbeddingsAdapter:
    """Adapter exposing LangChain-compatible embedding methods with concurrency."""

    def __init__(self, generator: "EmbeddingGenerator", batch_size: int = 100, max_workers: int = 3):
        self.generator = generator
        self.batch_size = batch_size
        self.max_workers = max_workers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.generator.embed_documents_concurrent(
            texts,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.generator.embed_query(text)


class EmbeddingGenerator:
    """Generate embeddings using AWS Bedrock Titan embedding models."""

    def __init__(
        self,
        model: str = None,
        region: str = None,
    ):
        """Initialize the embedding generator.

        Args:
            model: Bedrock embedding model ID to use
            region: AWS region (defaults to AWS_REGION env var)
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        # Initialize Bedrock embeddings
        credentials_config = {}

        if bearer_token:
            # Use bearer token if provided
            print(f"✓ Using AWS Bedrock with bearer token authentication")
            # Note: Bearer token setup may require additional boto3 configuration
            credentials_config["region_name"] = self.region
        elif aws_access_key and aws_secret_key:
            # Use access key/secret key
            print(f"✓ Using AWS Bedrock with access key authentication")
            credentials_config = {
                "region_name": self.region,
            }
            # boto3 will automatically pick up AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from environment
        else:
            # Try using default credentials (IAM role, etc.)
            print(f"✓ Using AWS Bedrock with default credentials (IAM role)")
            credentials_config["region_name"] = self.region

        self.embeddings = BedrockEmbeddings(
            model_id=self.model,
            **credentials_config
        )

        print(f"✓ Initialized Bedrock embeddings: {self.model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    def get_embeddings_instance(self) -> BedrockEmbeddings:
        """Get the underlying LangChain embeddings instance.

        Returns:
            BedrockEmbeddings instance
        """
        return self.embeddings

    def get_concurrent_embeddings_instance(
        self,
        batch_size: int = 100,
        max_workers: int = 3,
    ) -> Any:
        """Get a LangChain-compatible embeddings instance with concurrent document embedding."""
        return ConcurrentEmbeddingsAdapter(self, batch_size=batch_size, max_workers=max_workers)

    def embed_documents_concurrent(
        self,
        texts: List[str],
        batch_size: int = 100,
        max_workers: int = 3
    ) -> List[List[float]]:
        """Generate embeddings for documents with concurrent batch processing.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per batch
            max_workers: Maximum concurrent workers

        Returns:
            List of embedding vectors in original order
        """
        if len(texts) <= batch_size:
            # Small batch, process normally
            return self.embed_documents(texts)

        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches concurrently
        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.embeddings.embed_documents, batch): idx
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"⚠ Error embedding batch {idx}: {e}")
                    # Return zero vectors for failed batch
                    results[idx] = [[0.0] * 1536] * len(batches[idx])

        # Flatten results
        all_embeddings = []
        for batch_embeddings in results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)

        return all_embeddings
