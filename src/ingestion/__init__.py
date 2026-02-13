"""Document ingestion module for loading company documents."""

from .loader import DocumentLoader
from .memory_loader import InMemoryDocumentLoader

__all__ = ["DocumentLoader", "InMemoryDocumentLoader"]
