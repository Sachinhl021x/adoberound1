"""Document loader for PDF and DOCX files."""

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from src.extraction import PDFExtractor
from src.chunking import IntelligentChunker


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self, data_dir: str = "./data/raw"):
        """Initialize the document loader.

        Args:
            data_dir: Directory containing documents to load
        """
        self.data_dir = Path(data_dir)

    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects

        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        if extension == ".pdf":
            # Use structured extraction for PDFs
            try:
                print(f"  Using structured PDF extraction for {file_path.name}...")
                extractor = PDFExtractor()
                data = extractor.extract(str(file_path))
                
                # Use intelligent chunking on the extracted data
                chunker = IntelligentChunker()
                # We need to know where the JSON was saved. 
                # The extractor saves to ./data/staged/{stem}.json by default.
                # Let's get the path from the extractor or assume default.
                # Actually extractor.extract returns dict, but chunker expects file path currently.
                # Let's verify intelligent_splitter.py again. 
                # It has `chunk_from_staged(staged_file_path)`.
                
                json_path = extractor.output_dir / f"{file_path.stem}.json"
                documents = chunker.chunk_from_staged(str(json_path))
                print(f"  ✓ Structured extraction yielded {len(documents)} chunks")
                return documents
                
            except Exception as e:
                print(f"  ⚠ Structured extraction failed ({e}), falling back to standard loader...")
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                
        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()

        # Add source metadata for fallback/docx
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = file_path.name

        return documents

    def load_all_documents(self) -> List[Document]:
        """Load all supported documents from the data directory.

        Returns:
            List of all Document objects from the directory
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        all_documents = []

        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    print(f"Loading: {file_path.name}")
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                    print(f"  ✓ Loaded {len(documents)} pages/sections")
                except Exception as e:
                    print(f"  ✗ Error loading {file_path.name}: {e}")

        return all_documents
