"""Intelligent chunking that respects document structure (sections, tables, images)."""

import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IntelligentChunker:
    """Intelligent chunking that preserves document structure and metadata."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sections: bool = True,
        include_images: bool = True,
        include_tables: bool = True,
    ):
        """Initialize intelligent chunker.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            respect_sections: Don't split within sections
            include_images: Include image descriptions as chunks
            include_tables: Include tables as chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections
        self.include_images = include_images
        self.include_tables = include_tables

        # Standard text splitter for regular content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_from_staged(self, staged_file_path: str) -> List[Document]:
        """Read extracted data from staged directory and create intelligent chunks.

        Args:
            staged_file_path: Path to JSON file in data/staged/

        Returns:
            List of Document chunks with metadata
        """
        with open(staged_file_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        chunks = []
        metadata = extracted_data.get("metadata", {})

        # Process each page
        for page_data in extracted_data.get("pages", []):
            page_num = page_data["page_number"]

            # Process sections (if available and respect_sections is True)
            if self.respect_sections and page_data.get("sections"):
                for section in page_data["sections"]:
                    section_chunks = self._chunk_section(section, page_num, metadata)
                    chunks.extend(section_chunks)
            elif page_data.get("text"):
                # Fall back to regular text chunking
                text_chunks = self._chunk_text(page_data["text"], page_num, metadata)
                chunks.extend(text_chunks)

            # Process tables
            if self.include_tables and page_data.get("tables"):
                for table_idx, table in enumerate(page_data["tables"]):
                    table_chunk = self._create_table_chunk(table, page_num, table_idx, metadata)
                    chunks.append(table_chunk)

            # Process images
            if self.include_images and page_data.get("images"):
                for img_idx, image in enumerate(page_data["images"]):
                    image_chunk = self._create_image_chunk(image, page_num, img_idx, metadata)
                    chunks.append(image_chunk)

        return chunks

    def _chunk_section(self, section: Dict[str, Any], page_num: int, doc_metadata: Dict) -> List[Document]:
        """Chunk a section intelligently."""
        heading = section.get("heading", "")
        content = section.get("content", "")
        level = section.get("level", 1)

        # Combine heading and content
        full_text = f"{heading}\n\n{content}" if heading else content

        # If section is small enough, keep as single chunk
        if len(full_text) <= self.chunk_size:
            return [Document(
                page_content=full_text,
                metadata={
                    **doc_metadata,
                    "page": page_num,
                    "type": "section",
                    "section_heading": heading,
                    "section_level": level,
                    "chunk_size": len(full_text),
                }
            )]

        # Otherwise, split but preserve section context
        chunks = self.text_splitter.split_text(full_text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    **doc_metadata,
                    "page": page_num,
                    "type": "section",
                    "section_heading": heading,
                    "section_level": level,
                    "chunk_id": f"section_{page_num}_{idx}",
                    "chunk_size": len(chunk),
                }
            )
            for idx, chunk in enumerate(chunks)
        ]

    def _chunk_text(self, text: str, page_num: int, doc_metadata: Dict) -> List[Document]:
        """Chunk plain text."""
        chunks = self.text_splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    **doc_metadata,
                    "page": page_num,
                    "type": "text",
                    "chunk_id": f"text_{page_num}_{idx}",
                    "chunk_size": len(chunk),
                }
            )
            for idx, chunk in enumerate(chunks)
        ]

    def _create_table_chunk(
        self,
        table: Dict[str, Any],
        page_num: int,
        table_idx: int,
        doc_metadata: Dict
    ) -> Document:
        """Create a chunk for a table."""
        caption = table.get("caption", "")
        content = table.get("content", "")

        # Format table chunk
        table_text = f"Table: {caption}\n\n{content}" if caption else content

        return Document(
            page_content=table_text,
            metadata={
                **doc_metadata,
                "page": page_num,
                "type": "table",
                "table_caption": caption,
                "table_index": table_idx,
                "chunk_id": f"table_{page_num}_{table_idx}",
                "chunk_size": len(table_text),
            }
        )

    def _create_image_chunk(
        self,
        image: Dict[str, Any],
        page_num: int,
        img_idx: int,
        doc_metadata: Dict
    ) -> Document:
        """Create a chunk for an image (with OCR/LLM description)."""
        description = image.get("description", "[Image with no description]")
        image_type = image.get("type", "image")

        # Format image chunk
        image_text = f"[Image {img_idx + 1}]: {description}"

        return Document(
            page_content=image_text,
            metadata={
                **doc_metadata,
                "page": page_num,
                "type": "image",
                "image_index": img_idx,
                "image_format": image.get("format", "unknown"),
                "chunk_id": f"image_{page_num}_{img_idx}",
                "chunk_size": len(image_text),
            }
        )

    def chunk_batch_from_staged(self, staged_dir: str = "./data/staged") -> List[Document]:
        """Process all files in staged directory.

        Args:
            staged_dir: Directory containing staged JSON files

        Returns:
            List of all document chunks
        """
        staged_path = Path(staged_dir)
        all_chunks = []

        for json_file in staged_path.glob("*.json"):
            try:
                chunks = self.chunk_from_staged(str(json_file))
                all_chunks.extend(chunks)
                print(f"✓ Chunked {json_file.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error chunking {json_file.name}: {e}")

        return all_chunks
