#!/usr/bin/env python3
"""Enhanced index builder with extraction → chunking → embedding pipeline."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction import DocumentExtractor
from src.chunking.intelligent_splitter import IntelligentChunker
from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore

# Load environment
load_dotenv()
console = Console()


def main():
    """Build index using enhanced pipeline."""
    console.print("\n[bold cyan]🚀 Enhanced RAG Index Builder[/bold cyan]")
    console.print("Pipeline: Extraction → Intelligent Chunking → Embedding\n")

    # Configuration
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
    staged_dir = Path(os.getenv("DATA_STAGED_DIR", "./data/staged"))
    index_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "./indexes/chromadb"))

    console.print(f"📁 Raw documents: {raw_dir}")
    console.print(f"📁 Staged data: {staged_dir}")
    console.print(f"📁 Index: {index_dir}\n")

    # Step 1: Extract documents
    console.print("[bold yellow]Step 1: Document Extraction[/bold yellow]")
    extractor = DocumentExtractor(
        output_dir=str(staged_dir),
        use_image_llm=os.getenv("USE_IMAGE_LLM", "true").lower() == "true"
    )

    # Find all PDF and DOCX files
    documents = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.docx"))

    if not documents:
        console.print(f"[red]✗ No documents found in {raw_dir}[/red]")
        return

    console.print(f"Found {len(documents)} documents\n")

    extracted_files = extractor.extract_batch(
        [str(doc_path) for doc_path in documents],
        use_docling=os.getenv("USE_DOCLING", "true").lower() == "true",
        max_workers=int(os.getenv("EXTRACTION_MAX_WORKERS", "4")),
    )

    console.print(f"\n[green]✓ Extracted {len(extracted_files)} documents[/green]\n")

    # Step 2: Intelligent Chunking
    console.print("[bold yellow]Step 2: Intelligent Chunking[/bold yellow]")
    chunker = IntelligentChunker(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        respect_sections=True,
        include_images=True,
        include_tables=True,
    )

    chunks = chunker.chunk_batch_from_staged(str(staged_dir))
    console.print(f"\n[green]✓ Created {len(chunks)} intelligent chunks[/green]")

    # Show chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk.metadata.get("type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    console.print("\nChunk types:")
    for chunk_type, count in chunk_types.items():
        console.print(f"  - {chunk_type}: {count}")

    # Step 3: Generate Embeddings
    console.print(f"\n[bold yellow]Step 3: Generating Embeddings[/bold yellow]")
    embedding_generator = EmbeddingGenerator()
    console.print(f"Model: {embedding_generator.model}")

    concurrent_embeddings = embedding_generator.get_concurrent_embeddings_instance(
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        max_workers=int(os.getenv("EMBEDDING_MAX_WORKERS", "3")),
    )

    # Step 4: Store in ChromaDB
    console.print(f"\n[bold yellow]Step 4: Storing in ChromaDB[/bold yellow]")
    vectorstore = ChromaVectorStore(
        embedding_function=concurrent_embeddings,
        persist_directory=str(index_dir)
    )

    # Check for existing index and clear it
    try:
        vectorstore.load_index()
        existing_count = vectorstore.get_count()
        if existing_count > 0:
            console.print(f"Existing index has {existing_count} chunks")
            console.print("[yellow]⚠ Clearing existing index...[/yellow]")
            vectorstore.delete_collection()
    except FileNotFoundError:
        console.print("Creating new index...")

    # Create index with all chunks
    batch_size = 100
    first_batch = chunks[:batch_size] if len(chunks) > batch_size else chunks
    remaining_chunks = chunks[batch_size:] if len(chunks) > batch_size else []

    vectorstore.create_index(first_batch)

    # Add remaining chunks if any
    if remaining_chunks:
        for i in track(range(0, len(remaining_chunks), batch_size), description="Adding remaining chunks..."):
            batch = remaining_chunks[i:i + batch_size]
            vectorstore.add_documents(batch)

    # Persist
    vectorstore.persist()

    final_count = vectorstore.get_count()
    console.print(f"\n[bold green]✓ Index built successfully![/bold green]")
    console.print(f"Total chunks indexed: {final_count}")
    console.print(f"Index location: {index_dir}\n")


if __name__ == "__main__":
    main()
