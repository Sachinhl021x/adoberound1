#!/usr/bin/env python3
"""Query the RAG system with questions."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore
from src.retrieval import DocumentRetriever
from src.generation import AnswerGenerator

console = Console()


def main():
    """Main query loop."""
    # Load environment variables
    load_dotenv()

    console.print("\n[bold cyan]AI Leadership Insight Agent[/bold cyan]")
    console.print("=" * 60)

    # Initialize components
    try:
        console.print("\n[yellow]Loading index...[/yellow]")

        # Initialize embeddings
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.get_embeddings_instance()

        # Load vector store
        vectorstore = ChromaVectorStore(embedding_function=embeddings)
        vectorstore.load_index()

        # Initialize retriever
        retriever = DocumentRetriever(vectorstore)

        # Initialize generator
        generator = AnswerGenerator()

        console.print("[green]✓ System ready![/green]\n")

    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("\n[yellow]Run: python scripts/build_index.py[/yellow]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    # Example queries
    examples = [
        "What was Adobe's total revenue in Q4 FY2025?",
        "What is the year-over-year revenue growth?",
        "What are the key risk factors mentioned?",
    ]

    console.print("[bold]Example questions:[/bold]")
    for i, q in enumerate(examples, 1):
        console.print(f"  {i}. {q}")

    console.print("\n" + "=" * 60 + "\n")

    # Query loop
    while True:
        try:
            # Get query from user
            query = console.input("[bold cyan]Your question:[/bold cyan] ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]\n")
                break

            # Retrieve relevant documents
            console.print(f"\n[yellow]Searching...[/yellow]")
            documents = retriever.retrieve(query)

            if not documents:
                console.print(
                    "[red]No relevant information found. Try rephrasing your question.[/red]\n"
                )
                continue

            console.print(f"[green]Found {len(documents)} relevant chunks[/green]")

            # Generate answer
            console.print(f"[yellow]Generating answer...[/yellow]")
            result = generator.generate_answer(query, documents)

            # Display answer
            console.print("\n" + "=" * 60)
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            ))

            # Display sources
            if result["sources"]:
                console.print("\n[bold]Sources:[/bold]")
                for source in result["sources"]:
                    console.print(f"  • {source['filename']} (Page {source['page']})")

            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
