#!/usr/bin/env python3
"""Run RAG evaluation."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore
from src.retrieval import DocumentRetriever
from src.generation import AnswerGenerator
from src.eval import RAGEvaluator
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def main():
    print("Initializing RAG system for evaluation...")
    
    try:
        # Initialize components
        embedding_gen = EmbeddingGenerator()
        vectorstore = ChromaVectorStore(embedding_function=embedding_gen.embeddings)
        vectorstore.load_index()
        
        # Get documents for Hybrid Search
        all_docs = []
        try:
            data = vectorstore.get_vectorstore().get()
            if data and data.get('documents'):
                all_docs = [
                    Document(page_content=txt, metadata=meta or {})
                    for txt, meta in zip(data['documents'], data['metadatas'])
                ]
        except Exception as e:
            print(f"⚠ Could not load documents: {e}")

        retriever = DocumentRetriever(vectorstore, documents=all_docs)
        generator = AnswerGenerator()
        
        evaluator = RAGEvaluator(generator, retriever)
        
        # Define test questions (ground truth optional)
        test_questions = [
            "What was Adobe's total revenue in Q4 FY2025?",
            "What refer to the key risk factors?",
            "How did the Digital Media segment perform?",
        ]
        
        print(f"\nRunning evaluation on {len(test_questions)} questions...")
        
        for q in test_questions:
            print(f"\nQuestion: {q}")
            results = evaluator.evaluate_query(q)
            print("Results:", results)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
