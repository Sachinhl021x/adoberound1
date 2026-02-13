"""RAG Evaluation using Ragas."""

import os
import pandas as pd
from typing import List, Dict, Any
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print(f"⚠ Ragas import failed: {e}")
    evaluate = None

class RAGEvaluator:
    """Evaluator for RAG system using Ragas metrics."""

    def __init__(self, generator, retriever):
        """Initialize the evaluator.
        
        Args:
            generator: AnswerGenerator instance
            retriever: DocumentRetriever instance
        """
        self.generator = generator
        self.retriever = retriever
        self.llm = generator.llm
        self.embeddings = retriever.vectorstore.embedding_function

    def evaluate_query(self, question: str, ground_truth: str = None) -> Dict[str, Any]:
        """Evaluate a single query.
        
        Args:
            question: The query to evaluate
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not evaluate:
            return {"error": "Ragas not installed"}

        print(f"Evaluating query: {question}")
        
        # Retrieve context
        docs = self.retriever.retrieve(question)
        contexts = [doc.page_content for doc in docs]
        
        # Generate answer
        print("Generating answer...")
        result = self.generator.generate_answer(question, docs)
        answer = result["answer"]
        
        
        # Prepare dataset with Ragas v0.4 column names
        data = {
            "user_input": [question],  # Was 'question'
            "response": [answer],      # Was 'answer' 
            "retrieved_contexts": [contexts], # Was 'contexts'
        }
        
        selected_metrics = [faithfulness, answer_relevancy]
        
        if ground_truth:
            data["reference"] = [ground_truth]
            selected_metrics.append(context_precision)
            # context_recall also needs reference
            
        dataset = Dataset.from_dict(data)
        
        print(f"Running Ragas evaluation with metrics: {[m.name for m in selected_metrics]}...")
        
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=LangchainLLMWrapper(self.llm),
            embeddings=LangchainEmbeddingsWrapper(self.embeddings),
        )
        
        return results
