"""Grader to assess relevancy of retrieved documents."""
import os
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

try:
    from langchain_aws import ChatBedrockConverse
except ImportError:
    from langchain_aws import ChatBedrock as ChatBedrockConverse

class RelevanceGrader:
    """Grades the relevance of retrieved documents to the query."""

    SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

    Criteria:
    1. If the document contains keywords or concepts related to the user question, grade it as "relevant".
    2. If the document is completely unrelated, grade it as "irrelevant".
    3. Be lenient - if it might help answer the question, mark it relevant.

    Output format:
    Return only "yes" or "no". "yes" means relevant, "no" means irrelevant."""

    def __init__(self, model: str = None, region: str = None):
        """Initialize grader."""
        self.model = model or os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        # Initialize Bedrock client
        credentials_config = {"region_name": self.region}
        
        self.llm = ChatBedrockConverse(
            model=self.model,
            temperature=0, 
            max_tokens=10,
            **credentials_config
        )

    def grade(self, question: str, documents: List[Document]) -> str:
        """Grade the relevance of documents.
        
        Args:
            question: User question
            documents: List of retrieved documents
            
        Returns:
            "relevant" if at least one document is relevant, else "irrelevant"
        """
        # If no docs, automatically irrelevant
        if not documents:
            return "irrelevant"
            
        # Optimization: Concatenate first 2-3 docs context and ask if ANY are relevant
        # or grade individually. For speed, let's grade the set.
        
        context_preview = "\n\n".join([d.page_content[:500] for d in documents[:3]])
        
        user_prompt = f"""Retrieved document snippets:
        {context_preview}
        
        User Question: {question}
        
        Are these documents relevant to the question? (yes/no)"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            grade = response.content.strip().lower()
            
            if "yes" in grade:
                return "relevant"
            else:
                return "irrelevant"
                
        except Exception as e:
            print(f"⚠ Grading failed: {e}")
            return "relevant" # Fallback to avoid blocking
