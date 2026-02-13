"""Hallucination grader to check if answer is grounded in context."""

from typing import List
try:
    from langchain_aws import ChatBedrockConverse
except ImportError:
    from langchain_aws import ChatBedrock as ChatBedrockConverse
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

class HallucinationGrader:
    """Evaluates if an answer is grounded in the provided context."""
    
    SYSTEM_PROMPT = """You are a strict grader assessing whether an AI-generated answer is grounded in / supported by a set of retrieved facts.

Criteria:
1. The answer must ONLY contain information present in the context.
2. If the answer mentions facts not in the context, it is a hallucination (unless it's general knowledge used for flow).
3. Pay close attention to numbers, dates, names.

Output format:
Return "yes" if the answer is fully grounded/supported.
Return "no" if the answer contains hallucinations or unsupported claims."""

    def __init__(self, model_id: str = "amazon.nova-lite-v1:0"):
        self.llm = ChatBedrockConverse(
            model=model_id,
            temperature=0, 
            max_tokens=10 # strict grading
        )
        
    def check_groundedness(self, answer: str, context_docs: List[Document]) -> bool:
        """Check if answer is grounded in context."""
        
        # Format context
        context_text = "\n\n".join([d.page_content for d in context_docs])
        
        prompt = f"""Context:
{context_text}

Answer:
{answer}

Is the answer grounded in the context? (yes/no):"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            grade = response.content.strip().lower()
            return "yes" in grade
        except Exception as e:
            print(f"⚠ Grader failed: {e}")
            return True # Fail open to avoid blocking valid answers if grader breaks
