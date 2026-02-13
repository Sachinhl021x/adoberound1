"""Research agent logic for query decomposition and synthesis."""

import os
from typing import List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_aws import ChatBedrockConverse

class ResearchRefiner:
    """Decomposes queries into searchable sub-questions."""
    
    SYSTEM_PROMPT = """You are a Senior Research Architect. Your goal is to break down a user question into specific, searchable web queries to find missing information.

Guidelines:
1. Generate 3 distinct, high-quality search queries.
2. Focus on factual data, recent events, or specific metrics.
3. queries should be optimized for search engines (keyword-heavy, specific).
4. Return ONLY the list of queries, one per line. No numbering or bullets."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        # Initialize credentials config
        credentials_config = {}

        if bearer_token:
            credentials_config["region_name"] = region
        elif aws_access_key and aws_secret_key:
            credentials_config = {"region_name": region}
        else:
            credentials_config["region_name"] = region

        self.llm = ChatBedrockConverse(
            model=self.model,
            temperature=0,
            max_tokens=500,
            **credentials_config
        )

    def generate_queries(self, question: str) -> List[str]:
        """Generate research queries for a question."""
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"User Question: {question}\n\nGenerate 3 search queries:")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            # Split by line and clean up
            queries = [q.strip().lstrip("- ").lstrip("123. ") for q in content.split("\n") if q.strip()]
            print(f"✓ Generated {len(queries)} research queries")
            return queries[:3] # Ensure max 3
        except Exception as e:
            print(f"⚠ Failed to generate research queries: {e}")
            return [question] # Fallback to original

class ResearchSynthesizer:
    """Synthesizes web search results into a cohesive answer."""
    
    SYSTEM_PROMPT = """You are a Strategic Research Analyst. You have been provided with raw web search results to answer a user's question.

Guidelines:
1. Synthesize a comprehensive answer based ONLY on the provided Context.
2. CITATION RULE: You must cite your sources. Use [Web 1], [Web 2] format at the end of sentences.
3. If conflicts exist between sources, note them.
4. Organize the answer logically with clear headings.
5. If the context is insufficient, state what is missing.
6. Professional, objective tone."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        # Initialize credentials config
        credentials_config = {}

        if bearer_token:
            credentials_config["region_name"] = region
        elif aws_access_key and aws_secret_key:
            credentials_config = {"region_name": region}
        else:
            credentials_config["region_name"] = region

        self.llm = ChatBedrockConverse(
            model=self.model,
            temperature=0.1,
            max_tokens=2000,
            **credentials_config
        )

    def synthesize(self, question: str, web_context: str) -> str:
        """Synthesize answer from web context."""
        prompt = f"""User Question: {question}

Web Search Context:
{web_context}

Please provide a detailed report answering the question."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error synthesizing research: {e}"
