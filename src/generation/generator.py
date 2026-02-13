"""Answer generator using AWS Bedrock with retrieved context."""

import os
from typing import List
try:
    from langchain_aws import ChatBedrockConverse
except ImportError:
    # Fallback or error if not available (but we verified it is)
    from langchain_aws import ChatBedrock as ChatBedrockConverse
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


class AnswerGenerator:
    """Generate answers using AWS Bedrock Claude with retrieved context."""

    SYSTEM_PROMPT = """You are an AI assistant for company leadership. Your role is to answer questions about the company's performance, status, and operations based on the provided documents.

Guidelines:
- Provide concise, factual answers grounded in the provided context
- If the information is not in the context, clearly state that you don't have enough information
- Cite specific sources when possible (e.g., "According to the Q4 report...")
- Focus on key insights relevant to leadership decision-making
- Use clear, professional language
- Think step-by-step before answering to ensure logical consistency"""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        region: str = None,
    ):
        """Initialize the answer generator.

        Args:
            model: Bedrock model ID to use
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            region: AWS region (defaults to AWS_REGION env var)
        """
        self.model = model or os.getenv("LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "1000"))
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        # Initialize credentials config
        credentials_config = {}

        if bearer_token:
            # Use bearer token if provided
            print(f"✓ Using AWS Bedrock with bearer token authentication")
            credentials_config["region_name"] = self.region
        elif aws_access_key and aws_secret_key:
            # Use access key/secret key
            print(f"✓ Using AWS Bedrock with access key authentication")
            credentials_config = {
                "region_name": self.region,
            }
            # boto3 will automatically pick up AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from environment
        else:
            # Try using default credentials (IAM role, etc.)
            print(f"✓ Using AWS Bedrock with default credentials (IAM role)")
            credentials_config["region_name"] = self.region

        self.llm = ChatBedrockConverse(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **credentials_config
        )

        print(f"✓ Initialized Bedrock LLM: {self.model}")

    def generate_answer(
        self, query: str, context_documents: List[Document]
    ) -> dict:
        """Generate an answer based on query and retrieved documents.

        Args:
            query: User's question
            context_documents: Retrieved relevant documents

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Format context from documents
        context = self._format_context(context_documents)

        # Build prompt
        user_prompt = f"""Context from company documents:
{context}

Question: {query}

Please provide a concise, factual answer based on the context above. If the information is not available in the context, clearly state that."""

        # Generate response
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)

        # Extract sources
        sources = self._extract_sources(context_documents)

        return {
            "answer": response.content,
            "sources": sources,
            "context_used": len(context_documents),
        }

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "N/A")

            context_parts.append(
                f"[{i}] From {source} (Page {page}):\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def _extract_sources(self, documents: List[Document]) -> List[dict]:
        """Extract source information from documents.

        Args:
            documents: List of Document objects

        Returns:
            List of source dictionaries
        """
        sources = []
        seen = set()

        for doc in documents:
            source_key = (
                doc.metadata.get("filename", "Unknown"),
                doc.metadata.get("page", "N/A"),
            )

            if source_key not in seen:
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                })
                seen.add(source_key)

        return sources
