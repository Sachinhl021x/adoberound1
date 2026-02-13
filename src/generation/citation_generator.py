"""Enhanced answer generator with citation tracking and source highlighting."""

import os
from typing import List, Dict, Any
from langchain_core.documents import Document

try:
    from langchain_aws import ChatBedrockConverse
except ImportError:
    # Fallback or error if not available (but we verified it is)
    from langchain_aws import ChatBedrock as ChatBedrockConverse


class CitationAwareGenerator:
    """Generate answers with precise citation tracking."""

    SYSTEM_PROMPT = """You are an AI assistant for company leadership. Your role is to answer questions based on provided documents with PRECISE CITATIONS.

CRITICAL INSTRUCTIONS:
1. When you reference information, cite the specific source number in square brackets [1], [2], etc.
2. Use multiple citations if information comes from multiple sources [1][2]
3. Place citations immediately after the relevant statement
4. Be specific about which source supports which claim
5. If information is not in the provided context, clearly state "I don't have information about this in the provided documents."

Example:
"Adobe's Q4 revenue was $5.61 billion [1], representing a 10% year-over-year increase [1][2]. The Digital Media segment contributed $3.95 billion [2]."

Guidelines:
- Provide concise, factual answers grounded in the provided context
- Focus on key insights relevant to leadership decision-making
- Use clear, professional language
- Always cite your sources with [number] format
- Think step-by-step to connect facts from different sources logically"""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        region: str = None,
    ):
        """Initialize citation-aware generator."""
        self.model = model or os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0")
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "1000"))
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        # Initialize Bedrock client
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

        credentials_config = {}
        if bearer_token:
            print(f"✓ Using AWS Bedrock with bearer token authentication")
            credentials_config["region_name"] = self.region
        elif aws_access_key and aws_secret_key:
            print(f"✓ Using AWS Bedrock with access key authentication")
            credentials_config = {"region_name": self.region}
        else:
            print(f"✓ Using AWS Bedrock with default credentials (IAM role)")
            credentials_config["region_name"] = self.region

        self.llm = ChatBedrockConverse(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **credentials_config
        )

        print(f"✓ Initialized Bedrock LLM with citations: {self.model}")

    def generate_answer_with_citations(
        self,
        query: str,
        context_documents: List[Document]
    ) -> Dict[str, Any]:
        """Generate answer with inline citations.

        Args:
            query: User's question
            context_documents: Retrieved relevant documents

        Returns:
            Dictionary with:
                - answer: Text with inline citations [1], [2], etc.
                - sources: List of source objects with metadata
                - citations: Mapping of citation numbers to source details
                - context_used: Number of documents used
        """
        # Build numbered context with sources
        context_parts = []
        sources = []

        for i, doc in enumerate(context_documents, 1):
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chunk_type = doc.metadata.get("type", "text")
            section = doc.metadata.get("section_heading", "")

            # Build source reference
            source_ref = f"[{i}] {filename}"
            if page != "N/A":
                source_ref += f" (Page {page})"
            if section:
                source_ref += f" - {section}"
            if chunk_type != "text":
                source_ref += f" [{chunk_type}]"

            context_parts.append(f"{source_ref}:\n{doc.page_content}")

            # Store source metadata
            sources.append({
                "citation_number": i,
                "filename": filename,
                "page": page,
                "type": chunk_type,
                "section": section,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        context = "\n\n".join(context_parts)

        # Build prompt
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        user_prompt = f"""Context from company documents:
{context}

Question: {query}

Provide a precise answer with inline citations [1], [2], etc. referencing the numbered sources above."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Generate response
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": sources,
            "citations": self._extract_citations_from_answer(response.content, sources),
            "context_used": len(context_documents),
        }

    def _extract_citations_from_answer(
        self,
        answer: str,
        sources: List[Dict]
    ) -> Dict[int, Dict]:
        """Extract citation numbers from answer and map to sources.

        Args:
            answer: Generated answer with citations like [1], [2]
            sources: List of source metadata

        Returns:
            Mapping of citation number to source details
        """
        import re

        # Find all citation patterns [1], [2], etc.
        citations_found = re.findall(r'\[(\d+)\]', answer)
        citation_map = {}

        for citation_num_str in set(citations_found):
            citation_num = int(citation_num_str)
            # Find corresponding source
            for source in sources:
                if source["citation_number"] == citation_num:
                    citation_map[citation_num] = source
                    break

        return citation_map
