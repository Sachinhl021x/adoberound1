"""Agentic RAG using LangGraph."""

import operator
from typing import Annotated, Sequence, TypedDict, List, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Import our existing components
from src.retrieval import DocumentRetriever
from src.generation import AnswerGenerator
from src.tools import WebSearchTool
from src.eval.hallucination_grader import HallucinationGrader
from src.eval.relevance_grader import RelevanceGrader
from src.agent.research import ResearchRefiner, ResearchSynthesizer

class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: Sequence[Document]
    question: str
    retry_count: int
    context: str
    answer: str
    sources: List[dict]
    context_used: int
    research_queries: List[str] # List of sub-queries for research
    is_web: bool
    hallucination_grade: str

class AgentGraph:
    """Graph workflow for the autonomous agent."""

    def __init__(self, retriever: DocumentRetriever, generator: AnswerGenerator):
        """Initialize the agent graph.
        
        Args:
            retriever: DocumentRetriever instance
            generator: AnswerGenerator instance
        """
        self.retriever = retriever
        self.generator = generator
        self.web_search_tool = WebSearchTool()
        self.grader = HallucinationGrader()
        self.relevance_grader = RelevanceGrader()
        self.research_refiner = ResearchRefiner()
        self.research_synthesizer = ResearchSynthesizer()
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("plan_research", self.plan_research)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("synthesize_research", self.synthesize_research)
        workflow.add_node("check_hallucination", self.check_hallucination)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge from grader
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "plan_research": "plan_research",
            },
        )
        
        workflow.add_edge("transform_query", "retrieve")
        # workflow.add_edge("generate", END) # Changed to check hallucination
        workflow.add_edge("generate", "check_hallucination")
        
        # Conditional edge from hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            self.decide_after_check,
            {
                "end": END,
                "plan_research": "plan_research",
            }
        )
        
        workflow.add_edge("plan_research", "web_search")
        workflow.add_edge("web_search", "synthesize_research")
        workflow.add_edge("synthesize_research", END)
        
        # Compile
        self.app = workflow.compile()

    def retrieve(self, state: AgentState):
        """Retrieve documents based on the current question."""
        print(f"---RETRIEVE---")
        question = state["question"]
        
        # Use hybrid retrieval from our enhanced retriever
        documents = self.retriever.retrieve(question)
        print(f"Retrieved {len(documents)} documents for: {question}")
        
        return {"documents": documents, "question": question}

    def grade_documents(self, state: AgentState):
        """Grade retrieved documents for relevance."""
        print(f"---CHECK RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        
        # Grade documents using LLM
        grade = self.relevance_grader.grade(question, documents)
        
        if grade == "irrelevant":
           print(f"⚠ Documents graded IRRELEVANT for query: {question}")
           # If irrelevant, we treat it same as empty -> likely transform or web search
           return {"documents": [], "retry_count": retry_count}
           
        print("✓ Documents graded RELEVANT")
        return {"documents": documents, "retry_count": retry_count}

    def decide_to_generate(self, state: AgentState):
        """Determine whether to generate an answer or re-frame the question."""
        print(f"---DECIDE TO GENERATE---")
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        max_retries = 1 # Limit retries to prevent loops
        
        if not documents:
            # No relevant documents
            if retry_count < max_retries:
                print(f"DECISION: Transform Query (Attempt {retry_count + 1})")
                return "transform_query"
            else:
                print("DECISION: Max retries reached, falling back to Web Search (Deep Research).")
                return "plan_research"
        
        print("DECISION: Generate")
        return "generate"

    def transform_query(self, state: AgentState):
        """Transform the query to produce a better question."""
        print(f"---TRANSFORM QUERY---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        
        # Simple transformation: keyword extraction or rephrasing
        # In a real implementation use LLM to re-write
        # For now, append "overview" or simplify
        # Let's try to make it more broad
        
        better_question = f"{question} overview details" 
        
        print(f"Transformed query from '{question}' to '{better_question}'")
        
        return {"question": better_question, "retry_count": retry_count + 1}

    def generate(self, state: AgentState):
        """Generate answer."""
        print(f"---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Use our existing AnswerGenerator
        result = self.generator.generate_answer(question, documents)
        answer = result["answer"]
        
        return {
            "messages": [AIMessage(content=answer)],
            "context": self.retriever.format_context(documents),
            "answer": answer,
            "sources": result.get("sources", []),
            "context_used": result.get("context_used", 0)
        }

    def plan_research(self, state: AgentState):
        """Decompose query into research plan."""
        print(f"---PLAN RESEARCH---")
        question = state["question"]
        
        # Decompose using ResearchRefiner
        queries = self.research_refiner.generate_queries(question)
        
        return {"research_queries": queries}

    def web_search(self, state: AgentState):
        """Execute web search for planned queries."""
        print(f"---WEB SEARCH---")
        # Use planned queries if available, else fallback to main question
        queries = state.get("research_queries") or [state["question"]]
        
        # Execute search (tool now handles list)
        results = self.web_search_tool.search(queries)
        
        # Convert web results to pseudo-Documents for the generator
        documents = []
        from langchain_core.documents import Document
        for res in results:
            documents.append(Document(
                page_content=f"{res['title']}\n{res['body']}",
                metadata={"filename": "Web Search", "page": res['href'], "type": "web", "query": res.get("query", "Main")}
            ))
            
        return {"documents": documents, "is_web": True} # Flag to indicate web source
    
    def synthesize_research(self, state: AgentState):
        """Synthesize web results into a final answer."""
        print(f"---SYNTHESIZE RESEARCH---")
        question = state["question"]
        documents = state["documents"]
        
        # Format context from web docs
        web_context = self.web_search_tool.format_results([
            {"title": d.page_content.split("\n")[0], "body": d.page_content.split("\n")[1], "href": d.metadata["page"], "query": d.metadata.get("query", "")} 
            for d in documents
        ])
        
        # Synthesize using ResearchSynthesizer
        answer = self.research_synthesizer.synthesize(question, web_context)
        
        return {
            "messages": [AIMessage(content=answer)],
            "answer": answer,
            "context": web_context,
            "sources": [{"filename": "Web Search", "page": d.metadata["page"]} for d in documents],
        }

    def check_hallucination(self, state: AgentState):
        """Check if the answer is grounded."""
        print(f"---CHECK HALLUCINATION---")
        answer = state["answer"]
        documents = state["documents"]
        
        # If no docs, skip check (nothing to ground on, likely already apologized or used web)
        if not documents:
            return {"hallucination_grade": "grounded"}
            
        # Check for refusal/utility first to catch "I don't know" answers
        refusal_indicators = [
            "i don't have enough information",
            "context does not contain",
            "information is not available",
            "cannot provide an answer",
            "cannot answer",
            "answer based on the context provided",
        ]
        
        print(f"AG_LOG: Check Hallucination. Answer: {answer[:100]}...")
        
        if any(indicator in answer.lower() for indicator in refusal_indicators):
             print(f"AG_LOG: ⚠ Answer is a REFUSAL. Triggering Web Search Fallback.")
             return {"hallucination_grade": "hallucinated"} # Trigger fallback
            
        is_grounded = self.grader.check_groundedness(answer, documents)
        print(f"AG_LOG: Groundedness check result: {is_grounded}")
        
        if is_grounded:
            print("✓ Answer is grounded.")
            return {"hallucination_grade": "grounded"}
        else:
            print("⚠ Answer is NOT grounded (Hallucination detected).")
            return {"hallucination_grade": "hallucinated"}

    def decide_after_check(self, state: AgentState):
        """Decide what to do after hallucination check."""
        grade = state.get("hallucination_grade", "grounded")
        retry_count = state.get("retry_count", 0)
        
        if grade == "grounded":
            return "end"
        
        # If hallucinated, try web search if we haven't already (or just give up to avoid loops)
        # Check if we already did web search
        if state.get("is_web", False):
            # If we hallucinated even with web search, just end (or apologize)
            print("AG_LOG: Hallucination even with web search. Ending.")
            return "end"
            
        print("AG_LOG: Hallucination detected on internal docs. Falling back to Deep Research.")
        return "plan_research"
