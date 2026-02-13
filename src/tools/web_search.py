"""Web search tool using DuckDuckGo."""

from typing import List, Dict, Any, Union
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

class WebSearchTool:
    """Search the web for information."""

    def __init__(self, max_results: int = 5):
        """Initialize web search tool.
        
        Args:
            max_results: Number of results to return
        """
        self.max_results = max_results

    def search(self, query: Union[str, List[str]]) -> List[Dict[str, str]]:
        """Search for a query or list of queries."""
        if DDGS is None:
            print("⚠ duckduckgo-search not installed or import failed.")
            return []

        queries = [query] if isinstance(query, str) else query
        all_results = []
        
        try:
            with DDGS() as ddgs:
                for q in queries:
                    print(f"Internet Search: {q}")
                    # DDGS.text() returns a generator of dicts
                    # Limit per query to avoid overwhelming context
                    limit = 3 if len(queries) > 1 else self.max_results
                    ddgs_gen = ddgs.text(q, max_results=limit)
                    
                    count = 0
                    for r in ddgs_gen:
                        all_results.append({
                            "title": r.get("title", ""),
                            "href": r.get("href", ""),
                            "body": r.get("body", ""),
                            "query": q
                        })
                        count += 1
                        
            print(f"✓ Found {len(all_results)} total web results")
            return all_results
            
        except Exception as e:
            print(f"✗ Web search failed: {e}")
            return []

    def format_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results into a context string."""
        if not results:
            return "No web search results found."
            
        formatted = []
        for i, res in enumerate(results, 1):
            q_info = f" [Query: {res.get('query', 'Main')}]" if 'query' in res else ""
            formatted.append(
                f"[Web {i}]{q_info} {res['title']}\n"
                f"URL: {res['href']}\n"
                f"Snippet: {res['body']}\n"
            )
            
        return "\n".join(formatted)
