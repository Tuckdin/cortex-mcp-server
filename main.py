"""
The Cortex MCP Server
A Model Context Protocol server for semantic search over ICT trading knowledge
and Vanessa Van Edwards social intelligence content.
"""

import os
import json
import logging
from typing import Optional

from fastmcp import FastMCP
from openai import OpenAI
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

mcp = FastMCP(
    "cortex_mcp",
    instructions="""
    The Cortex MCP Server provides semantic search over Mr. Pak's curated knowledge base:
    - ICT Trading Knowledge: 776 transcripts covering order blocks, liquidity, market structure
    - Social Intelligence: Vanessa Van Edwards' books on communication and social skills
    Use search_cortex to find relevant information.
    """
)

# =============================================================================
# LAZY CLIENT INITIALIZATION
# =============================================================================

_openai_client: Optional[OpenAI] = None
_supabase_client: Optional[Client] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        _supabase_client = create_client(url, key)
    return _supabase_client


def generate_embedding(text: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def search_cortex(query: str, source: Optional[str] = None, num_results: int = 5) -> str:
    """
    Search The Cortex knowledge base for relevant information.
    
    Args:
        query: The search query - be specific for best results.
        source: Optional filter - "ict" for trading, "vanessa" for social skills.
        num_results: Number of results (1-10, default 5).
    """
    try:
        num_results = max(1, min(10, num_results))
        query_embedding = generate_embedding(query)
        supabase = get_supabase_client()
        
        # Use the search_ict_knowledge RPC function
        response = supabase.rpc('search_ict_knowledge', {
            'query_embedding': query_embedding,
            'match_threshold': 0.5,
            'match_count': num_results
        }).execute()
        
        if not response.data:
            return json.dumps({"status": "no_results", "message": f"No results for: {query}"})
        
        results = response.data
        if source:
            source_lower = source.lower()
            if source_lower == "ict":
                results = [r for r in results if "vanessa" not in r.get("source_transcript", "").lower()]
            elif source_lower == "vanessa":
                results = [r for r in results if "vanessa" in r.get("source_transcript", "").lower()]
        
        # Note: search function returns source_transcript, not source
        formatted = [{"rank": i, "content": r.get("content", ""), "source": r.get("source_transcript", ""), 
                      "similarity": round(r.get("similarity", 0), 4)} for i, r in enumerate(results, 1)]
        
        return json.dumps({"status": "success", "query": query, "results": formatted}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_cortex_stats() -> str:
    """Get statistics about The Cortex knowledge base."""
    try:
        supabase = get_supabase_client()
        # Correct table name: ict_chunks
        total_response = supabase.table('ict_chunks').select('id', count='exact').execute()
        total_count = total_response.count if total_response.count else 0
        return json.dumps({"status": "success", "total_chunks": total_count})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def search_ict(query: str, num_results: int = 5) -> str:
    """Search specifically for ICT trading content."""
    return search_cortex(query=query, source="ict", num_results=num_results)


@mcp.tool()
def search_social(query: str, num_results: int = 5) -> str:
    """Search specifically for Vanessa Van Edwards social intelligence content."""
    return search_cortex(query=query, source="vanessa", num_results=num_results)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting The Cortex MCP Server on port {port}...")
    
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=port
    )
