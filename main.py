"""
The Cortex MCP Server
A Model Context Protocol server for semantic search over ICT trading knowledge
and Vanessa Van Edwards social intelligence content.

This server provides Claude Desktop with direct access to The Cortex vector database.
"""

import os
import json
import logging
from typing import Optional

# MCP imports
from mcp.server.fastmcp import FastMCP

# External service imports
from openai import OpenAI
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

mcp = FastMCP(
    "cortex_mcp",
    instructions="""
    The Cortex MCP Server provides semantic search over Mr. Pak's curated knowledge base:
    
    - ICT Trading Knowledge: 776 transcripts (19,260 chunks) covering Inner Circle Trader 
      concepts including order blocks, liquidity, market structure, and trading psychology.
    
    - Social Intelligence: Vanessa Van Edwards' books (897 chunks) on communication, 
      body language, and social skills.
    
    Use search_cortex to find relevant information. Always cite sources when providing
    information from The Cortex.
    """
)

# =============================================================================
# LAZY CLIENT INITIALIZATION
# =============================================================================

_openai_client: Optional[OpenAI] = None
_supabase_client: Optional[Client] = None


def get_openai_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    return _openai_client


def get_supabase_client() -> Client:
    """Lazy initialization of Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        _supabase_client = create_client(url, key)
        logger.info("Supabase client initialized")
    return _supabase_client


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def generate_embedding(text: str) -> list[float]:
    """Generate embedding vector using OpenAI's text-embedding-3-small model."""
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def search_cortex(
    query: str,
    source: Optional[str] = None,
    num_results: int = 5
) -> str:
    """
    Search The Cortex knowledge base for relevant information.
    
    Args:
        query: The search query - be specific for best results.
        source: Optional filter - "ict" for trading content, "vanessa" for social skills, 
                or None for all sources.
        num_results: Number of results to return (1-10, default 5).
    
    Returns:
        Formatted search results with source citations.
    
    Examples:
        - search_cortex("order blocks") - Find ICT teachings on order blocks
        - search_cortex("first impressions", source="vanessa") - Social skills content
        - search_cortex("FOMO trading psychology", source="ict") - Trading psychology
    """
    try:
        # Validate inputs
        num_results = max(1, min(10, num_results))
        
        # Generate embedding for the query
        logger.info(f"Searching Cortex for: {query}")
        query_embedding = generate_embedding(query)
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Perform vector search
        response = supabase.rpc(
            'search_ict_knowledge',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': num_results
            }
        ).execute()
        
        if not response.data:
            return json.dumps({
                "status": "no_results",
                "message": f"No results found for query: {query}",
                "suggestion": "Try rephrasing your query or using different keywords."
            }, indent=2)
        
        # Filter by source if specified
        results = response.data
        if source:
            source_lower = source.lower()
            if source_lower == "ict":
                results = [r for r in results if "vanessa" not in r.get("source", "").lower()]
            elif source_lower == "vanessa":
                results = [r for r in results if "vanessa" in r.get("source", "").lower()]
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "content": result.get("content", ""),
                "source": result.get("source", "Unknown"),
                "similarity": round(result.get("similarity", 0), 4)
            })
        
        return json.dumps({
            "status": "success",
            "query": query,
            "source_filter": source,
            "result_count": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2)


@mcp.tool()
def get_cortex_stats() -> str:
    """
    Get statistics about The Cortex knowledge base.
    
    Returns:
        Database statistics including chunk counts by source.
    """
    try:
        supabase = get_supabase_client()
        
        # Get total count
        total_response = supabase.table('ict_knowledge').select('id', count='exact').execute()
        total_count = total_response.count if total_response.count else 0
        
        return json.dumps({
            "status": "success",
            "statistics": {
                "total_chunks": total_count,
                "sources": {
                    "ict_transcripts": {
                        "description": "ICT (Inner Circle Trader) video transcripts",
                        "topics": ["Order blocks", "Liquidity", "Market structure", "Trading psychology", "SMC concepts"]
                    },
                    "vanessa_van_edwards": {
                        "description": "Social intelligence and communication books",
                        "topics": ["Body language", "First impressions", "Conversation skills", "Charisma"]
                    }
                },
                "embedding_model": "text-embedding-3-small",
                "last_updated": "2024-12"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return json.dumps({
            "status": "error", 
            "message": str(e)
        }, indent=2)


@mcp.tool()
def search_ict(query: str, num_results: int = 5) -> str:
    """
    Search specifically for ICT (Inner Circle Trader) content.
    
    This is a convenience wrapper around search_cortex with source="ict".
    Use this for trading-related queries about order blocks, liquidity,
    market structure, and ICT methodology.
    
    Args:
        query: The search query about ICT/trading concepts.
        num_results: Number of results to return (1-10, default 5).
    
    Returns:
        Formatted search results from ICT content only.
    """
    return search_cortex(query=query, source="ict", num_results=num_results)


@mcp.tool()
def search_social(query: str, num_results: int = 5) -> str:
    """
    Search specifically for social intelligence content (Vanessa Van Edwards).
    
    This is a convenience wrapper around search_cortex with source="vanessa".
    Use this for queries about communication, body language, social skills,
    and interpersonal dynamics.
    
    Args:
        query: The search query about social/communication topics.
        num_results: Number of results to return (1-10, default 5).
    
    Returns:
        Formatted search results from Vanessa Van Edwards content only.
    """
    return search_cortex(query=query, source="vanessa", num_results=num_results)


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting The Cortex MCP Server...")
    logger.info(f"Port: {port} (from environment)")
    logger.info(f"Transport: streamable-http")
    
    # FastMCP reads PORT from environment automatically
    # We just specify transport and host
    mcp.run(transport="streamable-http", host="0.0.0.0")
