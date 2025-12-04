"""
The Cortex MCP Server
ATLAS's semantic search interface to The Cortex knowledge base

A proper MCP server using FastMCP with streamable-http transport.
Searches the Supabase vector database containing:
- 776 ICT trading transcripts (19,260 chunks)
- 3 Vanessa Van Edwards books (897 chunks)
- Total: 20,829 searchable chunks
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from supabase import create_client

# Initialize FastMCP server
mcp = FastMCP(
    "cortex_mcp",
    instructions="""You are connected to The Cortex - ATLAS's comprehensive knowledge base.
    
The Cortex contains:
- 776 ICT (Inner Circle Trader) transcripts covering order blocks, liquidity, market structure, 
  Fair Value Gaps, optimal trade entry, trading psychology, and all ICT methodology
- 3 Vanessa Van Edwards books on social intelligence, body language, and communication

When Mr. Pak asks about trading concepts, ICT methodology, social skills, or communication 
strategies, use the search_cortex tool to retrieve relevant teachings.

Always cite your sources: "According to ICT in [transcript name]..." or 
"Vanessa Van Edwards teaches in [book name]..."
"""
)

# Global clients (initialized lazily)
_openai_client: Optional[OpenAI] = None
_supabase_client = None


def get_openai() -> OpenAI:
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_supabase():
    """Get or create Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
        _supabase_client = create_client(url, key)
    return _supabase_client


def get_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI text-embedding-3-small"""
    client = get_openai()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def perform_search(
    query: str,
    match_count: int = 5,
    match_threshold: float = 0.5,
    source_filter: Optional[str] = None
) -> list[dict]:
    """
    Execute semantic search against The Cortex
    
    Args:
        query: Search query text
        match_count: Number of results to return
        match_threshold: Minimum similarity score (0-1)
        source_filter: "ict", "vanessa", or None for all
    
    Returns:
        List of matching chunks with content, source, and similarity
    """
    supabase = get_supabase()
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Call Supabase vector search function
    response = supabase.rpc(
        'search_ict_knowledge',
        {
            'query_embedding': query_embedding,
            'match_threshold': match_threshold,
            'match_count': match_count
        }
    ).execute()
    
    results = response.data or []
    
    # Apply source filter if specified
    if source_filter:
        filter_lower = source_filter.lower()
        if filter_lower == "ict":
            # Exclude Vanessa Van Edwards content
            vanessa_indicators = ["vanessa", "cues", "captivate", "lie detection"]
            results = [
                r for r in results 
                if not any(ind in r.get("source_transcript", "").lower() for ind in vanessa_indicators)
            ]
        elif filter_lower == "vanessa":
            # Only Vanessa Van Edwards content
            vanessa_indicators = ["vanessa", "cues", "captivate", "lie detection"]
            results = [
                r for r in results 
                if any(ind in r.get("source_transcript", "").lower() for ind in vanessa_indicators)
            ]
    
    return results


# Pydantic models for tool inputs
class SearchCortexInput(BaseModel):
    """Input parameters for searching The Cortex knowledge base."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    query: str = Field(
        ...,
        description="The search query - describe the concept, topic, or question you want to find information about. Be specific for better results.",
        min_length=2,
        max_length=500
    )
    source: Optional[str] = Field(
        default=None,
        description="Filter results by source: 'ict' for ICT trading methodology only, 'vanessa' for Vanessa Van Edwards social skills only, or omit/null for all sources"
    )
    num_results: Optional[int] = Field(
        default=5,
        description="Number of results to return (1-10)",
        ge=1,
        le=10
    )


class GetStatsInput(BaseModel):
    """Input for getting Cortex statistics."""
    pass


# MCP Tools
@mcp.tool(
    name="search_cortex",
    annotations={
        "title": "Search The Cortex Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def search_cortex(params: SearchCortexInput) -> str:
    """
    Search The Cortex knowledge base for ICT trading concepts and Vanessa Van Edwards social intelligence frameworks.
    
    Use this tool when Mr. Pak asks about:
    - ICT trading methodology (order blocks, liquidity, Fair Value Gaps, market structure, optimal trade entry)
    - Trading psychology and discipline
    - Social skills, body language, and communication strategies
    - Conversation techniques and first impressions
    
    Returns relevant excerpts with source citations.
    
    Args:
        params: SearchCortexInput containing query, optional source filter, and num_results
    
    Returns:
        Formatted string with search results including content excerpts and sources
    """
    try:
        results = perform_search(
            query=params.query,
            match_count=params.num_results or 5,
            match_threshold=0.5,
            source_filter=params.source
        )
        
        if not results:
            return f"No results found in The Cortex for: '{params.query}'\n\nTry rephrasing your query or broadening your search terms."
        
        # Format results for Claude
        output_parts = [f"## The Cortex Search Results\n**Query:** {params.query}\n**Results Found:** {len(results)}\n"]
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "").strip()
            source = result.get("source_transcript", "Unknown Source")
            similarity = result.get("similarity", 0)
            
            # Truncate very long content
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            output_parts.append(f"""
---
### Result {i} (Relevance: {similarity:.1%})
**Source:** {source}

{content}
""")
        
        return "\n".join(output_parts)
        
    except Exception as e:
        return f"Error searching The Cortex: {str(e)}"


@mcp.tool(
    name="cortex_stats",
    annotations={
        "title": "Get Cortex Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def cortex_stats(params: GetStatsInput) -> str:
    """
    Get statistics about The Cortex knowledge base.
    
    Returns the total number of searchable chunks and confirms database connectivity.
    
    Args:
        params: Empty input (no parameters needed)
    
    Returns:
        String with Cortex statistics
    """
    try:
        supabase = get_supabase()
        response = supabase.table('ict_chunks').select('id', count='exact').execute()
        total_chunks = response.count or 0
        
        return f"""## The Cortex Statistics

**Status:** Connected
**Total Searchable Chunks:** {total_chunks:,}

**Content Breakdown:**
- ICT Trading Transcripts: ~776 files (~19,260 chunks)
- Vanessa Van Edwards Books: 3 books (~897 chunks)

**Available Sources:**
- ICT methodology: Order blocks, liquidity, market structure, FVG, OTE, trading psychology
- Vanessa Van Edwards: Cues, Captivate, body language, social intelligence
"""
    except Exception as e:
        return f"Error connecting to The Cortex: {str(e)}"


# Run the server
if __name__ == "__main__":
    import sys
    
    # Get port from environment (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    print(f"Starting The Cortex MCP Server on port {port}...")
    print(f"Transport: streamable-http")
    
    # Run with streamable-http transport for remote access
    mcp.run(transport="streamable-http", port=port)
