"""
The Cortex MCP Server - WITH INGESTION CAPABILITY
A Model Context Protocol server for semantic search AND content ingestion
over ICT trading knowledge and Vanessa Van Edwards social intelligence content.
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
    The Cortex MCP Server provides semantic search AND ingestion over Mr. Pak's knowledge base:
    - ICT Trading Knowledge: 776+ transcripts covering order blocks, liquidity, market structure
    - Social Intelligence: Vanessa Van Edwards' books on communication and social skills
    
    Tools available:
    - search_cortex: Semantic search with optional source filtering
    - search_text: Direct text search for exact phrase matching
    - ingest_content: Add new content to The Cortex
    - get_cortex_stats: Database statistics
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


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by approximate word count."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        
    return chunks


# =============================================================================
# MCP TOOLS - SEARCH
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
        
        # Use lower threshold (0.35) for better recall
        response = supabase.rpc('search_ict_knowledge', {
            'query_embedding': query_embedding,
            'match_threshold': 0.35,
            'match_count': num_results * 2  # Get extra for filtering
        }).execute()
        
        if not response.data:
            return json.dumps({"status": "no_results", "message": f"No results for: {query}"})
        
        results = response.data
        
        # Apply source filter if specified
        if source:
            source_lower = source.lower()
            if source_lower == "ict":
                results = [r for r in results if "vanessa" not in r.get("source_transcript", "").lower()
                          and "cues" not in r.get("source_transcript", "").lower()
                          and "captivate" not in r.get("source_transcript", "").lower()]
            elif source_lower == "vanessa":
                results = [r for r in results if any(term in r.get("source_transcript", "").lower() 
                          for term in ["vanessa", "cues", "captivate", "lie detection"])]
        
        results = results[:num_results]
        
        formatted = [
            {"rank": i, "content": r.get("content", ""), "source": r.get("source_transcript", ""), 
             "similarity": round(r.get("similarity", 0), 4)}
            for i, r in enumerate(results, 1)
        ]
        
        return json.dumps({"status": "success", "query": query, "results": formatted}, indent=2)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def search_text(query: str, num_results: int = 10) -> str:
    """
    Direct text search for exact phrase matching.
    Use this when semantic search fails to find specific terms like 'turtle soup'.
    
    Args:
        query: The exact text to search for.
        num_results: Number of results (1-20, default 10).
    """
    try:
        num_results = max(1, min(20, num_results))
        supabase = get_supabase_client()
        
        response = supabase.table('ict_chunks') \
            .select('id, content, source_transcript') \
            .ilike('content', f'%{query}%') \
            .limit(num_results) \
            .execute()
        
        if not response.data:
            return json.dumps({"status": "no_results", "message": f"No exact matches for: {query}"})
        
        formatted = [
            {"rank": i, "content": r.get("content", ""), "source": r.get("source_transcript", "")}
            for i, r in enumerate(response.data, 1)
        ]
        
        return json.dumps({"status": "success", "query": query, "results": formatted, "count": len(formatted)}, indent=2)
    except Exception as e:
        logger.error(f"Text search error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# MCP TOOLS - INGESTION
# =============================================================================

@mcp.tool()
def ingest_content(content: str, source_name: str) -> str:
    """
    Ingest new content into The Cortex.
    Chunks the content, generates embeddings, and stores in Supabase.
    
    Args:
        content: The full text content to ingest.
        source_name: The source name (e.g., "ICT Turtle Soup Lecture.txt")
    
    Returns:
        Status message with number of chunks ingested.
    """
    try:
        if not content or len(content.strip()) < 100:
            return json.dumps({"status": "error", "message": "Content too short (min 100 chars)"})
        
        # Chunk the content
        chunks = chunk_text(content.strip())
        logger.info(f"Ingesting {len(chunks)} chunks from: {source_name}")
        
        supabase = get_supabase_client()
        ingested = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = generate_embedding(chunk)
                
                # Insert into Supabase
                supabase.table('ict_chunks').insert({
                    'content': chunk,
                    'embedding': embedding,
                    'source_transcript': source_name,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }).execute()
                
                ingested += 1
                
            except Exception as e:
                logger.warning(f"Failed to ingest chunk {i}: {e}")
        
        return json.dumps({
            "status": "success",
            "message": f"Ingested {ingested}/{len(chunks)} chunks from {source_name}",
            "source": source_name,
            "chunks_created": ingested
        })
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def check_source_exists(source_name: str) -> str:
    """
    Check if a source already exists in The Cortex.
    
    Args:
        source_name: The source name to check for.
    
    Returns:
        Whether the source exists and how many chunks it has.
    """
    try:
        supabase = get_supabase_client()
        
        response = supabase.table('ict_chunks') \
            .select('id') \
            .eq('source_transcript', source_name) \
            .execute()
        
        count = len(response.data) if response.data else 0
        
        return json.dumps({
            "status": "success",
            "source": source_name,
            "exists": count > 0,
            "chunk_count": count
        })
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# MCP TOOLS - STATS
# =============================================================================

@mcp.tool()
def get_cortex_stats() -> str:
    """Get statistics about The Cortex knowledge base."""
    try:
        supabase = get_supabase_client()
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
