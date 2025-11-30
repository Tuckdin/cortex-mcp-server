“””
The Cortex MCP Server
ATLAS’s bridge to the knowledge base

Searches the Supabase vector database containing:

- 776 ICT trading transcripts (19,260 chunks)
- 3 Vanessa Van Edwards books (897 chunks)
- Total: 20,829 searchable chunks
  “””

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
from typing import Optional
import uvicorn

# Initialize FastAPI

app = FastAPI(
title=“The Cortex MCP Server”,
description=“ATLAS’s semantic search interface to the knowledge base”,
version=“1.0.0”
)

# Add CORS middleware

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# Initialize clients (will use environment variables)

openai_client: Optional[OpenAI] = None
supabase_client = None

def get_clients():
“”“Initialize clients on first request”””
global openai_client, supabase_client

```
if openai_client is None:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

if supabase_client is None:
    supabase_client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

return openai_client, supabase_client
```

# Request/Response models

class SearchRequest(BaseModel):
query: str
match_count: int = 5
match_threshold: float = 0.5
source_filter: Optional[str] = None  # “ict”, “vanessa”, or None for all

class SearchResult(BaseModel):
content: str
source: str
similarity: float

class SearchResponse(BaseModel):
query: str
results: list[SearchResult]
total_found: int

class HealthResponse(BaseModel):
status: str
database: str
total_chunks: int

# Embedding function

def get_embedding(text: str) -> list[float]:
“”“Generate embedding using OpenAI”””
openai, _ = get_clients()
response = openai.embeddings.create(
model=“text-embedding-3-small”,
input=text
)
return response.data[0].embedding

# Search function

def search_cortex(
query: str,
match_count: int = 5,
match_threshold: float = 0.5,
source_filter: Optional[str] = None
) -> list[dict]:
“””
Search The Cortex for relevant chunks

```
Args:
    query: The search query
    match_count: Number of results to return
    match_threshold: Minimum similarity score (0-1)
    source_filter: Optional filter - "ict" or "vanessa"

Returns:
    List of matching chunks with content, source, and similarity
"""
_, supabase = get_clients()

# Generate embedding for query
query_embedding = get_embedding(query)

# Call the search function in Supabase
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
    if source_filter.lower() == "ict":
        results = [r for r in results if "vanessa" not in r.get("source_transcript", "").lower() 
                  and "cues" not in r.get("source_transcript", "").lower()
                  and "captivate" not in r.get("source_transcript", "").lower()]
    elif source_filter.lower() == "vanessa":
        results = [r for r in results if "vanessa" in r.get("source_transcript", "").lower()
                  or "cues" in r.get("source_transcript", "").lower()
                  or "captivate" in r.get("source_transcript", "").lower()
                  or "lie detection" in r.get("source_transcript", "").lower()]

return results
```

# API Endpoints

@app.get(”/”, response_model=HealthResponse)
async def health_check():
“”“Health check endpoint”””
try:
_, supabase = get_clients()
# Get chunk count
response = supabase.table(‘ict_chunks’).select(‘id’, count=‘exact’).execute()
total = response.count or 0
return HealthResponse(
status=“healthy”,
database=“connected”,
total_chunks=total
)
except Exception as e:
return HealthResponse(
status=“error”,
database=str(e),
total_chunks=0
)

@app.post(”/search”, response_model=SearchResponse)
async def search(request: SearchRequest):
“””
Search The Cortex

```
- **query**: What to search for
- **match_count**: Number of results (default 5)
- **match_threshold**: Minimum similarity 0-1 (default 0.5)
- **source_filter**: "ict" for trading, "vanessa" for social skills, or omit for all
"""
try:
    results = search_cortex(
        query=request.query,
        match_count=request.match_count,
        match_threshold=request.match_threshold,
        source_filter=request.source_filter
    )
    
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                content=r.get("content", ""),
                source=r.get("source_transcript", "Unknown"),
                similarity=round(r.get("similarity", 0), 3)
            )
            for r in results
        ],
        total_found=len(results)
    )
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/search/ict”)
async def search_ict(query: str, limit: int = 5):
“”“Quick endpoint for ICT-only searches”””
results = search_cortex(query, match_count=limit, source_filter=“ict”)
return {“query”: query, “results”: results}

@app.get(”/search/vanessa”)
async def search_vanessa(query: str, limit: int = 5):
“”“Quick endpoint for Vanessa Van Edwards searches”””
results = search_cortex(query, match_count=limit, source_filter=“vanessa”)
return {“query”: query, “results”: results}

# MCP Tool Definition (for Claude integration)

MCP_TOOLS = {
“search_cortex”: {
“description”: “Search The Cortex knowledge base containing ICT trading methodology and Vanessa Van Edwards social intelligence frameworks. Use this when Mr. Pak asks about trading concepts, market psychology, social skills, body language, or communication strategies.”,
“parameters”: {
“type”: “object”,
“properties”: {
“query”: {
“type”: “string”,
“description”: “The search query - what concept or topic to find”
},
“source”: {
“type”: “string”,
“enum”: [“all”, “ict”, “vanessa”],
“description”: “Filter by source: ‘ict’ for trading, ‘vanessa’ for social skills, ‘all’ for both”
}
},
“required”: [“query”]
}
}
}

@app.get(”/mcp/tools”)
async def get_mcp_tools():
“”“Return MCP tool definitions for Claude integration”””
return MCP_TOOLS

if **name** == “**main**”:
port = int(os.environ.get(“PORT”, 8000))
uvicorn.run(app, host=“0.0.0.0”, port=port)
