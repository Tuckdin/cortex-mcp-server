# The Cortex MCP Server

ATLAS's semantic search interface to the knowledge base.

## What's Inside The Cortex

- **776 ICT Trading Transcripts** → 19,260 searchable chunks
- **3 Vanessa Van Edwards Books** → 897 searchable chunks
- **Total: 20,829 chunks** of searchable wisdom

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/search` | POST | Full search with filters |
| `/search/ict` | GET | Quick ICT-only search |
| `/search/vanessa` | GET | Quick Vanessa-only search |
| `/mcp/tools` | GET | MCP tool definitions |

## Search Example

```bash
curl -X POST "https://your-app.railway.app/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "order blocks", "match_count": 5}'
```

## Environment Variables

Set these in Railway:

- `OPENAI_API_KEY` - Your OpenAI API key
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Your Supabase anon/service key

## Built for ATLAS

*Advanced Trading & Life Assistance System*
