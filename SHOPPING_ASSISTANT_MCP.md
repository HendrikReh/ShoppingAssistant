# ShoppingAssistant MCP Server

## Overview
The ShoppingAssistant MCP Server exposes the app's hybrid search functionality (BM25 + vector search + cross-encoder reranking) through the Model Context Protocol, making it accessible to any MCP-compatible client including Claude Desktop.

## Features

### Search Tools
1. **search_products** - Search for products using hybrid retrieval
2. **search_reviews** - Search for product reviews  
3. **hybrid_search** - Search both products and reviews
4. **web_search** - Search the web via Tavily (requires API key)
5. **hybrid_search_with_web** - Combine local and web results

### Resources
- **search://products/list** - List product categories and counts
- **search://stats** - Get system statistics

## Architecture

### Search Pipeline
```
Query → BM25 Search → Vector Search → RRF Fusion → Cross-Encoder Reranking → Results
         ↓              ↓                ↓                    ↓
     Token-based    Semantic      Reciprocal Rank      Neural Reranking
     Keyword Match  Similarity       Fusion             (ms-marco)
```

### Components
- **BM25**: Traditional keyword-based search on products and reviews
- **Vector Search**: Semantic search using all-MiniLM-L6-v2 embeddings (384-dim)
- **Qdrant**: Vector database for storing and searching embeddings
- **RRF Fusion**: Combines BM25 and vector results using Reciprocal Rank Fusion
- **Cross-Encoder**: ms-marco-MiniLM-L-12-v2 for neural reranking of top candidates
- **Web Search**: Optional Tavily integration for real-time web results

## Installation

### 1. Install Dependencies
```bash
uv add fastmcp uvloop
```

### 2. Ensure Data is Ingested
```bash
# Ingest products and reviews into Qdrant
uv run python -m app.cli ingest
```

### 3. Start Infrastructure
```bash
# Start Qdrant and Redis
docker-compose up -d
```

## Usage

### Direct Python Usage
```python
from app.mcp import get_shopping_assistant

# Create assistant
assistant = get_shopping_assistant()

# Search products
results = assistant.search(
    query="wireless earbuds",
    search_type="products",
    top_k=10
)

for result in results:
    print(f"{result.title} (score: {result.score:.3f})")
```

### Claude Desktop Integration
1. Copy configuration:
```bash
# macOS
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/

# Windows
copy claude_desktop_config.json %APPDATA%\Claude\
```

2. Set environment variables (optional):
```bash
export TAVILY_API_KEY="your-key"  # For web search
```

3. Restart Claude Desktop

4. Use in Claude:
- "Search for wireless earbuds"
- "Find reviews about battery life"
- "Show me Fire TV Stick products"

### MCP Server Direct
```bash
# Run server directly
uv run python app/mcp/shopping_assistant_mcp_server.py
```

## Testing

### Direct Function Test
```bash
# Test search functions directly
uv run python test_shopping_direct.py
```

### Full MCP Test (requires fixes for stdio connection)
```bash
# Test via MCP protocol
uv run python test_shopping_mcp.py
```

## Configuration

### Environment Variables
- `TAVILY_API_KEY` - Enable web search functionality
- `REDIS_URL` - Redis connection for caching (default: redis://localhost:6379/0)

### Search Parameters
- `top_k` - Number of results to return (default: 20)
- `use_reranking` - Enable cross-encoder reranking (default: true)
- `rerank_top_k` - Number of candidates to rerank (default: 30)
- `rrf_k` - RRF fusion parameter (default: 60)

## Performance

### Typical Search Times
- Product search: ~0.5-1s
- Review search: ~0.5-1s
- Hybrid search: ~1-2s
- Web search: ~2-3s (depends on Tavily)

### Resource Usage
- Memory: ~2GB (models + indices)
- CPU/GPU: Uses MPS on Mac, CUDA on Linux/Windows
- Storage: ~500MB for models, varies for data

## API Reference

### SearchResult
```python
@dataclass
class SearchResult:
    id: str                    # Unique identifier
    type: str                  # "product", "review", or "web"
    title: str                 # Result title
    content: str               # Result content (truncated to 500 chars)
    score: float               # RRF fusion score
    ce_score: float            # Cross-encoder score (if reranked)
    metadata: Optional[Dict]   # Additional metadata
```

### SearchResponse
```python
@dataclass
class SearchResponse:
    query: str                  # Original query
    results: List[SearchResult] # Search results
    total_found: int           # Total results found
    search_type: str           # Type of search performed
```

## Troubleshooting

### "No module named 'fastmcp'"
```bash
uv add fastmcp
```

### "Products data not found"
```bash
uv run python -m app.cli ingest
```

### "Cannot connect to Qdrant"
```bash
docker-compose up -d
# Check if Qdrant is running on http://localhost:6333
```

### Slow first search
This is normal - models are loaded on first use. Subsequent searches will be faster.

### Web search not working
Set TAVILY_API_KEY environment variable:
```bash
export TAVILY_API_KEY="your-api-key"
```

## Benefits

1. **Universal Access** - Any MCP client can use ShoppingAssistant search
2. **Claude Desktop Integration** - Native integration with Claude
3. **High-Quality Search** - State-of-the-art hybrid retrieval pipeline
4. **Flexible** - Products only, reviews only, or combined search
5. **Web Enhancement** - Optional real-time web results
6. **Fast** - Optimized with caching and efficient models

## Implementation Status

✅ **Complete**
- MCP server with 5 search tools and 2 resources
- Client wrapper with async/sync support
- Backward compatible adapter
- Direct function testing
- Claude Desktop configuration

⚠️ **Known Issues**
- MCP stdio connection needs refinement for full client testing
- Resource duplicate warning (cosmetic issue)

## Next Steps

1. Fix stdio connection for full MCP client testing
2. Add more search filters (category, price range, rating)
3. Implement search history and favorites
4. Add bulk search capabilities
5. Create web UI for MCP server