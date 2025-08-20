# Tavily MCP Server

This directory contains the Model Context Protocol (MCP) implementation for Tavily web search, allowing any MCP-compatible client (including Claude Desktop) to perform web searches.

## Components

### 1. `tavily_mcp_server.py`
The MCP server that exposes Tavily search functionality as tools and resources.

**Tools provided:**
- `web_search` - General web search with domain filtering
- `search_product_prices` - Find product prices across retailers
- `search_product_reviews` - Search for product reviews
- `compare_products` - Compare two products
- `check_availability` - Check product stock status
- `find_alternatives` - Find alternative products

**Resources provided:**
- `search://cache/list` - List cached searches
- `search://cache/{type}/{query}` - Get specific cached result

### 2. `tavily_mcp_client.py`
Python client for connecting to the MCP server from within the ShoppingAssistant app.

**Classes:**
- `TavilyMCPClient` - Async/sync client for MCP server
- `TavilyMCPAdapter` - Backward compatible adapter (drop-in replacement for TavilyWebSearchAgent)

## Setup

### 1. Environment Variables
```bash
export TAVILY_API_KEY="your-tavily-api-key"
export REDIS_URL="redis://localhost:6379/0"  # Optional, for caching
```

### 2. Install Dependencies
```bash
uv add mcp tavily-python redis
```

### 3. Start Redis (Optional, for caching)
```bash
docker run -d -p 6379:6379 redis:alpine
```

## Usage

### Running the MCP Server Standalone
```bash
# From project root
uv run python -m app.mcp.tavily_mcp_server
```

### Using in ShoppingAssistant

#### Option 1: Drop-in Replacement
```python
from app.mcp.tavily_mcp_client import get_mcp_agent

# Instead of:
# agent = TavilyWebSearchAgent(config, cache)

# Use:
agent = get_mcp_agent(config, cache)

# All existing code works unchanged
results = agent.search("product query")
```

#### Option 2: Direct Client Usage
```python
from app.mcp.tavily_mcp_client import TavilyMCPClient
import asyncio

async def search_example():
    client = TavilyMCPClient()
    await client.connect()
    
    results = await client.search_async(
        query="best laptops 2024",
        search_type="general",
        max_results=10
    )
    
    for result in results:
        print(f"{result.title}: {result.url}")
    
    await client.disconnect()

asyncio.run(search_example())
```

### Using with Claude Desktop

1. **Copy configuration to Claude Desktop config directory:**

macOS:
```bash
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Windows:
```bash
copy claude_desktop_config.json %APPDATA%\Claude\claude_desktop_config.json
```

2. **Set environment variable in your shell profile:**
```bash
echo 'export TAVILY_API_KEY="your-key"' >> ~/.zshrc
```

3. **Restart Claude Desktop**

4. **Use in Claude:**
```
Claude can now search the web directly! Try:
- "Search for the latest iPhone reviews"
- "Find prices for Samsung Galaxy S24"
- "Compare MacBook Pro vs Dell XPS"
```

## Testing

Run the test script to verify everything works:

```bash
# From project root
uv run python test_mcp_server.py
```

Expected output:
```
Testing Tavily MCP Server
==================================================
1. Connecting to MCP server...
✅ Connected successfully

2. Testing web search...
✅ Found 5 results

3. Testing product price search...
✅ Found 3 price results

4. Testing find alternatives...
✅ Found 3 alternatives

5. Disconnecting...
✅ Disconnected successfully
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Claude Desktop │     │ ShoppingAssist  │     │ Other MCP    │
│     (Client)    │     │    (Client)     │     │   Clients    │
└────────┬────────┘     └────────┬────────┘     └──────┬───────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
                          MCP Protocol
                                 │
                    ┌────────────▼────────────┐
                    │   Tavily MCP Server     │
                    │  (tavily_mcp_server.py) │
                    └────────────┬────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Tavily API    │
                         └────────────────┘
```

## Benefits

1. **Universal Access** - Any MCP client can use Tavily search
2. **Claude Desktop Integration** - Claude can search the web directly
3. **Standardized Interface** - Follows MCP specification
4. **Caching** - Redis-based caching for repeated queries
5. **Backward Compatible** - Works with existing ShoppingAssistant code
6. **Parallel Processing** - Support for concurrent searches

## Troubleshooting

### Server won't start
- Check TAVILY_API_KEY is set: `echo $TAVILY_API_KEY`
- Verify Python environment: `uv run python --version`
- Check MCP is installed: `uv pip show mcp`

### No search results
- Verify API key is valid
- Check internet connection
- Look at server logs for errors

### Claude Desktop can't connect
- Ensure config file is in correct location
- Check server path in config is absolute
- Restart Claude Desktop after config changes
- Check Claude Desktop logs: `~/Library/Logs/Claude/`

### Cache not working
- Verify Redis is running: `redis-cli ping`
- Check Redis connection URL
- Look for cache-related errors in logs

## Contributing

To add new search tools:

1. Add tool function to `tavily_mcp_server.py`:
```python
@mcp.tool()
async def my_new_search(query: str) -> Dict:
    """Description of your search tool."""
    # Implementation
    pass
```

2. Update client wrapper if needed
3. Add tests to `test_mcp_server.py`
4. Update this README

## License

Same as ShoppingAssistant project.