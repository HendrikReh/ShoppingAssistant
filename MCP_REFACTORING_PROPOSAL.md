# MCP Refactoring Proposal: TavilyWebSearchAgent

## Executive Summary

Refactoring `TavilyWebSearchAgent` to follow the Model Context Protocol (MCP) would transform it from a simple tool wrapper into a proper MCP server that can be used by any MCP-compatible client (including Claude Desktop).

## Current vs Proposed Architecture

### Current Implementation
```
ShoppingAssistant App
        ↓
TavilyWebSearchAgent (internal class)
        ↓
    Tavily API
```

### Proposed MCP Architecture
```
Any MCP Client (Claude Desktop, ShoppingAssistant, etc.)
        ↓ (MCP Protocol)
Tavily MCP Server (standalone service)
        ↓
    Tavily API
```

## Implementation Design

### 1. Create MCP Server: `tavily_mcp_server.py`

```python
from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from tavily import TavilyClient
import os

# Initialize MCP server
mcp = FastMCP("Tavily Web Search")

# Configuration
class SearchConfig(BaseModel):
    api_key: str = os.getenv("TAVILY_API_KEY", "")
    search_depth: str = "basic"
    max_results: int = 10
    include_images: bool = False
    include_raw_content: bool = False

config = SearchConfig()
tavily_client = None

def get_client():
    global tavily_client
    if tavily_client is None:
        tavily_client = TavilyClient(api_key=config.api_key)
    return tavily_client

# Define Resources (data access)
@mcp.resource("search://cache/{query}")
async def get_cached_search(query: str) -> Dict[str, Any]:
    """Get cached search results for a query."""
    # Implementation would check Redis cache
    return {"query": query, "cached": False, "results": []}

# Define Tools (actions)
@mcp.tool()
async def web_search(
    query: str,
    search_type: str = "general",
    max_results: int = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Perform web search using Tavily API.
    
    Args:
        query: Search query
        search_type: Type of search (general, price, availability, review)
        max_results: Maximum number of results
        include_domains: Domains to include
        exclude_domains: Domains to exclude
        ctx: MCP context for progress reporting
    
    Returns:
        Search results with title, url, content, score
    """
    if ctx:
        await ctx.info(f"Searching for: {query}")
    
    client = get_client()
    
    # Build search parameters
    params = {
        "query": query,
        "search_depth": config.search_depth,
        "max_results": max_results,
        "include_images": config.include_images,
        "include_raw_content": config.include_raw_content
    }
    
    if include_domains:
        params["include_domains"] = include_domains
    if exclude_domains:
        params["exclude_domains"] = exclude_domains
    
    # Perform search
    try:
        response = client.search(**params)
        
        # Parse results
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published_date": item.get("published_date")
            })
        
        if ctx:
            await ctx.info(f"Found {len(results)} results")
        
        return {
            "query": query,
            "type": search_type,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Search failed: {str(e)}")
        return {"error": str(e), "results": []}

@mcp.tool()
async def search_product_prices(
    product_name: str,
    currency: str = "USD",
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Search for product prices across retailers.
    
    Args:
        product_name: Name of the product
        currency: Currency for prices
        ctx: MCP context
    
    Returns:
        Price information from various retailers
    """
    query = f"{product_name} price {currency} where to buy deals"
    
    # Use retailer domains
    retailer_domains = [
        "amazon.com", "bestbuy.com", "walmart.com",
        "target.com", "newegg.com", "bhphotovideo.com"
    ]
    
    result = await web_search(
        query=query,
        search_type="price",
        include_domains=retailer_domains,
        ctx=ctx
    )
    
    # Extract price information
    prices = []
    for item in result.get("results", []):
        # Parse prices from content (simplified)
        if "$" in item.get("content", ""):
            prices.append({
                "retailer": item.get("url", "").split("/")[2],
                "url": item.get("url", ""),
                "price_text": item.get("content", "")[:200]
            })
    
    return {
        "product": product_name,
        "currency": currency,
        "prices": prices
    }

@mcp.tool()
async def search_product_reviews(
    product_name: str,
    min_rating: float = 0.0,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Search for product reviews and ratings.
    
    Args:
        product_name: Name of the product
        min_rating: Minimum rating filter
        ctx: MCP context
    
    Returns:
        Review information from various sources
    """
    query = f"{product_name} reviews ratings user experience"
    
    # Review site domains
    review_domains = [
        "amazon.com", "youtube.com", "reddit.com",
        "trustpilot.com", "cnet.com", "techradar.com",
        "wirecutter.com", "rtings.com"
    ]
    
    result = await web_search(
        query=query,
        search_type="review",
        include_domains=review_domains,
        ctx=ctx
    )
    
    return {
        "product": product_name,
        "review_sources": len(result.get("results", [])),
        "reviews": result.get("results", [])
    }

@mcp.tool()
async def compare_products(
    product1: str,
    product2: str,
    aspects: List[str] = None,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Compare two products across various aspects.
    
    Args:
        product1: First product name
        product2: Second product name
        aspects: Aspects to compare (price, features, reviews)
        ctx: MCP context
    
    Returns:
        Comparison data for both products
    """
    if aspects is None:
        aspects = ["price", "features", "reviews"]
    
    comparison = {
        "product1": product1,
        "product2": product2,
        "aspects": {}
    }
    
    for aspect in aspects:
        if ctx:
            await ctx.report_progress(
                progress=aspects.index(aspect) / len(aspects),
                message=f"Comparing {aspect}"
            )
        
        query = f"{product1} vs {product2} comparison {aspect}"
        result = await web_search(query=query, search_type="comparison", ctx=ctx)
        comparison["aspects"][aspect] = result
    
    return comparison

# Server startup
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream=read_stream,
                write_stream=write_stream,
                init_options={}
            )
    
    asyncio.run(main())
```

### 2. MCP Client Integration in ShoppingAssistant

```python
# app/mcp_client.py
from mcp import Client
import asyncio
from typing import Dict, Any, List

class TavilyMCPClient:
    """Client for interacting with Tavily MCP Server."""
    
    def __init__(self):
        self.client = None
        
    async def connect(self):
        """Connect to the Tavily MCP server."""
        self.client = Client("tavily-search")
        await self.client.connect_to_server(
            command=["python", "-m", "tavily_mcp_server"]
        )
        
    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform web search through MCP."""
        return await self.client.call_tool(
            "web_search",
            query=query,
            **kwargs
        )
    
    async def search_prices(self, product: str) -> Dict[str, Any]:
        """Search product prices through MCP."""
        return await self.client.call_tool(
            "search_product_prices",
            product_name=product
        )
    
    async def compare(self, product1: str, product2: str) -> Dict[str, Any]:
        """Compare products through MCP."""
        return await self.client.call_tool(
            "compare_products",
            product1=product1,
            product2=product2
        )
```

### 3. Integration with Claude Desktop

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "tavily-search": {
      "command": "python",
      "args": ["-m", "tavily_mcp_server"],
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}"
      }
    }
  }
}
```

## Benefits of MCP Refactoring

### 1. **Universal Accessibility**
- Any MCP client can use the Tavily search functionality
- Claude Desktop can directly search the web
- Other AI applications can integrate seamlessly

### 2. **Standardized Interface**
- Tools and resources follow MCP specification
- Consistent error handling and progress reporting
- Type-safe with automatic validation

### 3. **Better Separation of Concerns**
- Search server runs independently
- Can be deployed as a microservice
- Easy to test and maintain

### 4. **Enhanced Features**
- Progress reporting for long searches
- Structured output with Pydantic models
- Built-in caching through resources

### 5. **Ecosystem Benefits**
- Can be published to MCP registry
- Community can contribute improvements
- Follows industry standard

## Migration Path

### Phase 1: Create MCP Server
1. Implement `tavily_mcp_server.py` with core search tools
2. Add caching resources
3. Test with MCP test client

### Phase 2: Add to ShoppingAssistant
1. Keep existing `TavilyWebSearchAgent` for backward compatibility
2. Add `TavilyMCPClient` as alternative
3. Feature flag to switch between implementations

### Phase 3: Claude Desktop Integration
1. Configure Claude Desktop to use MCP server
2. Test direct web search from Claude
3. Document usage patterns

### Phase 4: Deprecate Old Implementation
1. Migrate all usage to MCP client
2. Remove `TavilyWebSearchAgent`
3. Fully embrace MCP architecture

## Considerations

### Pros
- ✅ Industry standard protocol
- ✅ Reusable across projects
- ✅ Better tool/resource organization
- ✅ Native Claude Desktop integration
- ✅ Community ecosystem

### Cons
- ⚠️ Additional complexity for simple use cases
- ⚠️ Requires running separate server process
- ⚠️ Learning curve for MCP concepts
- ⚠️ Currently limited to certain languages

## Recommendation

**YES, refactor to MCP if:**
- You want Claude Desktop to directly search the web
- You plan to share this functionality with others
- You want to follow emerging industry standards
- You need better tool organization

**NO, keep current implementation if:**
- The current implementation works well for your needs
- You don't need external tool access
- You want to minimize complexity
- You're not using Claude Desktop

## Conclusion

Refactoring `TavilyWebSearchAgent` to MCP would modernize the architecture and make it compatible with the broader AI ecosystem. While it adds some complexity, the benefits of standardization, reusability, and Claude Desktop integration make it a worthwhile investment for the future.