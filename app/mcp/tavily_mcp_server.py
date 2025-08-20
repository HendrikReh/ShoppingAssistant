#!/usr/bin/env python
"""
Tavily Web Search MCP Server

This server exposes Tavily web search functionality through the Model Context Protocol,
allowing any MCP-compatible client (including Claude Desktop) to perform web searches.
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from tavily import TavilyClient
import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tavily-mcp")

# Initialize MCP server
mcp = FastMCP("Tavily Web Search")

# ---------------------
# Configuration
# ---------------------

class SearchConfig(BaseModel):
    """Configuration for Tavily search."""
    api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    search_depth: str = Field(default="basic", description="Search depth (basic or advanced)")
    max_results: int = Field(default=10, description="Maximum number of results")
    include_images: bool = Field(default=False, description="Include images in results")
    include_raw_content: bool = Field(default=False, description="Include raw HTML content")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL for caching")

config = SearchConfig()

# Global clients
tavily_client: Optional[TavilyClient] = None
redis_client: Optional[redis.Redis] = None

# ---------------------
# Client Management
# ---------------------

def get_tavily_client() -> TavilyClient:
    """Get or create Tavily client."""
    global tavily_client
    if tavily_client is None:
        if not config.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        tavily_client = TavilyClient(api_key=config.api_key)
    return tavily_client

def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client for caching."""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(config.redis_url, decode_responses=True)
            redis_client.ping()
            logger.info("Connected to Redis for caching")
        except Exception as e:
            logger.warning(f"Redis not available for caching: {e}")
            redis_client = None
    return redis_client

# ---------------------
# Cache Helpers
# ---------------------

def get_cache_key(query: str, search_type: str) -> str:
    """Generate cache key for a search."""
    return f"mcp:tavily:{search_type}:{query}"

def get_from_cache(query: str, search_type: str) -> Optional[Dict[str, Any]]:
    """Get cached search results."""
    client = get_redis_client()
    if client:
        try:
            key = get_cache_key(query, search_type)
            data = client.get(key)
            if data:
                logger.info(f"Cache hit for query: {query[:50]}")
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
    return None

def save_to_cache(query: str, search_type: str, data: Dict[str, Any], ttl: Optional[int] = None):
    """Save search results to cache."""
    client = get_redis_client()
    if client:
        try:
            key = get_cache_key(query, search_type)
            ttl = ttl or config.cache_ttl
            client.setex(key, ttl, json.dumps(data))
            logger.info(f"Cached results for query: {query[:50]}")
        except Exception as e:
            logger.error(f"Cache write error: {e}")

# ---------------------
# Result Models
# ---------------------

class SearchResult(BaseModel):
    """Individual search result."""
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None

class SearchResponse(BaseModel):
    """Search response with results."""
    query: str
    search_type: str
    count: int
    cached: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    results: List[SearchResult]

class PriceInfo(BaseModel):
    """Price information for a product."""
    retailer: str
    url: str
    price_text: str
    extracted_price: Optional[float] = None

class PriceSearchResponse(BaseModel):
    """Price search response."""
    product: str
    currency: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    prices: List[PriceInfo]

# ---------------------
# MCP Resources
# ---------------------

@mcp.resource("search://cache/list")
async def list_cached_searches() -> Dict[str, Any]:
    """List all cached search queries."""
    client = get_redis_client()
    if not client:
        return {"error": "Cache not available", "keys": []}
    
    try:
        pattern = "mcp:tavily:*"
        keys = client.keys(pattern)
        
        # Parse keys to extract queries and types
        searches = []
        for key in keys[:100]:  # Limit to 100 most recent
            parts = key.split(":", 3)
            if len(parts) >= 4:
                searches.append({
                    "type": parts[2],
                    "query": parts[3],
                    "ttl": client.ttl(key)
                })
        
        return {
            "count": len(searches),
            "searches": searches
        }
    except Exception as e:
        return {"error": str(e), "keys": []}

@mcp.resource("search://cache/{search_type}/{query}")
async def get_cached_search(search_type: str, query: str) -> Dict[str, Any]:
    """Get cached search results for a specific query."""
    cached = get_from_cache(query, search_type)
    if cached:
        return cached
    return {"error": "Not found in cache", "query": query, "type": search_type}

# ---------------------
# MCP Tools
# ---------------------

@mcp.tool()
async def web_search(
    query: str,
    search_type: str = "general",
    max_results: int = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    use_cache: bool = True
) -> SearchResponse:
    """
    Perform web search using Tavily API.
    
    Args:
        query: Search query string
        search_type: Type of search (general, price, availability, review, comparison)
        max_results: Maximum number of results to return
        include_domains: List of domains to include in search
        exclude_domains: List of domains to exclude from search
        use_cache: Whether to use cached results if available
    
    Returns:
        SearchResponse with list of search results
    """
    # Check cache first
    if use_cache:
        cached = get_from_cache(query, search_type)
        if cached:
            response = SearchResponse(**cached)
            response.cached = True
            return response
    
    try:
        client = get_tavily_client()
        
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
        logger.info(f"Searching Tavily for: {query[:100]}")
        response = client.search(**params)
        
        # Parse results
        results = []
        for item in response.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date")
            ))
        
        search_response = SearchResponse(
            query=query,
            search_type=search_type,
            count=len(results),
            results=results
        )
        
        # Cache the results
        save_to_cache(query, search_type, search_response.model_dump())
        
        return search_response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return SearchResponse(
            query=query,
            search_type=search_type,
            count=0,
            results=[]
        )

@mcp.tool()
async def search_product_prices(
    product_name: str,
    currency: str = "USD"
) -> PriceSearchResponse:
    """
    Search for product prices across major retailers.
    
    Args:
        product_name: Name of the product to search
        currency: Currency for price search (default: USD)
    
    Returns:
        PriceSearchResponse with prices from various retailers
    """
    # Build price-focused query
    query = f"{product_name} price {currency} where to buy deals discount"
    
    # Focus on major retailer domains
    retailer_domains = [
        "amazon.com", "bestbuy.com", "walmart.com",
        "target.com", "newegg.com", "bhphotovideo.com",
        "costco.com", "ebay.com", "microcenter.com"
    ]
    
    # Search with retailer focus
    search_result = await web_search(
        query=query,
        search_type="price",
        max_results=15,
        include_domains=retailer_domains,
        use_cache=True
    )
    
    # Extract price information
    prices = []
    for result in search_result.results:
        # Extract retailer from URL
        retailer = result.url.split("/")[2].replace("www.", "")
        
        # Look for price patterns in content
        import re
        price_pattern = r'\$[\d,]+\.?\d*'
        price_matches = re.findall(price_pattern, result.content)
        
        if price_matches:
            # Take the first price found
            price_text = price_matches[0]
            try:
                # Extract numeric value
                extracted = float(price_text.replace("$", "").replace(",", ""))
            except:
                extracted = None
            
            prices.append(PriceInfo(
                retailer=retailer,
                url=result.url,
                price_text=price_text,
                extracted_price=extracted
            ))
    
    return PriceSearchResponse(
        product=product_name,
        currency=currency,
        prices=prices
    )

@mcp.tool()
async def search_product_reviews(
    product_name: str,
    focus: str = "all"
) -> SearchResponse:
    """
    Search for product reviews and ratings.
    
    Args:
        product_name: Name of the product
        focus: Review focus (all, professional, user, video)
    
    Returns:
        SearchResponse with review results
    """
    # Build review-focused query
    if focus == "professional":
        query = f"{product_name} professional review test benchmark"
        domains = ["cnet.com", "techradar.com", "wirecutter.com", 
                  "rtings.com", "tomsguide.com", "pcmag.com"]
    elif focus == "user":
        query = f"{product_name} user reviews customer experience"
        domains = ["amazon.com", "reddit.com", "trustpilot.com", 
                  "bestbuy.com", "walmart.com"]
    elif focus == "video":
        query = f"{product_name} review video unboxing"
        domains = ["youtube.com", "vimeo.com"]
    else:
        query = f"{product_name} reviews ratings pros cons"
        domains = None
    
    return await web_search(
        query=query,
        search_type="review",
        max_results=10,
        include_domains=domains,
        use_cache=True
    )

@mcp.tool()
async def compare_products(
    product1: str,
    product2: str,
    aspects: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare two products across various aspects.
    
    Args:
        product1: First product name
        product2: Second product name
        aspects: Specific aspects to compare (default: price, features, reviews)
    
    Returns:
        Detailed comparison between the two products
    """
    if aspects is None:
        aspects = ["price", "features", "reviews", "specs"]
    
    comparison = {
        "product1": product1,
        "product2": product2,
        "timestamp": datetime.utcnow().isoformat(),
        "comparisons": {}
    }
    
    for aspect in aspects:
        # Build comparison query
        query = f"{product1} vs {product2} comparison {aspect} difference"
        
        # Search for comparison
        result = await web_search(
            query=query,
            search_type="comparison",
            max_results=5,
            use_cache=True
        )
        
        comparison["comparisons"][aspect] = {
            "query": query,
            "sources": len(result.results),
            "results": [r.model_dump() for r in result.results]
        }
    
    return comparison

@mcp.tool()
async def check_availability(
    product_name: str,
    location: str = "US"
) -> Dict[str, Any]:
    """
    Check product availability and stock status.
    
    Args:
        product_name: Name of the product
        location: Location/country for availability check
    
    Returns:
        Availability information from various retailers
    """
    query = f"{product_name} in stock availability {location} shipping"
    
    result = await web_search(
        query=query,
        search_type="availability",
        max_results=10,
        use_cache=False  # Don't cache availability
    )
    
    # Parse availability from results
    availability = {
        "product": product_name,
        "location": location,
        "timestamp": datetime.utcnow().isoformat(),
        "retailers": []
    }
    
    for r in result.results:
        # Look for availability keywords
        content_lower = r.content.lower()
        is_available = any(term in content_lower for term in 
                          ["in stock", "available", "ships", "ready"])
        is_unavailable = any(term in content_lower for term in 
                            ["out of stock", "unavailable", "sold out"])
        
        retailer = r.url.split("/")[2].replace("www.", "")
        
        availability["retailers"].append({
            "retailer": retailer,
            "url": r.url,
            "status": "available" if is_available and not is_unavailable else 
                     "unavailable" if is_unavailable else "unknown",
            "snippet": r.content[:200]
        })
    
    return availability

@mcp.tool()
async def find_alternatives(
    product_name: str,
    criteria: Optional[str] = None
) -> SearchResponse:
    """
    Find alternative products similar to the specified one.
    
    Args:
        product_name: Name of the product to find alternatives for
        criteria: Specific criteria for alternatives (e.g., "cheaper", "better rated")
    
    Returns:
        SearchResponse with alternative product suggestions
    """
    if criteria:
        query = f"{product_name} alternatives similar {criteria}"
    else:
        query = f"{product_name} alternatives competitors similar products comparison"
    
    return await web_search(
        query=query,
        search_type="alternatives",
        max_results=15,
        use_cache=True
    )

# ---------------------
# Server Entry Point
# ---------------------

async def main():
    """Main entry point for the MCP server."""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Tavily MCP Server...")
    logger.info(f"API Key configured: {'Yes' if config.api_key else 'No'}")
    logger.info(f"Redis cache: {'Checking...' if config.redis_url else 'Disabled'}")
    
    # Test Redis connection
    get_redis_client()
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server ready, waiting for connections...")
        await mcp.run(
            read_stream=read_stream,
            write_stream=write_stream,
            init_options={}
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)