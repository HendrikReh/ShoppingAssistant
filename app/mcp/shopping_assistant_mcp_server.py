#!/usr/bin/env python
"""
MCP Server for ShoppingAssistant Search Functionality

This server exposes the ShoppingAssistant's hybrid search capabilities
(BM25 + vector search + cross-encoder reranking) and web search through
the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP

# Import search functionality from app
from app.cli import (
    _load_st_model,
    _qdrant_client,
    _bm25_from_files,
    _hybrid_search_inline,
    _device_str,
    DATA_PRODUCTS,
    DATA_REVIEWS,
    EMBED_MODEL,
    CROSS_ENCODER_MODEL,
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("ShoppingAssistant Search")

# Global variables for models and data (loaded on first use)
_st_model = None
_ce_model = None
_qdrant_client_instance = None
_bm25_prod = None
_bm25_rev = None
_id_to_product = None
_id_to_review = None
_web_agent = None
_orchestrator = None
_initialized = False

@dataclass
class SearchResult:
    """Search result from ShoppingAssistant."""
    id: str
    type: str  # "product" or "review"
    title: str
    content: str
    score: float
    ce_score: float = 0.0
    metadata: Optional[Dict] = None

@dataclass 
class SearchResponse:
    """Response from search operations."""
    query: str
    results: List[SearchResult]
    total_found: int
    search_type: str  # "hybrid", "products", "reviews", "web"

def ensure_initialized():
    """Ensure models and data are loaded."""
    global _initialized, _st_model, _ce_model, _qdrant_client_instance
    global _bm25_prod, _bm25_rev, _id_to_product, _id_to_review
    
    if _initialized:
        return
    
    logger.info("Initializing ShoppingAssistant search components...")
    
    # Load device
    device = _device_str()
    logger.info(f"Using device: {device}")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    _st_model = _load_st_model(EMBED_MODEL, device=device)
    
    # Load cross-encoder model
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        _ce_model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    except Exception as e:
        logger.warning(f"Could not load cross-encoder: {e}")
        _ce_model = None
    
    # Connect to Qdrant
    logger.info("Connecting to Qdrant...")
    _qdrant_client_instance = _qdrant_client()
    
    # Load BM25 indices
    logger.info("Loading BM25 indices and data...")
    _bm25_prod, _bm25_rev, _id_to_product, _id_to_review = _bm25_from_files(
        DATA_PRODUCTS, DATA_REVIEWS
    )
    
    _initialized = True
    logger.info("Initialization complete!")

def ensure_web_search():
    """Ensure web search is initialized."""
    global _web_agent, _orchestrator
    
    if _web_agent is not None:
        return _web_agent, _orchestrator
    
    # Check for Tavily API key
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        logger.warning("TAVILY_API_KEY not set, web search disabled")
        return None, None
    
    try:
        from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig, WebSearchCache
        from app.hybrid_retrieval_orchestrator import HybridRetrievalOrchestrator
        import redis
        
        # Set up Redis cache if available
        cache = None
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            cache = WebSearchCache(redis_client)
            logger.info("Web search cache enabled (Redis)")
        except:
            logger.info("Web search cache disabled (Redis not available)")
        
        # Create web search agent
        config = WebSearchConfig(api_key=tavily_key, enable_web_search=True)
        _web_agent = TavilyWebSearchAgent(config, cache)
        
        # Create orchestrator
        ensure_initialized()
        _orchestrator = HybridRetrievalOrchestrator(
            st_model=_st_model,
            qdrant_client=_qdrant_client_instance,
            bm25_products=_bm25_prod,
            bm25_reviews=_bm25_rev,
            web_search_agent=_web_agent,
            enable_web_search=True
        )
        
        logger.info("Web search initialized successfully")
        return _web_agent, _orchestrator
        
    except Exception as e:
        logger.error(f"Failed to initialize web search: {e}")
        return None, None

def format_search_result(doc_id: str, rrf_score: float, ce_score: float, payload: dict) -> SearchResult:
    """Format a search result from internal format."""
    # Determine type
    if doc_id.startswith("prod::"):
        result_type = "product"
        title = payload.get("title", "Unknown Product")
        content = payload.get("description", "")
        metadata = {
            "asin": payload.get("parent_asin", ""),
            "price": payload.get("price", ""),
            "rating": payload.get("average_rating", 0),
            "reviews_count": payload.get("rating_number", 0),
            "categories": payload.get("categories", []),
        }
    elif doc_id.startswith("rev::"):
        result_type = "review"
        title = payload.get("title", "Review")
        content = payload.get("text", "")
        metadata = {
            "asin": payload.get("asin", ""),
            "rating": payload.get("rating", 0),
            "helpful_vote": payload.get("helpful_vote", 0),
            "verified": payload.get("verified_purchase", False),
        }
    else:
        # Web result or other
        result_type = "web"
        title = payload.get("title", "")
        content = payload.get("content", "")
        metadata = {
            "url": payload.get("url", ""),
            "source": payload.get("source", "web"),
        }
    
    return SearchResult(
        id=doc_id,
        type=result_type,
        title=title,
        content=content[:500],  # Truncate content
        score=rrf_score,
        ce_score=ce_score,
        metadata=metadata
    )

async def search_products_impl(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30
) -> SearchResponse:
    """
    Search for products using hybrid retrieval (BM25 + vector search).
    
    Args:
        query: Search query
        top_k: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        rerank_top_k: Number of candidates to rerank
        
    Returns:
        SearchResponse with product results
    """
    ensure_initialized()
    
    # Run search in thread pool since it's CPU-bound
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _hybrid_search_inline,
        query,
        _st_model,
        _qdrant_client_instance,
        _bm25_prod,
        _bm25_rev,
        _id_to_product,
        _id_to_review,
        _ce_model if use_reranking else None,
        top_k,
        60,  # rrf_k
        rerank_top_k,
        True,  # products_only
        None,  # variant
        None,  # products_file
        None,  # reviews_file
        None,  # enhanced
        None,  # product_filter
    )
    
    # Format results
    search_results = []
    for doc_id, rrf_score, ce_score, payload in results:
        result = format_search_result(doc_id, rrf_score, ce_score, payload)
        if result.type == "product":  # Filter to products only
            search_results.append(result)
    
    return SearchResponse(
        query=query,
        results=search_results,
        total_found=len(search_results),
        search_type="products"
    )

# MCP tool wrapper
@mcp.tool()
async def search_products(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30
) -> SearchResponse:
    """Search for products using hybrid retrieval."""
    return await search_products_impl(query, top_k, use_reranking, rerank_top_k)

async def search_reviews_impl(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30
) -> SearchResponse:
    """
    Search for product reviews using hybrid retrieval.
    
    Args:
        query: Search query
        top_k: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        rerank_top_k: Number of candidates to rerank
        
    Returns:
        SearchResponse with review results
    """
    ensure_initialized()
    
    # Run search in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _hybrid_search_inline,
        query,
        _st_model,
        _qdrant_client_instance,
        _bm25_prod,
        _bm25_rev,
        _id_to_product,
        _id_to_review,
        _ce_model if use_reranking else None,
        top_k * 2,  # Get more to filter
        60,  # rrf_k
        rerank_top_k,
        False,  # not products_only
        None,  # variant
        None,  # products_file
        None,  # reviews_file
        None,  # enhanced
        None,  # product_filter
    )
    
    # Format and filter to reviews only
    search_results = []
    for doc_id, rrf_score, ce_score, payload in results:
        result = format_search_result(doc_id, rrf_score, ce_score, payload)
        if result.type == "review" and len(search_results) < top_k:
            search_results.append(result)
    
    return SearchResponse(
        query=query,
        results=search_results,
        total_found=len(search_results),
        search_type="reviews"
    )

# MCP tool wrapper
@mcp.tool()
async def search_reviews(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30
) -> SearchResponse:
    """Search for product reviews using hybrid retrieval."""
    return await search_reviews_impl(query, top_k, use_reranking, rerank_top_k)

async def hybrid_search_impl(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30,
    rrf_k: int = 60
) -> SearchResponse:
    """
    Search for both products and reviews using hybrid retrieval.
    
    Args:
        query: Search query
        top_k: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        rerank_top_k: Number of candidates to rerank
        rrf_k: RRF fusion parameter
        
    Returns:
        SearchResponse with mixed product and review results
    """
    ensure_initialized()
    
    # Run search in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _hybrid_search_inline,
        query,
        _st_model,
        _qdrant_client_instance,
        _bm25_prod,
        _bm25_rev,
        _id_to_product,
        _id_to_review,
        _ce_model if use_reranking else None,
        top_k,
        rrf_k,
        rerank_top_k,
        False,  # not products_only
        None,  # variant
        None,  # products_file
        None,  # reviews_file
        None,  # enhanced
        None,  # product_filter
    )
    
    # Format results
    search_results = []
    for doc_id, rrf_score, ce_score, payload in results:
        search_results.append(format_search_result(doc_id, rrf_score, ce_score, payload))
    
    return SearchResponse(
        query=query,
        results=search_results,
        total_found=len(search_results),
        search_type="hybrid"
    )

# MCP tool wrapper
@mcp.tool()
async def hybrid_search(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    rerank_top_k: int = 30,
    rrf_k: int = 60
) -> SearchResponse:
    """Search for both products and reviews using hybrid retrieval."""
    return await hybrid_search_impl(query, top_k, use_reranking, rerank_top_k, rrf_k)

async def web_search_impl(
    query: str,
    top_k: int = 10,
    search_type: str = "general",
    use_cache: bool = True
) -> SearchResponse:
    """
    Search the web for product information using Tavily.
    
    Args:
        query: Search query
        top_k: Number of results to return
        search_type: Type of search (general, price, review, comparison)
        use_cache: Whether to use cached results
        
    Returns:
        SearchResponse with web search results
    """
    web_agent, orchestrator = ensure_web_search()
    
    if web_agent is None:
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            search_type="web"
        )
    
    try:
        # Use web agent directly
        from app.web_search_agent import WebSearchResult as WSResult
        
        results = web_agent.search(
            query=query,
            search_type=search_type,
            use_cache=use_cache
        )
        
        # Convert to our format
        search_results = []
        for idx, r in enumerate(results[:top_k]):
            search_results.append(SearchResult(
                id=f"web::{idx}",
                type="web",
                title=r.title,
                content=r.content or "",
                score=r.score if hasattr(r, 'score') else 0.8,
                ce_score=0.0,
                metadata={
                    "url": r.url,
                    "published_date": r.published_date if hasattr(r, 'published_date') else None
                }
            ))
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_found=len(search_results),
            search_type="web"
        )
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return SearchResponse(
            query=query,
            results=[],
            total_found=0,
            search_type="web"
        )

# MCP tool wrapper
@mcp.tool()
async def web_search(
    query: str,
    top_k: int = 10,
    search_type: str = "general",
    use_cache: bool = True
) -> SearchResponse:
    """Search the web for product information using Tavily."""
    return await web_search_impl(query, top_k, search_type, use_cache)

async def hybrid_search_with_web_impl(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    force_web: bool = False
) -> SearchResponse:
    """
    Search combining local data and web results.
    
    Args:
        query: Search query
        top_k: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        force_web: Force web-only search
        
    Returns:
        SearchResponse with combined local and web results
    """
    web_agent, orchestrator = ensure_web_search()
    
    if orchestrator is None:
        # Fall back to local-only search
        return await hybrid_search(query, top_k, use_reranking)
    
    try:
        from app.hybrid_retrieval_orchestrator import HybridResult
        
        # Use orchestrator for combined retrieval
        results = orchestrator.retrieve(
            query=query,
            top_k=top_k,
            force_web=force_web
        )
        
        # Convert to our format
        search_results = []
        for r in results:
            if r.is_web:
                result = SearchResult(
                    id=f"web::{r.id}",
                    type="web",
                    title=r.title,
                    content=r.content,
                    score=r.relevance_score,
                    ce_score=0.0,
                    metadata={"url": r.metadata.get("url", ""), "source": "web"}
                )
            else:
                # Local result
                if "product" in r.source.lower():
                    result_type = "product"
                elif "review" in r.source.lower():
                    result_type = "review"
                else:
                    result_type = "local"
                
                result = SearchResult(
                    id=r.id,
                    type=result_type,
                    title=r.title,
                    content=r.content,
                    score=r.relevance_score,
                    ce_score=0.0,
                    metadata=r.metadata
                )
            
            search_results.append(result)
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_found=len(search_results),
            search_type="hybrid_web"
        )
        
    except Exception as e:
        logger.error(f"Hybrid web search failed: {e}")
        # Fall back to local-only
        return await hybrid_search(query, top_k, use_reranking)

# MCP tool wrapper
@mcp.tool()
async def hybrid_search_with_web(
    query: str,
    top_k: int = 20,
    use_reranking: bool = True,
    force_web: bool = False
) -> SearchResponse:
    """Search combining local data and web results."""
    return await hybrid_search_with_web_impl(query, top_k, use_reranking, force_web)

# Resources for browsing cached data
@mcp.resource("search://products/list")
async def list_products_impl() -> str:
    """List available product categories."""
    ensure_initialized()
    
    # Get unique categories
    categories = set()
    for pid, product in _id_to_product.items():
        if 'categories' in product:
            for cat in product.get('categories', []):
                if cat:
                    categories.add(cat)
    
    return json.dumps({
        "total_products": len(_id_to_product),
        "total_reviews": len(_id_to_review),
        "categories": sorted(list(categories)[:50])  # Top 50 categories
    }, indent=2)

# MCP resource wrapper
@mcp.resource("search://products/list")
async def list_products() -> str:
    """List available product categories."""
    return await list_products_impl()

async def get_stats_impl() -> str:
    """Get statistics about the search system."""
    ensure_initialized()
    
    stats = {
        "products_indexed": len(_id_to_product),
        "reviews_indexed": len(_id_to_review),
        "embedding_model": EMBED_MODEL,
        "cross_encoder_model": CROSS_ENCODER_MODEL if _ce_model else "Not loaded",
        "vector_collections": [COLLECTION_PRODUCTS, COLLECTION_REVIEWS],
        "web_search_enabled": _web_agent is not None,
        "device": _device_str()
    }
    
    return json.dumps(stats, indent=2)

# MCP resource wrapper
@mcp.resource("search://stats")
async def get_stats() -> str:
    """Get statistics about the search system."""
    return await get_stats_impl()

# Custom JSON encoder for dataclasses
class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)

# Override MCP's JSON encoding
mcp.json_encoder = DataclassEncoder

def main():
    """Run the MCP server."""
    import uvloop
    
    # Use uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    logger.info("Starting ShoppingAssistant MCP Server...")
    logger.info(f"Products data: {DATA_PRODUCTS}")
    logger.info(f"Reviews data: {DATA_REVIEWS}")
    
    # Check for Tavily API key
    if os.getenv("TAVILY_API_KEY"):
        logger.info("Tavily API key found - web search will be available")
    else:
        logger.info("No Tavily API key - web search disabled")
    
    # Run the server
    mcp.run()

if __name__ == "__main__":
    main()