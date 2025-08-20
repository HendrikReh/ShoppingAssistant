"""
MCP Client for ShoppingAssistant Search

This client provides access to the ShoppingAssistant MCP server,
allowing external applications to perform product and review searches.
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

@dataclass
class ShoppingSearchResult:
    """Search result from ShoppingAssistant."""
    id: str
    type: str  # "product", "review", or "web"
    title: str
    content: str
    score: float
    ce_score: float = 0.0
    metadata: Optional[Dict] = None

class ShoppingAssistantMCPClient:
    """
    MCP client for ShoppingAssistant search server.
    
    This client provides access to hybrid search capabilities including:
    - Product search (BM25 + vector + reranking)
    - Review search
    - Web search (if Tavily API key is configured)
    - Combined hybrid search
    """
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._context = None
        self._read_stream = None
        self._write_stream = None
        
    async def connect(self):
        """Connect to the ShoppingAssistant MCP server."""
        if self._connected:
            return
            
        try:
            # Start server process
            server_script = Path(__file__).parent / "shopping_assistant_mcp_server.py"
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "python", str(server_script)],
                env={
                    **os.environ,  # Include all env vars
                    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0")
                }
            )
            
            # Create context and connect
            self._context = stdio_client(server_params)
            self._read_stream, self._write_stream = await self._context.__aenter__()
            
            # Create client session
            self.session = ClientSession(self._read_stream, self._write_stream)
            
            # Initialize session
            await self.session.initialize()
            
            self._connected = True
            logger.info("Connected to ShoppingAssistant MCP server")
            
            # List available tools
            tools = await self.session.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            try:
                await self.session.close()
            except:
                pass
            self.session = None
        
        if self._context:
            try:
                await self._context.__aexit__(None, None, None)
            except:
                pass
            self._context = None
            
        self._read_stream = None
        self._write_stream = None
        self._connected = False
    
    async def search_products_async(
        self,
        query: str,
        top_k: int = 20,
        use_reranking: bool = True,
        rerank_top_k: int = 30
    ) -> List[ShoppingSearchResult]:
        """
        Search for products asynchronously.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            rerank_top_k: Number of candidates to rerank
            
        Returns:
            List of ShoppingSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "search_products",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking,
                    "rerank_top_k": rerank_top_k
                }
            )
            
            return self._parse_search_response(result)
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []
    
    async def search_reviews_async(
        self,
        query: str,
        top_k: int = 20,
        use_reranking: bool = True,
        rerank_top_k: int = 30
    ) -> List[ShoppingSearchResult]:
        """
        Search for reviews asynchronously.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            rerank_top_k: Number of candidates to rerank
            
        Returns:
            List of ShoppingSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "search_reviews",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking,
                    "rerank_top_k": rerank_top_k
                }
            )
            
            return self._parse_search_response(result)
            
        except Exception as e:
            logger.error(f"Review search failed: {e}")
            return []
    
    async def hybrid_search_async(
        self,
        query: str,
        top_k: int = 20,
        use_reranking: bool = True,
        rerank_top_k: int = 30,
        rrf_k: int = 60
    ) -> List[ShoppingSearchResult]:
        """
        Search for both products and reviews asynchronously.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            rerank_top_k: Number of candidates to rerank
            rrf_k: RRF fusion parameter
            
        Returns:
            List of ShoppingSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "hybrid_search",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking,
                    "rerank_top_k": rerank_top_k,
                    "rrf_k": rrf_k
                }
            )
            
            return self._parse_search_response(result)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def web_search_async(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "general",
        use_cache: bool = True
    ) -> List[ShoppingSearchResult]:
        """
        Search the web for product information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: Type of search (general, price, review, comparison)
            use_cache: Whether to use cached results
            
        Returns:
            List of ShoppingSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "web_search",
                {
                    "query": query,
                    "top_k": top_k,
                    "search_type": search_type,
                    "use_cache": use_cache
                }
            )
            
            return self._parse_search_response(result)
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def hybrid_search_with_web_async(
        self,
        query: str,
        top_k: int = 20,
        use_reranking: bool = True,
        force_web: bool = False
    ) -> List[ShoppingSearchResult]:
        """
        Search combining local data and web results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            force_web: Force web-only search
            
        Returns:
            List of ShoppingSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "hybrid_search_with_web",
                {
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking,
                    "force_web": force_web
                }
            )
            
            return self._parse_search_response(result)
            
        except Exception as e:
            logger.error(f"Hybrid web search failed: {e}")
            return []
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Get statistics about the search system.
        
        Returns:
            Dictionary with system statistics
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.read_resource("search://stats")
            
            if result.contents and len(result.contents) > 0:
                data = result.contents[0]
                if hasattr(data, 'text'):
                    return json.loads(data.text)
                else:
                    return data
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _parse_search_response(self, result) -> List[ShoppingSearchResult]:
        """Parse MCP response into search results."""
        if not result.content or len(result.content) == 0:
            return []
        
        data = result.content[0]
        if hasattr(data, 'text'):
            response_data = json.loads(data.text)
        else:
            response_data = data
        
        results = []
        for item in response_data.get("results", []):
            results.append(ShoppingSearchResult(
                id=item.get("id", ""),
                type=item.get("type", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                ce_score=item.get("ce_score", 0.0),
                metadata=item.get("metadata")
            ))
        
        return results
    
    # Synchronous wrappers for convenience
    def search_products(self, query: str, **kwargs) -> List[ShoppingSearchResult]:
        """Synchronous wrapper for product search."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(self.search_products_async(query, **kwargs))
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(self.search_products_async(query, **kwargs))
    
    def search_reviews(self, query: str, **kwargs) -> List[ShoppingSearchResult]:
        """Synchronous wrapper for review search."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(self.search_reviews_async(query, **kwargs))
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(self.search_reviews_async(query, **kwargs))
    
    def hybrid_search(self, query: str, **kwargs) -> List[ShoppingSearchResult]:
        """Synchronous wrapper for hybrid search."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(self.hybrid_search_async(query, **kwargs))
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(self.hybrid_search_async(query, **kwargs))
    
    def web_search(self, query: str, **kwargs) -> List[ShoppingSearchResult]:
        """Synchronous wrapper for web search."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(self.web_search_async(query, **kwargs))
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(self.web_search_async(query, **kwargs))
    
    def get_stats(self) -> Dict[str, Any]:
        """Synchronous wrapper for getting stats."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(self.get_stats_async())
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(self.get_stats_async())
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.session:
            try:
                asyncio.run(self.disconnect())
            except:
                pass


class ShoppingAssistantAdapter:
    """
    Simple adapter for easy use of the ShoppingAssistant MCP server.
    
    Example:
        assistant = ShoppingAssistantAdapter()
        results = assistant.search("wireless earbuds")
        for r in results:
            print(f"{r.title} (score: {r.score})")
    """
    
    def __init__(self):
        self.client = ShoppingAssistantMCPClient()
        self._connected = False
    
    def _ensure_connected(self):
        """Ensure client is connected."""
        if not self._connected:
            asyncio.run(self.client.connect())
            self._connected = True
    
    def search(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 20,
        **kwargs
    ) -> List[ShoppingSearchResult]:
        """
        Perform a search based on type.
        
        Args:
            query: Search query
            search_type: Type of search (products, reviews, hybrid, web, hybrid_web)
            top_k: Number of results
            **kwargs: Additional parameters
            
        Returns:
            List of search results
        """
        self._ensure_connected()
        
        if search_type == "products":
            return self.client.search_products(query, top_k=top_k, **kwargs)
        elif search_type == "reviews":
            return self.client.search_reviews(query, top_k=top_k, **kwargs)
        elif search_type == "web":
            return self.client.web_search(query, top_k=top_k, **kwargs)
        elif search_type == "hybrid_web":
            return self.client.hybrid_search_with_web_async(query, top_k=top_k, **kwargs)
        else:  # default to hybrid
            return self.client.hybrid_search(query, top_k=top_k, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        self._ensure_connected()
        return self.client.get_stats()


# Factory function for easy instantiation
def get_shopping_assistant() -> ShoppingAssistantAdapter:
    """
    Get a ShoppingAssistant MCP client adapter.
    
    Returns:
        ShoppingAssistantAdapter ready for use
    """
    return ShoppingAssistantAdapter()