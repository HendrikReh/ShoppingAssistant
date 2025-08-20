"""
MCP Client for Tavily Web Search

This client provides a bridge between the ShoppingAssistant application
and the Tavily MCP server, maintaining backward compatibility with the
existing TavilyWebSearchAgent interface.
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
class WebSearchResult:
    """Web search result matching existing interface."""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None

class TavilyMCPClient:
    """
    MCP client for Tavily web search server.
    
    This client maintains backward compatibility with TavilyWebSearchAgent
    while using the MCP protocol for communication.
    """
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._context = None
        self._read_stream = None
        self._write_stream = None
        
    async def connect(self):
        """Connect to the Tavily MCP server."""
        if self._connected:
            return  # Already connected
            
        try:
            # Start server process and connect
            server_script = Path(__file__).parent / "tavily_mcp_server.py"
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "python", str(server_script)],
                env={
                    **os.environ,  # Include all env vars
                    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                    "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0")
                }
            )
            
            # Create context and enter it
            self._context = stdio_client(server_params)
            self._read_stream, self._write_stream = await self._context.__aenter__()
            
            # Create client session
            self.session = ClientSession(self._read_stream, self._write_stream)
            
            # Initialize session
            await self.session.initialize()
            
            self._connected = True
            logger.info("Connected to Tavily MCP server")
            
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
    
    async def search_async(
        self,
        query: str,
        search_type: str = "general",
        max_results: int = 10,
        use_cache: bool = True,
        **kwargs
    ) -> List[WebSearchResult]:
        """
        Perform asynchronous web search through MCP.
        
        Args:
            query: Search query
            search_type: Type of search
            max_results: Maximum results
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            List of WebSearchResult objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Call MCP tool
            result = await self.session.call_tool(
                "web_search",
                {
                    "query": query,
                    "search_type": search_type,
                    "max_results": max_results,
                    "use_cache": use_cache,
                    **kwargs
                }
            )
            
            # Parse response
            if result.content and len(result.content) > 0:
                data = result.content[0]
                if hasattr(data, 'text'):
                    # Parse JSON text response
                    response_data = json.loads(data.text)
                else:
                    response_data = data
                
                # Convert to WebSearchResult objects
                results = []
                for item in response_data.get("results", []):
                    results.append(WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                        published_date=item.get("published_date")
                    ))
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search(
        self,
        query: str,
        search_type: str = "general",
        use_cache: bool = True,
        **kwargs
    ) -> List[WebSearchResult]:
        """
        Synchronous wrapper for search (backward compatibility).
        
        Args:
            query: Search query
            search_type: Type of search
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            List of WebSearchResult objects
        """
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task
            future = asyncio.create_task(
                self.search_async(query, search_type, use_cache=use_cache, **kwargs)
            )
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(
                self.search_async(query, search_type, use_cache=use_cache, **kwargs)
            )
    
    async def search_product_info_async(
        self,
        product_name: str,
        info_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Asynchronous product information search.
        
        Args:
            product_name: Product name
            info_type: Type of info (price, availability, reviews, all)
            
        Returns:
            Dictionary with product information
        """
        if not self._connected:
            await self.connect()
        
        info = {}
        
        try:
            if info_type in ["price", "all"]:
                result = await self.session.call_tool(
                    "search_product_prices",
                    {"product_name": product_name}
                )
                if result.content:
                    data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else result.content[0]
                    info["prices"] = data.get("prices", [])
            
            if info_type in ["availability", "all"]:
                result = await self.session.call_tool(
                    "check_availability",
                    {"product_name": product_name}
                )
                if result.content:
                    data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else result.content[0]
                    info["availability"] = data.get("retailers", [])
            
            if info_type in ["reviews", "all"]:
                result = await self.session.call_tool(
                    "search_product_reviews",
                    {"product_name": product_name}
                )
                if result.content:
                    data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else result.content[0]
                    info["reviews"] = data.get("results", [])
            
        except Exception as e:
            logger.error(f"Product info search failed: {e}")
        
        return info
    
    def search_product_info(
        self,
        product_name: str,
        info_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for product info search.
        
        Args:
            product_name: Product name
            info_type: Type of info
            
        Returns:
            Dictionary with product information
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(
                self.search_product_info_async(product_name, info_type)
            )
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(
                self.search_product_info_async(product_name, info_type)
            )
    
    async def find_alternatives_async(
        self,
        product_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find alternative products asynchronously.
        
        Args:
            product_name: Product to find alternatives for
            max_results: Maximum results
            
        Returns:
            List of alternative products
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.session.call_tool(
                "find_alternatives",
                {"product_name": product_name}
            )
            
            if result.content:
                data = json.loads(result.content[0].text) if hasattr(result.content[0], 'text') else result.content[0]
                return data.get("results", [])[:max_results]
            
            return []
            
        except Exception as e:
            logger.error(f"Find alternatives failed: {e}")
            return []
    
    def find_alternatives(
        self,
        product_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for finding alternatives.
        
        Args:
            product_name: Product name
            max_results: Maximum results
            
        Returns:
            List of alternatives
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.create_task(
                self.find_alternatives_async(product_name, max_results)
            )
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            return asyncio.run(
                self.find_alternatives_async(product_name, max_results)
            )
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.session:
            try:
                asyncio.run(self.disconnect())
            except:
                pass


class TavilyMCPAdapter:
    """
    Adapter class that provides full backward compatibility with TavilyWebSearchAgent.
    
    This allows gradual migration from the old implementation to MCP.
    """
    
    def __init__(self, config=None, cache=None):
        """
        Initialize adapter.
        
        Args:
            config: WebSearchConfig (ignored, uses MCP server config)
            cache: WebSearchCache (ignored, MCP server handles caching)
        """
        self.client = TavilyMCPClient()
        self._connected = False
    
    def _ensure_connected(self):
        """Ensure client is connected."""
        if not self._connected:
            asyncio.run(self.client.connect())
            self._connected = True
    
    def search(
        self,
        query: str,
        search_type: str = "general",
        use_cache: bool = True,
        **kwargs
    ) -> List[WebSearchResult]:
        """
        Perform web search (backward compatible interface).
        
        Args:
            query: Search query
            search_type: Type of search
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            List of WebSearchResult objects
        """
        self._ensure_connected()
        return self.client.search(query, search_type, use_cache, **kwargs)
    
    def search_product_info(
        self,
        product_name: str,
        info_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Search for product information.
        
        Args:
            product_name: Product name
            info_type: Type of information
            
        Returns:
            Dictionary with product info
        """
        self._ensure_connected()
        return self.client.search_product_info(product_name, info_type)
    
    def find_alternatives(
        self,
        product_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find alternative products.
        
        Args:
            product_name: Product name
            max_results: Maximum results
            
        Returns:
            List of alternatives
        """
        self._ensure_connected()
        return self.client.find_alternatives(product_name, max_results)
    
    def parallel_search(
        self,
        queries: List[str],
        search_type: str = "general"
    ) -> Dict[str, List[WebSearchResult]]:
        """
        Perform multiple searches in parallel.
        
        Args:
            queries: List of queries
            search_type: Type of search
            
        Returns:
            Dictionary mapping queries to results
        """
        self._ensure_connected()
        
        async def _parallel_search():
            tasks = [
                self.client.search_async(q, search_type)
                for q in queries
            ]
            results = await asyncio.gather(*tasks)
            return dict(zip(queries, results))
        
        return asyncio.run(_parallel_search())


# For drop-in replacement
def get_mcp_agent(config=None, cache=None):
    """
    Factory function to get MCP-based Tavily agent.
    
    This can be used as a drop-in replacement for TavilyWebSearchAgent:
    
    Before:
        agent = TavilyWebSearchAgent(config, cache)
    
    After:
        agent = get_mcp_agent(config, cache)
    """
    return TavilyMCPAdapter(config, cache)