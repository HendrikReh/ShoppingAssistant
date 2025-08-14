"""Tavily Web Search Agent for real-time product information and enhanced retrieval.

This module integrates Tavily's web search API to augment local product search with:
- Real-time pricing and availability
- Professional reviews and comparisons
- Latest product releases and updates
- Market trends and alternatives
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("Tavily client not found. Install with: pip install tavily-python")

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass
class WebSearchConfig:
    """Configuration for web search behavior."""
    api_key: str
    enable_web_search: bool = True
    search_depth: str = "advanced"  # "basic" or "advanced"
    max_results: int = 10
    include_images: bool = False
    include_raw_content: bool = False
    cache_ttl_seconds: int = 7200  # 2 hours for prices
    review_cache_ttl: int = 86400  # 24 hours for reviews
    domains_whitelist: List[str] = None
    domains_blacklist: List[str] = None
    min_relevance_threshold: float = 0.3
    
    def __post_init__(self):
        if self.domains_whitelist is None:
            self.domains_whitelist = [
                "amazon.com",
                "bestbuy.com",
                "newegg.com",
                "walmart.com",
                "target.com",
                "bhphotovideo.com",
                "cnet.com",
                "techradar.com",
                "rtings.com",
                "wirecutter.com",
                "tomsguide.com",
                "pcmag.com"
            ]
        if self.domains_blacklist is None:
            self.domains_blacklist = []


@dataclass
class WebSearchResult:
    """Structured web search result."""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None
    domain: str = ""
    image_url: Optional[str] = None
    raw_content: Optional[str] = None
    search_query: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # Extract domain from URL
        if self.url and not self.domain:
            from urllib.parse import urlparse
            self.domain = urlparse(self.url).netloc


class WebSearchCache:
    """Redis-based cache for web search results."""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 7200):
        self.redis = redis_client
        self.default_ttl = default_ttl
        
    def _get_cache_key(self, query: str, search_type: str = "general") -> str:
        """Generate cache key from query and search type."""
        normalized_query = query.lower().strip()
        key_string = f"tavily:{search_type}:{normalized_query}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, search_type: str = "general") -> Optional[List[WebSearchResult]]:
        """Retrieve cached results if available and not expired."""
        try:
            key = self._get_cache_key(query, search_type)
            cached_data = self.redis.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                # Reconstruct WebSearchResult objects
                results = []
                for item in data:
                    # Convert timestamp string back to datetime
                    if 'timestamp' in item:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    results.append(WebSearchResult(**item))
                logger.info(f"Cache hit for query: {query[:50]}...")
                return results
                
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def set(self, query: str, results: List[WebSearchResult], 
            search_type: str = "general", ttl: Optional[int] = None):
        """Cache search results with TTL."""
        try:
            key = self._get_cache_key(query, search_type)
            ttl = ttl or self.default_ttl
            
            # Serialize results
            data = []
            for result in results:
                item = asdict(result)
                # Convert datetime to string for JSON serialization
                if 'timestamp' in item and item['timestamp']:
                    item['timestamp'] = item['timestamp'].isoformat()
                data.append(item)
            
            self.redis.setex(key, ttl, json.dumps(data))
            logger.info(f"Cached {len(results)} results for query: {query[:50]}...")
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.warning(f"Cache storage error: {e}")


class TavilyWebSearchAgent:
    """Agent for performing web searches using Tavily API."""
    
    def __init__(self, config: WebSearchConfig, cache: Optional[WebSearchCache] = None):
        self.config = config
        self.cache = cache
        
        # Initialize Tavily client only when needed to allow caching-only fast paths
        self._lazy_api_key = config.api_key
        self.client: Optional[TavilyClient] = None
        
        # Thread pool for parallel searches
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def search(self, query: str, 
               search_type: str = "general",
               use_cache: bool = True,
               **kwargs) -> List[WebSearchResult]:
        """
        Perform web search with caching and error handling.
        
        Args:
            query: Search query string
            search_type: Type of search (general, price, availability, review)
            use_cache: Whether to use cached results
            **kwargs: Additional Tavily search parameters
            
        Returns:
            List of WebSearchResult objects
        """
        # Check cache first
        if use_cache and self.cache:
            cached_results = self.cache.get(query, search_type)
            if cached_results:
                return cached_results
        
        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": self.config.search_depth,
                "max_results": self.config.max_results,
                "include_images": self.config.include_images,
                "include_raw_content": self.config.include_raw_content,
            }
            
            # Add domain filters if specified
            if self.config.domains_whitelist:
                search_params["include_domains"] = self.config.domains_whitelist
            if self.config.domains_blacklist:
                search_params["exclude_domains"] = self.config.domains_blacklist
            
            # Override with any kwargs
            search_params.update(kwargs)
            
            # Perform search
            logger.info(f"Performing Tavily search: {query[:50]}...")
            if self.client is None:
                self.client = TavilyClient(api_key=self._lazy_api_key)
            response = self.client.search(**search_params)
            
            # Parse results
            results = self._parse_results(response, query)
            
            # Cache results
            if self.cache and results:
                ttl = self._get_ttl_for_search_type(search_type)
                self.cache.set(query, results, search_type, ttl)
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    def search_product_info(self, product_name: str, 
                            info_type: str = "all") -> Dict[str, Any]:
        """
        Search for specific product information.
        
        Args:
            product_name: Name of the product
            info_type: Type of info (price, availability, reviews, specs, all)
            
        Returns:
            Dictionary with requested information
        """
        info = {}
        
        if info_type in ["price", "all"]:
            price_query = f"{product_name} price deals where to buy"
            price_results = self.search(price_query, search_type="price")
            info["prices"] = self._extract_prices(price_results)
        
        if info_type in ["availability", "all"]:
            avail_query = f"{product_name} in stock availability"
            avail_results = self.search(avail_query, search_type="availability")
            info["availability"] = self._extract_availability(avail_results)
        
        if info_type in ["reviews", "all"]:
            review_query = f"{product_name} review comparison pros cons"
            review_results = self.search(review_query, search_type="review")
            info["reviews"] = self._summarize_reviews(review_results)
        
        if info_type in ["specs", "all"]:
            specs_query = f"{product_name} specifications features technical details"
            specs_results = self.search(specs_query, search_type="specs")
            info["specifications"] = self._extract_specs(specs_results)
        
        return info
    
    def search_alternatives(self, product_name: str, 
                           max_alternatives: int = 5) -> List[Dict[str, Any]]:
        """Find alternative products."""
        query = f"{product_name} alternatives similar products vs comparison better than"
        results = self.search(query, search_type="alternatives")
        
        alternatives = []
        for result in results[:max_alternatives]:
            alt = {
                "name": self._extract_product_name(result.content),
                "reason": self._extract_comparison_reason(result.content),
                "source": result.domain,
                "url": result.url
            }
            alternatives.append(alt)
        
        return alternatives
    
    async def parallel_search(self, queries: List[Tuple[str, str]]) -> Dict[str, List[WebSearchResult]]:
        """
        Perform multiple searches in parallel.
        
        Args:
            queries: List of (query, search_type) tuples
            
        Returns:
            Dictionary mapping query to results
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for query, search_type in queries:
            task = loop.run_in_executor(
                self.executor,
                self.search,
                query,
                search_type
            )
            tasks.append((query, task))
        
        results = {}
        for query, task in tasks:
            try:
                search_results = await task
                results[query] = search_results
            except Exception as e:
                logger.error(f"Parallel search error for '{query}': {e}")
                results[query] = []
        
        return results
    
    def _parse_results(self, response: Dict, query: str) -> List[WebSearchResult]:
        """Parse Tavily API response into WebSearchResult objects."""
        results = []
        
        if "results" not in response:
            return results
        
        for item in response["results"]:
            result = WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
                image_url=item.get("image_url") if self.config.include_images else None,
                raw_content=item.get("raw_content") if self.config.include_raw_content else None,
                search_query=query
            )
            
            # Filter by relevance threshold
            if result.score >= self.config.min_relevance_threshold:
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _get_ttl_for_search_type(self, search_type: str) -> int:
        """Get appropriate cache TTL based on search type."""
        ttl_map = {
            "price": 3600,  # 1 hour for prices
            "availability": 1800,  # 30 minutes for stock
            "review": 86400,  # 24 hours for reviews
            "specs": 604800,  # 1 week for specifications
            "general": self.config.cache_ttl_seconds
        }
        return ttl_map.get(search_type, self.config.cache_ttl_seconds)
    
    def _extract_prices(self, results: List[WebSearchResult]) -> List[Dict[str, Any]]:
        """Extract price information from search results."""
        prices = []
        for result in results:
            # Simple price extraction (could be enhanced with regex or NLP)
            import re
            price_pattern = r'\$[\d,]+\.?\d*'
            found_prices = re.findall(price_pattern, result.content)
            
            if found_prices:
                prices.append({
                    "price": found_prices[0],
                    "source": result.domain,
                    "url": result.url,
                    "date": result.timestamp.isoformat()
                })
        
        return prices
    
    def _extract_availability(self, results: List[WebSearchResult]) -> List[Dict[str, Any]]:
        """Extract availability information from search results."""
        availability = []
        
        keywords = ["in stock", "available", "out of stock", "ships", "delivery"]
        
        for result in results:
            content_lower = result.content.lower()
            status = "unknown"
            
            if "in stock" in content_lower or "available" in content_lower:
                status = "in_stock"
            elif "out of stock" in content_lower or "unavailable" in content_lower:
                status = "out_of_stock"
            
            availability.append({
                "retailer": result.domain,
                "status": status,
                "url": result.url,
                "checked_at": result.timestamp.isoformat()
            })
        
        return availability
    
    def _summarize_reviews(self, results: List[WebSearchResult]) -> Dict[str, Any]:
        """Summarize review information from search results."""
        pros = []
        cons = []
        ratings = []
        
        for result in results:
            # Extract pros/cons (simplified - could use NLP)
            content = result.content.lower()
            
            if "pros:" in content or "advantages:" in content:
                # Extract pros section
                pros.append(result.content)
            
            if "cons:" in content or "disadvantages:" in content:
                # Extract cons section
                cons.append(result.content)
            
            # Extract ratings
            import re
            rating_pattern = r'(\d+(?:\.\d+)?)\s*(?:out of\s*)?(?:/\s*)?5'
            found_ratings = re.findall(rating_pattern, result.content)
            ratings.extend([float(r) for r in found_ratings])
        
        return {
            "pros_mentioned": len(pros),
            "cons_mentioned": len(cons),
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "sources": [r.domain for r in results[:5]]
        }
    
    def _extract_specs(self, results: List[WebSearchResult]) -> Dict[str, Any]:
        """Extract specifications from search results."""
        specs = {}
        
        for result in results:
            # Simple spec extraction (could be enhanced)
            content = result.content
            
            # Look for common spec patterns
            spec_keywords = ["processor", "ram", "storage", "display", "battery",
                           "weight", "dimensions", "resolution", "capacity"]
            
            for keyword in spec_keywords:
                if keyword in content.lower():
                    specs[keyword] = "Found in " + result.domain
        
        return specs
    
    def _extract_product_name(self, content: str) -> str:
        """Extract product name from content."""
        # Simplified extraction - could use NER
        lines = content.split('\n')
        if lines:
            return lines[0][:100]  # First line, max 100 chars
        return "Unknown Product"
    
    def _extract_comparison_reason(self, content: str) -> str:
        """Extract why this is an alternative."""
        # Look for comparison phrases
        phrases = ["better than", "alternative to", "similar to", "instead of"]
        
        content_lower = content.lower()
        for phrase in phrases:
            if phrase in content_lower:
                # Extract sentence containing the phrase
                sentences = content.split('.')
                for sentence in sentences:
                    if phrase in sentence.lower():
                        return sentence.strip()
        
        return "Similar product"