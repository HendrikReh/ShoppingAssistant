"""Hybrid Retrieval Orchestrator that combines local and web search intelligently.

This module orchestrates retrieval from multiple sources:
- Local BM25 and vector search
- Tavily web search when needed
- Intelligent routing based on query intent and local result quality
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

from app.web_search_agent import TavilyWebSearchAgent, WebSearchResult, WebSearchConfig
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RetrievalSource(Enum):
    """Source of retrieval results."""
    LOCAL_BM25 = "local_bm25"
    LOCAL_VECTOR = "local_vector"
    LOCAL_HYBRID = "local_hybrid"
    WEB_SEARCH = "web_search"
    HYBRID_ALL = "hybrid_all"


@dataclass
class RetrievalDecision:
    """Decision about which retrieval strategies to use."""
    use_local: bool = True
    use_web: bool = False
    web_search_reason: str = ""
    priority_source: RetrievalSource = RetrievalSource.LOCAL_HYBRID
    confidence: float = 1.0


@dataclass
class HybridResult:
    """Unified result from hybrid retrieval."""
    id: str
    title: str
    content: str
    score: float
    source: RetrievalSource
    metadata: Dict[str, Any]
    url: Optional[str] = None
    
    @property
    def is_product(self) -> bool:
        return self.id.startswith("prod::")
    
    @property
    def is_review(self) -> bool:
        return self.id.startswith("rev::")
    
    @property
    def is_web(self) -> bool:
        return self.source == RetrievalSource.WEB_SEARCH


class QueryIntentAnalyzer:
    """Analyze query intent to determine retrieval strategy."""
    
    # Keywords indicating need for web search
    WEB_SEARCH_TRIGGERS = {
        "price": ["price", "cost", "cheap", "expensive", "deal", "discount", "sale"],
        "availability": ["stock", "available", "buy", "purchase", "where to", "ship"],
        "comparison": ["vs", "versus", "compare", "better", "alternative", "or"],
        "latest": ["latest", "newest", "2024", "2025", "recent", "new", "updated"],
        "review": ["review", "rating", "opinion", "experience", "recommend"],
        "specs": ["specifications", "specs", "technical", "details", "features"]
    }
    
    # Keywords indicating local search is sufficient
    LOCAL_SEARCH_KEYWORDS = {
        "basic": ["what is", "how does", "explain", "describe"],
        "known_products": ["fire tv", "echo", "kindle", "airpods", "iphone"]
    }
    
    def analyze(self, query: str, local_relevance_scores: List[float] = None) -> RetrievalDecision:
        """
        Analyze query to determine retrieval strategy.
        
        Args:
            query: User query
            local_relevance_scores: Relevance scores from initial local search
            
        Returns:
            RetrievalDecision object
        """
        decision = RetrievalDecision()
        query_lower = query.lower()
        
        # Check if query needs web search
        web_search_needed = False
        web_search_reasons = []
        
        # Check for web search triggers
        for category, keywords in self.WEB_SEARCH_TRIGGERS.items():
            if any(keyword in query_lower for keyword in keywords):
                web_search_needed = True
                web_search_reasons.append(category)
        
        # Check local result quality if available
        if local_relevance_scores:
            avg_score = sum(local_relevance_scores) / len(local_relevance_scores) if local_relevance_scores else 0
            max_score = max(local_relevance_scores) if local_relevance_scores else 0
            
            if max_score < 0.3:  # Very poor local results
                web_search_needed = True
                web_search_reasons.append("low_local_relevance")
                decision.confidence = 0.3
            elif max_score < 0.5:  # Mediocre local results
                web_search_needed = True
                web_search_reasons.append("medium_local_relevance")
                decision.confidence = 0.6
            else:
                decision.confidence = 0.9
        
        # Check if query is asking for latest information
        if any(year in query_lower for year in ["2024", "2025", "latest", "newest"]):
            web_search_needed = True
            web_search_reasons.append("temporal_query")
        
        # Set decision
        decision.use_web = web_search_needed
        decision.web_search_reason = ", ".join(web_search_reasons) if web_search_reasons else ""
        
        # Determine priority source
        if web_search_needed and not decision.use_local:
            decision.priority_source = RetrievalSource.WEB_SEARCH
        elif web_search_needed and decision.use_local:
            decision.priority_source = RetrievalSource.HYBRID_ALL
        else:
            decision.priority_source = RetrievalSource.LOCAL_HYBRID
        
        return decision


class HybridRetrievalOrchestrator:
    """Orchestrates retrieval from multiple sources."""
    
    def __init__(self,
                 st_model: Optional[SentenceTransformer] = None,
                 qdrant_client: Optional[QdrantClient] = None,
                 bm25_products: Optional[BM25Okapi] = None,
                 bm25_reviews: Optional[BM25Okapi] = None,
                 web_search_agent: Optional[TavilyWebSearchAgent] = None,
                 enable_web_search: bool = True):
        """
        Initialize orchestrator with retrieval components.
        
        Args:
            st_model: Sentence transformer for embeddings
            qdrant_client: Qdrant client for vector search
            bm25_products: BM25 index for products
            bm25_reviews: BM25 index for reviews
            web_search_agent: Tavily web search agent
            enable_web_search: Whether to enable web search
        """
        self.st_model = st_model
        self.qdrant = qdrant_client
        self.bm25_products = bm25_products
        self.bm25_reviews = bm25_reviews
        self.web_agent = web_search_agent
        # Keep flag exactly as passed; downstream can still check agent presence
        self.enable_web_search = enable_web_search
        
        self.intent_analyzer = QueryIntentAnalyzer()
    
    def retrieve(self,
                 query: str,
                 top_k: int = 20,
                 enable_web: Optional[bool] = None,
                 force_web: bool = False) -> List[HybridResult]:
        """
        Main retrieval method that orchestrates all sources.
        
        Args:
            query: User query
            top_k: Number of results to return
            enable_web: Override web search setting
            force_web: Force web search regardless of intent
            
        Returns:
            List of HybridResult objects
        """
        # Determine if web search should be used
        use_web = enable_web if enable_web is not None else self.enable_web_search
        
        # Perform initial local search to assess quality
        local_results = []
        local_scores = []
        
        if not force_web:
            local_results = self._local_hybrid_search(query, top_k=top_k*2)
            local_scores = [r.score for r in local_results[:10]]
        
        # Analyze query intent
        decision = self.intent_analyzer.analyze(query, local_scores)
        
        # Override with force_web if specified
        if force_web:
            decision.use_web = True
            decision.use_local = False
            decision.priority_source = RetrievalSource.WEB_SEARCH
        
        # Collect results based on decision
        all_results = []
        
        if decision.use_local and local_results:
            all_results.extend(local_results)
        
        if decision.use_web and use_web and self.web_agent:
            web_results = self._web_search(query, top_k=top_k)
            all_results.extend(web_results)
            logger.info(f"Added {len(web_results)} web results for query: {query[:50]}...")
        
        # Merge and rank results
        final_results = self._merge_and_rank(all_results, decision, top_k)
        
        return final_results
    
    async def retrieve_async(self,
                            query: str,
                            top_k: int = 20,
                            enable_web: Optional[bool] = None) -> List[HybridResult]:
        """Asynchronous version of retrieve for better performance."""
        use_web = enable_web if enable_web is not None else self.enable_web_search
        
        # Run local and web search in parallel
        tasks = []
        
        # Local search task
        loop = asyncio.get_event_loop()
        local_task = loop.run_in_executor(None, self._local_hybrid_search, query, top_k*2)
        tasks.append(("local", local_task))
        
        # Analyze intent (quick, can be synchronous)
        decision = self.intent_analyzer.analyze(query, [])
        
        # Web search task if needed
        if decision.use_web and use_web and self.web_agent:
            web_task = loop.run_in_executor(None, self._web_search, query, top_k)
            tasks.append(("web", web_task))
        
        # Gather results
        all_results = []
        for source, task in tasks:
            try:
                results = await task
                all_results.extend(results)
                logger.info(f"Retrieved {len(results)} results from {source}")
            except Exception as e:
                logger.error(f"Error in {source} search: {e}")
        
        # Merge and rank
        final_results = self._merge_and_rank(all_results, decision, top_k)
        
        return final_results
    
    def _local_hybrid_search(self, query: str, top_k: int = 20) -> List[HybridResult]:
        """Perform local hybrid search (BM25 + vector)."""
        results = []
        
        if not (self.bm25_products and self.bm25_reviews):
            return results
        
        # Tokenize query for BM25
        import re
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # BM25 search
        prod_scores = self.bm25_products.get_scores(query_tokens)
        rev_scores = self.bm25_reviews.get_scores(query_tokens)
        
        # Get top BM25 results (simplified - assumes we have id mappings)
        # In production, this would be properly integrated
        
        # Vector search if available
        if self.st_model and self.qdrant:
            try:
                q_vec = self.st_model.encode([query], normalize_embeddings=True)[0].tolist()
                
                # Search products collection
                prod_hits = self.qdrant.search(
                    collection_name="products_minilm",
                    query_vector=q_vec,
                    limit=top_k
                )
                
                for hit in prod_hits:
                    result = HybridResult(
                        id=f"prod::{hit.id}",
                        title=hit.payload.get("title", ""),
                        content=hit.payload.get("description", ""),
                        score=hit.score,
                        source=RetrievalSource.LOCAL_VECTOR,
                        metadata=hit.payload
                    )
                    results.append(result)
                
                # Search reviews collection
                rev_hits = self.qdrant.search(
                    collection_name="reviews_minilm",
                    query_vector=q_vec,
                    limit=top_k
                )
                
                for hit in rev_hits:
                    result = HybridResult(
                        id=f"rev::{hit.id}",
                        title=hit.payload.get("title", ""),
                        content=hit.payload.get("text", ""),
                        score=hit.score,
                        source=RetrievalSource.LOCAL_VECTOR,
                        metadata=hit.payload
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Vector search error: {e}")
        
        return results
    
    def _web_search(self, query: str, top_k: int = 10) -> List[HybridResult]:
        """Perform web search and convert to HybridResult."""
        if not self.web_agent:
            return []
        
        results = []
        
        # Search for general product information
        web_results = self.web_agent.search(query, search_type="general")
        
        for web_result in web_results[:top_k]:
            hybrid_result = HybridResult(
                id=f"web::{hash(web_result.url)}",
                title=web_result.title,
                content=web_result.content,
                score=web_result.score,
                source=RetrievalSource.WEB_SEARCH,
                metadata={
                    "domain": web_result.domain,
                    "published_date": web_result.published_date,
                    "timestamp": web_result.timestamp.isoformat() if web_result.timestamp else None
                },
                url=web_result.url
            )
            results.append(hybrid_result)
        
        return results
    
    def _merge_and_rank(self,
                        results: List[HybridResult],
                        decision: RetrievalDecision,
                        top_k: int) -> List[HybridResult]:
        """
        Merge and rank results from different sources.
        
        Strategy:
        - Weight based on source and decision confidence
        - Deduplicate similar results
        - Ensure diversity (mix of products, reviews, web)
        """
        # Simple deduplication by title similarity
        seen_titles = set()
        unique_results = []
        
        for result in results:
            # Simple title normalization for deduplication
            title_key = result.title.lower()[:50] if result.title else ""
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(result)
        
        # Apply source-based weighting
        for result in unique_results:
            if result.source == RetrievalSource.WEB_SEARCH:
                # Boost or penalize web results based on decision
                if decision.priority_source == RetrievalSource.WEB_SEARCH:
                    result.score *= 1.2  # Boost web results
                elif decision.priority_source == RetrievalSource.LOCAL_HYBRID:
                    result.score *= 0.8  # Penalize web results
            elif result.is_product:
                # Slightly boost products over reviews
                result.score *= 1.1
        
        # Sort by adjusted score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Ensure diversity in top results
        final_results = self._ensure_diversity(unique_results, top_k)
        
        return final_results
    
    def _ensure_diversity(self,
                         results: List[HybridResult],
                         top_k: int,
                         min_web: int = 2,
                         min_local: int = 3) -> List[HybridResult]:
        """
        Ensure diversity in results.
        
        Args:
            results: Sorted results
            top_k: Number of results to return
            min_web: Minimum web results to include if available
            min_local: Minimum local results to include if available
        """
        final = []
        web_count = 0
        local_count = 0
        
        # First pass: collect by quotas
        for result in results:
            if len(final) >= top_k:
                break
            
            if result.is_web and web_count < min_web:
                final.append(result)
                web_count += 1
            elif not result.is_web and local_count < min_local:
                final.append(result)
                local_count += 1
            elif len(final) < top_k:
                # Fill remaining slots with best results
                final.append(result)
        
        # Re-sort by score to maintain ranking
        final.sort(key=lambda x: x.score, reverse=True)
        
        # Ensure quotas are respected; if not enough web/local, fill from others
        if len(final) < top_k:
            for r in results:
                if len(final) >= top_k:
                    break
                if r not in final:
                    final.append(r)
        return final[:top_k]
    
    def get_product_details(self, product_name: str) -> Dict[str, Any]:
        """
        Get comprehensive product details using all sources.
        
        Args:
            product_name: Name of the product
            
        Returns:
            Dictionary with product details from all sources
        """
        details = {
            "local_info": {},
            "web_info": {},
            "combined": {}
        }
        
        # Get local information
        local_results = self._local_hybrid_search(product_name, top_k=5)
        if local_results:
            details["local_info"] = {
                "found": True,
                "top_result": {
                    "title": local_results[0].title,
                    "content": local_results[0].content,
                    "metadata": local_results[0].metadata
                }
            }
        
        # Get web information if enabled
        if self.web_agent:
            web_info = self.web_agent.search_product_info(product_name, info_type="all")
            details["web_info"] = web_info
        
        # Combine information
        details["combined"] = self._combine_product_info(details["local_info"], details["web_info"])
        
        return details
    
    def _combine_product_info(self, local_info: Dict, web_info: Dict) -> Dict[str, Any]:
        """Intelligently combine local and web product information."""
        combined = {}
        
        # Use local for stable features
        if local_info.get("found"):
            combined["description"] = local_info["top_result"].get("content", "")
            combined["local_metadata"] = local_info["top_result"].get("metadata", {})
        
        # Use web for dynamic information
        if web_info:
            combined["current_prices"] = web_info.get("prices", [])
            combined["availability"] = web_info.get("availability", [])
            combined["latest_reviews"] = web_info.get("reviews", {})
            combined["specifications"] = web_info.get("specifications", {})
        
        return combined