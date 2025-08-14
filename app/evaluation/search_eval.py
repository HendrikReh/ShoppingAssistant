"""Search evaluation functionality."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search evaluation results."""
    query: str
    variant: str
    contexts_retrieved: List[Dict]
    num_results: int
    search_time: float
    success: bool
    metadata: Dict[str, Any] = None


class SearchEvaluator:
    """Evaluate search system performance."""
    
    def __init__(self, search_fn):
        """Initialize search evaluator.
        
        Args:
            search_fn: Function that performs search
        """
        self.search_fn = search_fn
        self.results = []
    
    def evaluate_query(
        self,
        query: str,
        variant: str = "hybrid",
        top_k: int = 20,
        **kwargs
    ) -> SearchResult:
        """Evaluate a single query.
        
        Args:
            query: Search query
            variant: Search variant to use
            top_k: Number of results to retrieve
            **kwargs: Additional search parameters
            
        Returns:
            SearchResult object
        """
        start_time = time.time()
        
        try:
            # Perform search
            search_results = self.search_fn(
                query=query,
                variant=variant,
                top_k=top_k,
                **kwargs
            )
            
            # Extract contexts
            contexts = []
            for doc_id, score, payload in search_results:
                contexts.append({
                    "id": doc_id,
                    "score": score,
                    "content": payload
                })
            
            result = SearchResult(
                query=query,
                variant=variant,
                contexts_retrieved=contexts,
                num_results=len(contexts),
                search_time=time.time() - start_time,
                success=len(contexts) > 0,
                metadata=kwargs
            )
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            result = SearchResult(
                query=query,
                variant=variant,
                contexts_retrieved=[],
                num_results=0,
                search_time=time.time() - start_time,
                success=False,
                metadata={"error": str(e)}
            )
        
        self.results.append(result)
        return result
    
    def evaluate_dataset(
        self,
        queries: List[str],
        variant: str = "hybrid",
        top_k: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate multiple queries.
        
        Args:
            queries: List of queries
            variant: Search variant
            top_k: Number of results
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics
        """
        results = []
        for query in queries:
            result = self.evaluate_query(query, variant, top_k, **kwargs)
            results.append(result)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        metrics["variant"] = variant
        metrics["detailed_results"] = [asdict(r) for r in results]
        
        return metrics
    
    def calculate_metrics(self, results: List[SearchResult]) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {}
        
        num_queries = len(results)
        successful_queries = sum(1 for r in results if r.success)
        total_results = sum(r.num_results for r in results)
        avg_time = sum(r.search_time for r in results) / num_queries
        
        metrics = {
            "num_queries": num_queries,
            "queries_with_results": successful_queries / num_queries,
            "avg_contexts_retrieved": total_results / num_queries,
            "avg_search_time": avg_time,
            "success_rate": successful_queries / num_queries
        }
        
        return metrics


def evaluate_search_variants(
    queries: List[str],
    search_fn,
    variants: List[str] = ["bm25", "vec", "rrf", "rrf_ce"],
    top_k: int = 20,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """Evaluate multiple search variants.
    
    Args:
        queries: List of queries
        search_fn: Search function
        variants: List of variants to evaluate
        top_k: Number of results
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping variant to metrics
    """
    results = {}
    
    for variant in variants:
        logger.info(f"Evaluating variant: {variant}")
        evaluator = SearchEvaluator(search_fn)
        metrics = evaluator.evaluate_dataset(
            queries, variant, top_k, **kwargs
        )
        results[variant] = metrics
    
    return results


def compare_variants(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare performance across variants.
    
    Args:
        results: Variant evaluation results
        
    Returns:
        Comparison summary
    """
    comparison = {
        "best_by_success_rate": "",
        "best_by_results_count": "",
        "best_by_speed": "",
        "variant_rankings": {}
    }
    
    # Find best performers
    best_success = ("", 0)
    best_results = ("", 0)
    best_speed = ("", float('inf'))
    
    for variant, metrics in results.items():
        success_rate = metrics.get("success_rate", 0)
        avg_results = metrics.get("avg_contexts_retrieved", 0)
        avg_time = metrics.get("avg_search_time", float('inf'))
        
        if success_rate > best_success[1]:
            best_success = (variant, success_rate)
        
        if avg_results > best_results[1]:
            best_results = (variant, avg_results)
        
        if avg_time < best_speed[1]:
            best_speed = (variant, avg_time)
        
        comparison["variant_rankings"][variant] = {
            "success_rate": success_rate,
            "avg_results": avg_results,
            "avg_time": avg_time
        }
    
    comparison["best_by_success_rate"] = best_success[0]
    comparison["best_by_results_count"] = best_results[0]
    comparison["best_by_speed"] = best_speed[0]
    
    return comparison