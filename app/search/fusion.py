"""Reciprocal Rank Fusion (RRF) for combining search results."""

import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def rrf_fuse(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    product_boost: float = 1.5
) -> Dict[str, float]:
    """Reciprocal Rank Fusion to combine multiple result lists.
    
    Args:
        result_lists: List of result lists, each containing (doc_id, score) tuples
        k: RRF parameter (typically 60)
        product_boost: Boost factor for product results over reviews
        
    Returns:
        Dictionary mapping doc_id to fused score
    """
    fused_scores = {}
    
    for result_list in result_lists:
        for rank, (doc_id, _) in enumerate(result_list, start=1):
            # Apply product boost if this is a product
            boost = product_boost if doc_id.startswith("prod::") else 1.0
            
            # RRF formula: 1 / (k + rank)
            score = boost / (k + rank)
            
            # Accumulate scores
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += score
    
    return fused_scores


def combine_search_results(
    bm25_results: List[Tuple[str, float]],
    vector_results: List[Tuple[str, float]],
    rrf_k: int = 60,
    product_boost: float = 1.5
) -> List[Tuple[str, float]]:
    """Combine BM25 and vector search results using RRF.
    
    Args:
        bm25_results: BM25 search results
        vector_results: Vector search results
        rrf_k: RRF parameter
        product_boost: Boost for products
        
    Returns:
        Combined and sorted results
    """
    # Apply RRF fusion
    fused_scores = rrf_fuse(
        [bm25_results, vector_results],
        k=rrf_k,
        product_boost=product_boost
    )
    
    # Sort by score
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


def filter_by_score(
    results: List[Tuple[str, float]],
    min_score: float
) -> List[Tuple[str, float]]:
    """Filter results by minimum score.
    
    Args:
        results: List of (doc_id, score) tuples
        min_score: Minimum score threshold
        
    Returns:
        Filtered results
    """
    return [(doc_id, score) for doc_id, score in results if score >= min_score]


def deduplicate_results(
    results: List[Tuple[str, float, Dict]]
) -> List[Tuple[str, float, Dict]]:
    """Remove duplicate results, keeping highest score.
    
    Args:
        results: List of (doc_id, score, payload) tuples
        
    Returns:
        Deduplicated results
    """
    seen = {}
    for doc_id, score, payload in results:
        if doc_id not in seen or score > seen[doc_id][0]:
            seen[doc_id] = (score, payload)
    
    # Reconstruct list
    deduped = [(doc_id, score, payload) for doc_id, (score, payload) in seen.items()]
    
    # Sort by score
    deduped.sort(key=lambda x: x[1], reverse=True)
    
    return deduped