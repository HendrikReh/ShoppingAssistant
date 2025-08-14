"""Search retrieval improvements to boost context relevance.

This module implements enhancements to improve search quality:
1. Query expansion and preprocessing
2. Optimized retrieval parameters
3. Fallback strategies
4. Better keyword extraction
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QueryEnhancement:
    """Enhanced query with expansions and metadata."""
    original: str
    expanded: str
    keywords: List[str]
    entities: List[str]
    query_type: str  # factual, comparative, recommendation, etc.


class QueryPreprocessor:
    """Preprocess and expand queries for better retrieval."""
    
    def __init__(self):
        # Common synonyms and expansions for e-commerce
        self.synonyms = {
            "laptop": ["notebook", "computer", "pc", "macbook", "chromebook"],
            "earbuds": ["earphones", "headphones", "airpods", "in-ear"],
            "mouse": ["mice", "pointing device", "trackball"],
            "keyboard": ["keypad", "typing device", "mechanical keyboard"],
            "speaker": ["speakers", "audio", "sound system", "bluetooth speaker"],
            "tablet": ["ipad", "tab", "slate"],
            "monitor": ["display", "screen", "lcd", "led display"],
            "webcam": ["camera", "web camera", "video camera"],
            "storage": ["hard drive", "ssd", "hdd", "disk", "memory"],
            "cable": ["wire", "cord", "connector", "adapter"],
            
            # Feature synonyms
            "wireless": ["wifi", "bluetooth", "cordless", "wi-fi"],
            "gaming": ["gamer", "game", "esports", "gaming-grade"],
            "budget": ["cheap", "affordable", "inexpensive", "value"],
            "best": ["top", "premium", "highest rated", "recommended"],
            "portable": ["compact", "travel", "mobile", "lightweight"],
            "fast": ["quick", "rapid", "high-speed", "speedy"],
            "noise cancelling": ["noise canceling", "anc", "noise reduction"],
            "waterproof": ["water resistant", "water-resistant", "ipx", "weatherproof"],
        }
        
        # Brand variations
        self.brand_variations = {
            "apple": ["apple inc", "mac", "iphone", "ipad"],
            "samsung": ["samsung electronics", "galaxy"],
            "microsoft": ["ms", "msft", "windows"],
            "hp": ["hewlett packard", "hewlett-packard"],
            "dell": ["dell technologies", "dell inc"],
            "sony": ["sony corporation"],
            "bose": ["bose corporation"],
            "jbl": ["jbl by harman"],
        }
        
        # Query type patterns
        self.patterns = {
            "comparative": r"(compare|versus|vs|better|difference between)",
            "recommendation": r"(recommend|suggest|best|top|good for|should i)",
            "technical": r"(specification|spec|feature|support|compatible)",
            "price": r"(under|below|less than|cheaper|budget|\$\d+)",
            "problem": r"(issue|problem|fix|trouble|error|not working)",
        }
    
    def identify_query_type(self, query: str) -> str:
        """Identify the type of query for better handling."""
        query_lower = query.lower()
        
        for qtype, pattern in self.patterns.items():
            if re.search(pattern, query_lower):
                return qtype
        
        # Default to factual
        return "factual"
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract product entities and brands from query."""
        entities = []
        query_lower = query.lower()
        
        # Check for brands
        for brand, variations in self.brand_variations.items():
            if brand in query_lower:
                entities.append(brand)
            for var in variations:
                if var in query_lower:
                    entities.append(brand)
                    break
        
        # Check for product types
        product_types = ["laptop", "earbuds", "mouse", "keyboard", "speaker", 
                        "tablet", "monitor", "webcam", "storage", "cable",
                        "headphones", "earphones", "computer", "display"]
        
        for product in product_types:
            if product in query_lower:
                entities.append(product)
        
        return list(set(entities))
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and variations."""
        expanded_terms = [query]  # Start with original
        query_lower = query.lower()
        
        # Add synonyms
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add relevant synonyms
                for syn in synonyms[:2]:  # Limit to avoid over-expansion
                    expanded_terms.append(query_lower.replace(term, syn))
        
        # Add brand variations
        for brand, variations in self.brand_variations.items():
            if brand in query_lower:
                for var in variations[:1]:  # Add one variation
                    if var not in query_lower:
                        expanded_terms.append(f"{query} {var}")
        
        # Combine unique terms
        expanded = " ".join(list(set(expanded_terms)))
        
        # Limit length to avoid dilution
        if len(expanded.split()) > 30:
            expanded = " ".join(expanded.split()[:30])
        
        return expanded
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for BM25."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
                     'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'what', 'which', 'who', 'where',
                     'when', 'how', 'why', 'i', 'me', 'my', 'find', 'show'}
        
        # Tokenize and filter
        tokens = re.findall(r'\w+', query.lower())
        keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        # Add bigrams for compound terms
        bigrams = []
        for i in range(len(tokens) - 1):
            if tokens[i] not in stop_words and tokens[i+1] not in stop_words:
                bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
        
        return keywords + bigrams[:3]  # Limit bigrams
    
    def process(self, query: str) -> QueryEnhancement:
        """Full query processing pipeline."""
        query_type = self.identify_query_type(query)
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        expanded = self.expand_query(query)
        
        return QueryEnhancement(
            original=query,
            expanded=expanded,
            keywords=keywords,
            entities=entities,
            query_type=query_type
        )


class ImprovedRetriever:
    """Enhanced retrieval with multiple strategies."""
    
    def __init__(self, top_k: int = 40, rrf_k: int = 100):
        """Initialize with improved default parameters.
        
        Args:
            top_k: Increased from 20 to 40 for better recall
            rrf_k: Increased from 60 to 100 for better fusion
        """
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.preprocessor = QueryPreprocessor()
    
    def retrieve_with_fallback(
        self,
        query: str,
        bm25_func,
        vector_func,
        id_to_doc: Dict
    ) -> List[Tuple[str, float]]:
        """Retrieve with fallback strategies.
        
        If initial retrieval has low scores, try alternative approaches.
        """
        # Process query
        enhanced = self.preprocessor.process(query)
        
        # Get initial results
        results = self._hybrid_retrieve(
            enhanced.expanded,
            bm25_func,
            vector_func,
            enhanced.keywords
        )
        
        # Check quality
        if not results or (results and results[0][1] < 0.3):
            # Low quality - try fallback
            if enhanced.entities:
                # Entity-focused search
                entity_query = " ".join(enhanced.entities)
                entity_results = self._hybrid_retrieve(
                    entity_query,
                    bm25_func,
                    vector_func,
                    enhanced.entities
                )
                results = self._merge_results(results, entity_results)
            
            # If still poor, try keyword-only
            if results and results[0][1] < 0.4:
                keyword_query = " ".join(enhanced.keywords[:5])
                keyword_results = self._hybrid_retrieve(
                    keyword_query,
                    bm25_func,
                    vector_func,
                    enhanced.keywords
                )
                results = self._merge_results(results, keyword_results)
        
        return results[:self.top_k]
    
    def _hybrid_retrieve(
        self,
        query: str,
        bm25_func,
        vector_func,
        keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """Hybrid retrieval with optimized fusion."""
        # Get BM25 results - use keywords for better matching
        bm25_scores = bm25_func(keywords if keywords else query.split())
        
        # Get vector results - use full expanded query
        vector_scores = vector_func(query)
        
        # Rank-based RRF fusion to avoid mixing incomparable score scales
        bm25_sorted = sorted(bm25_scores, key=lambda x: x[1], reverse=True)
        vec_sorted = sorted(vector_scores, key=lambda x: x[1], reverse=True)

        fused_scores: Dict[str, float] = {}

        bm25_weight = 1.2 if len(keywords) > 0 else 1.0
        vector_weight = 1.0

        # Add BM25 ranks
        for rank_idx, (doc_id, _score) in enumerate(bm25_sorted, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + bm25_weight / (self.rrf_k + rank_idx)

        # Add vector ranks
        for rank_idx, (doc_id, _score) in enumerate(vec_sorted, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + vector_weight / (self.rrf_k + rank_idx)

        results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return results
    
    def _merge_results(
        self,
        results1: List[Tuple[str, float]],
        results2: List[Tuple[str, float]],
        weight1: float = 0.6,
        weight2: float = 0.4
    ) -> List[Tuple[str, float]]:
        """Merge two result sets with weighted scores."""
        merged = {}
        
        for doc_id, score in results1:
            merged[doc_id] = score * weight1
        
        for doc_id, score in results2:
            if doc_id in merged:
                merged[doc_id] += score * weight2
            else:
                merged[doc_id] = score * weight2
        
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)


def optimize_search_parameters(current_relevance: float) -> Dict[str, int]:
    """Recommend optimized parameters based on current performance.
    
    Args:
        current_relevance: Current context relevance score (0-1)
    
    Returns:
        Dictionary with recommended parameters
    """
    if current_relevance < 0.4:
        # Very poor - aggressive settings
        return {
            "top_k": 50,
            "rrf_k": 120,
            "rerank_top_k": 40,
            "retrieval_boost": True
        }
    elif current_relevance < 0.6:
        # Poor - moderate boost
        return {
            "top_k": 40,
            "rrf_k": 100,
            "rerank_top_k": 35,
            "retrieval_boost": True
        }
    elif current_relevance < 0.75:
        # Acceptable - slight boost
        return {
            "top_k": 30,
            "rrf_k": 80,
            "rerank_top_k": 30,
            "retrieval_boost": False
        }
    else:
        # Good - maintain
        return {
            "top_k": 20,
            "rrf_k": 60,
            "rerank_top_k": 30,
            "retrieval_boost": False
        }


class RetrievalDiagnostics:
    """Diagnose retrieval issues and suggest fixes."""
    
    @staticmethod
    def analyze_failure_patterns(
        queries: List[str],
        relevance_scores: List[float]
    ) -> Dict[str, any]:
        """Analyze which types of queries are failing."""
        preprocessor = QueryPreprocessor()
        
        failures_by_type = {}
        
        for query, score in zip(queries, relevance_scores):
            if score < 0.5:  # Failed query
                qtype = preprocessor.identify_query_type(query)
                if qtype not in failures_by_type:
                    failures_by_type[qtype] = []
                failures_by_type[qtype].append((query, score))
        
        # Generate report
        report = {
            "total_failures": sum(1 for s in relevance_scores if s < 0.5),
            "failure_rate": sum(1 for s in relevance_scores if s < 0.5) / len(relevance_scores),
            "failures_by_type": {
                qtype: {
                    "count": len(queries),
                    "avg_score": np.mean([s for _, s in queries]),
                    "examples": queries[:3]
                }
                for qtype, queries in failures_by_type.items()
            },
            "recommendations": []
        }
        
        # Add specific recommendations
        if "comparative" in failures_by_type:
            report["recommendations"].append(
                "Comparative queries failing - implement multi-document retrieval"
            )
        
        if "technical" in failures_by_type:
            report["recommendations"].append(
                "Technical queries failing - enhance specification extraction"
            )
        
        if report["failure_rate"] > 0.5:
            report["recommendations"].append(
                "High failure rate - consider reindexing with better embeddings"
            )
        
        return report


# Configuration for improved search
IMPROVED_SEARCH_CONFIG = {
    "top_k": 40,  # Increased from 20
    "rrf_k": 100,  # Increased from 60
    "rerank_top_k": 35,  # Increased from 30
    "use_query_expansion": True,
    "use_fallback": True,
    "max_expansion_terms": 30,
    "entity_boost": 1.2,
    "keyword_boost": 1.3
}