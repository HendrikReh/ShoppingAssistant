"""Cross-encoder reranking for search results."""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Global cache for cross-encoder models
_model_cache = {}


class CrossEncoderReranker:
    """Cross-encoder model for reranking search results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: Optional[str] = None):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to use (cpu, cuda, mps, or None for auto)
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.model = self._load_model()
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _load_model(self):
        """Load or get cached cross-encoder model."""
        cache_key = f"{self.model_name}_{self.device}"
        
        if cache_key not in _model_cache:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                model = CrossEncoder(self.model_name, device=self.device)
                _model_cache[cache_key] = model
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                return None
        
        return _model_cache[cache_key]
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Rerank candidates based on relevance to query.
        
        Args:
            query: Query string
            candidates: List of (doc_id, text) tuples
            top_k: Return only top-k results
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        if not self.model or not candidates:
            return []
        
        # Prepare input pairs
        pairs = [[query, text] for _, text in candidates]
        
        try:
            # Get scores from model
            scores = self.model.predict(pairs, batch_size=32)
            
            # Combine with doc IDs
            results = [(candidates[i][0], float(scores[i])) for i in range(len(scores))]
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k if specified
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order as fallback
            return [(doc_id, 0.5) for doc_id, _ in candidates]
    
    def rerank_with_payloads(
        self,
        query: str,
        results: List[Tuple[str, float, Dict]],
        text_extractor=None,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Rerank results with payloads.
        
        Args:
            query: Query string
            results: List of (doc_id, score, payload) tuples
            text_extractor: Function to extract text from payload
            top_k: Return only top-k results
            
        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []
        
        # Default text extractor
        if text_extractor is None:
            text_extractor = self._default_text_extractor
        
        # Extract candidates
        candidates = [(doc_id, text_extractor(payload)) for doc_id, _, payload in results]
        
        # Rerank
        reranked_scores = self.rerank(query, candidates, top_k)
        
        # Create score mapping
        score_map = {doc_id: score for doc_id, score in reranked_scores}
        
        # Update results with new scores
        reranked_results = []
        for doc_id, _, payload in results:
            if doc_id in score_map:
                reranked_results.append((doc_id, score_map[doc_id], payload))
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k] if top_k else reranked_results
    
    @staticmethod
    def _default_text_extractor(payload: Dict) -> str:
        """Default text extraction from payload."""
        # Try common field names
        for field in ['text', 'content', 'description', 'review', 'title']:
            if field in payload:
                return str(payload[field])
        
        # Combine title and description if available
        parts = []
        if 'title' in payload:
            parts.append(str(payload['title']))
        if 'description' in payload:
            parts.append(str(payload['description']))
        if 'category' in payload:
            parts.append(str(payload['category']))
        
        return ' '.join(parts) if parts else str(payload)


def get_cross_encoder(model_name: str, device: Optional[str] = None):
    """Get or create a cross-encoder model.
    
    Args:
        model_name: Model name
        device: Device to use
        
    Returns:
        CrossEncoder model or None
    """
    cache_key = f"{model_name}_{device or 'auto'}"
    
    if cache_key not in _model_cache:
        try:
            from sentence_transformers import CrossEncoder
            
            # Detect device if not specified
            if device is None:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            model = CrossEncoder(model_name, device=device)
            _model_cache[cache_key] = model
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            return None
    
    return _model_cache[cache_key]