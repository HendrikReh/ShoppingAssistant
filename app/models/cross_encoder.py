"""Cross-encoder model management."""

import logging
from typing import Optional, List, Tuple
import numpy as np

from .cache import ModelCache

logger = logging.getLogger(__name__)


class CrossEncoderManager:
    """Manage cross-encoder models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, cache: Optional[ModelCache] = None):
        """Initialize cross-encoder manager.
        
        Args:
            model_name: Name of the model
            device: Device to use (cpu, cuda, mps, or None for auto)
            cache: Model cache instance
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.cache = cache or ModelCache()
        self.model = None
        self._load_model()
    
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
        """Load the model."""
        cache_key = f"ce_{self.model_name}_{self.device}"
        
        # Check cache first
        self.model = self.cache.get(cache_key)
        
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
                self.model = CrossEncoder(self.model_name, device=self.device)
                
                # Cache the model
                self.cache.set(cache_key, self.model)
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                logger.info("Will attempt to work without the model")
    
    def predict(
        self,
        sentence_pairs: List[List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Predict relevance scores for sentence pairs.
        
        Args:
            sentence_pairs: List of [query, document] pairs
            batch_size: Batch size for prediction
            show_progress: Whether to show progress bar
            
        Returns:
            Array of scores
        """
        if not self.model:
            logger.error("Model not loaded")
            return np.array([])
        
        try:
            scores = self.model.predict(
                sentence_pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            return scores
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.array([])
    
    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rank documents by relevance to query.
        
        Args:
            query: Query string
            documents: List of documents
            top_k: Return only top-k documents
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        if not documents:
            return []
        
        # Create sentence pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.predict(pairs)
        
        if len(scores) == 0:
            return []
        
        # Create index-score pairs
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]
        
        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_results(
        self,
        query: str,
        results: List[Tuple[str, float, str]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, str]]:
        """Rerank search results.
        
        Args:
            query: Query string
            results: List of (id, score, text) tuples
            top_k: Return only top-k results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Extract texts
        texts = [text for _, _, text in results]
        
        # Rank texts
        ranked_indices = self.rank(query, texts, top_k)
        
        # Reorder results
        reranked = []
        for idx, score in ranked_indices:
            original_id = results[idx][0]
            original_text = results[idx][2]
            reranked.append((original_id, score, original_text))
        
        return reranked
    
    def warmup(self):
        """Warm up the model with a test prediction."""
        if self.model:
            try:
                _ = self.predict([["test query", "test document"]])
                logger.info(f"Model {self.model_name} warmed up successfully")
            except Exception as e:
                logger.error(f"Model warmup failed: {e}")


def load_cross_encoder(
    model_name: str,
    device: Optional[str] = None,
    use_cache: bool = True
) -> Optional[CrossEncoderManager]:
    """Load a cross-encoder model.
    
    Args:
        model_name: Model name
        device: Device to use
        use_cache: Whether to use model cache
        
    Returns:
        CrossEncoderManager instance or None
    """
    try:
        cache = ModelCache() if use_cache else None
        manager = CrossEncoderManager(model_name, device, cache)
        if manager.model:
            return manager
        return None
    except Exception as e:
        logger.error(f"Failed to load cross-encoder: {e}")
        return None