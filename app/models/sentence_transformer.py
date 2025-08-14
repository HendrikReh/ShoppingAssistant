"""Sentence Transformer model management."""

import logging
import os
from typing import Optional, List
import numpy as np

from .cache import ModelCache

logger = logging.getLogger(__name__)


class SentenceTransformerManager:
    """Manage sentence transformer models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, cache: Optional[ModelCache] = None):
        """Initialize sentence transformer manager.
        
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
        cache_key = f"st_{self.model_name}_{self.device}"
        
        # Check cache first
        self.model = self.cache.get(cache_key)
        
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Disable symlinks warning
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
                
                logger.info(f"Loading sentence transformer: {self.model_name} on {self.device}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # Cache the model
                self.cache.set(cache_key, self.model)
                
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                logger.info("Will attempt to work without the model")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            logger.error("Model not loaded")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                device=self.device,
                show_progress_bar=show_progress,
                convert_to_numpy=convert_to_numpy
            )
            return embeddings
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return np.array([])
    
    def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        """Encode a single text.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize
            
        Returns:
            Embedding vector as list
        """
        embeddings = self.encode([text], batch_size=1, normalize=normalize)
        if len(embeddings) > 0:
            return embeddings[0].tolist()
        return []
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0
    
    def warmup(self):
        """Warm up the model with a test encoding."""
        if self.model:
            try:
                _ = self.encode(["test"], batch_size=1)
                logger.info(f"Model {self.model_name} warmed up successfully")
            except Exception as e:
                logger.error(f"Model warmup failed: {e}")


def load_sentence_transformer(
    model_name: str,
    device: Optional[str] = None,
    use_cache: bool = True
) -> Optional[SentenceTransformerManager]:
    """Load a sentence transformer model.
    
    Args:
        model_name: Model name
        device: Device to use
        use_cache: Whether to use model cache
        
    Returns:
        SentenceTransformerManager instance or None
    """
    try:
        cache = ModelCache() if use_cache else None
        manager = SentenceTransformerManager(model_name, device, cache)
        if manager.model:
            return manager
        return None
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        return None


def get_device() -> str:
    """Get the best available device.
    
    Returns:
        Device string (cpu, cuda, or mps)
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"