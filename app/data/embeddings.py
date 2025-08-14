"""Embedding generation utilities."""

import logging
from typing import List, Optional, Iterable
import numpy as np

logger = logging.getLogger(__name__)

# Global cache for embedding models
_model_cache = {}


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
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
        """Load or get cached sentence transformer model."""
        cache_key = f"{self.model_name}_{self.device}"
        
        if cache_key not in _model_cache:
            try:
                from sentence_transformers import SentenceTransformer
                import os
                
                # Disable symlinks warning
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
                
                logger.info(f"Loading embedding model: {self.model_name}")
                model = SentenceTransformer(self.model_name, device=self.device)
                _model_cache[cache_key] = model
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                logger.info("Attempting to work in offline mode or with cached models")
                return None
        
        return _model_cache[cache_key]
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            logger.error("Model not loaded, returning empty embeddings")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                device=self.device,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])
    
    def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize embedding
            
        Returns:
            Embedding vector as list
        """
        embeddings = self.encode([text], batch_size=1, normalize=normalize)
        if len(embeddings) > 0:
            return embeddings[0].tolist()
        return []


def embed_texts(
    model,
    texts: List[str],
    device: str,
    batch_size: int = 32
) -> List[List[float]]:
    """Generate embeddings for texts using provided model.
    
    Args:
        model: Sentence transformer model
        texts: Texts to embed
        device: Device to use
        batch_size: Batch size
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            device=device,
            convert_to_numpy=True
        )
        embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def chunked(items: Iterable, n: int) -> Iterable[list]:
    """Yield successive n-sized chunks from items.
    
    Args:
        items: Iterable to chunk
        n: Chunk size
        
    Yields:
        Chunks of size n
    """
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def load_embedding_model(model_name: str, device: Optional[str] = None):
    """Load or get cached sentence transformer model.
    
    Args:
        model_name: Model name
        device: Device to use
        
    Returns:
        SentenceTransformer model or None
    """
    cache_key = f"{model_name}_{device or 'auto'}"
    
    if cache_key not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            # Detect device if not specified
            if device is None:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
            model = SentenceTransformer(model_name, device=device)
            _model_cache[cache_key] = model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    
    return _model_cache[cache_key]