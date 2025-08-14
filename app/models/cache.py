"""Model caching utilities."""

import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# Global model cache
_global_cache: Dict[str, Any] = {}


class ModelCache:
    """Cache for ML models to avoid repeated loading."""
    
    def __init__(self, use_global: bool = True):
        """Initialize model cache.
        
        Args:
            use_global: Whether to use global cache
        """
        self.use_global = use_global
        if not use_global:
            self.local_cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached model or None
        """
        if self.use_global:
            model = _global_cache.get(key)
        else:
            model = self.local_cache.get(key)
        
        if model:
            logger.debug(f"Cache hit for key: {key}")
        else:
            logger.debug(f"Cache miss for key: {key}")
        
        return model
    
    def set(self, key: str, model: Any) -> None:
        """Set model in cache.
        
        Args:
            key: Cache key
            model: Model to cache
        """
        if self.use_global:
            _global_cache[key] = model
        else:
            self.local_cache[key] = model
        
        logger.debug(f"Cached model with key: {key}")
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if self.use_global:
            return key in _global_cache
        else:
            return key in self.local_cache
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            if self.use_global:
                _global_cache.pop(key, None)
            else:
                self.local_cache.pop(key, None)
            logger.debug(f"Cleared cache key: {key}")
        else:
            if self.use_global:
                _global_cache.clear()
            else:
                self.local_cache.clear()
            logger.debug("Cleared all cache")
    
    def keys(self) -> list:
        """Get all cache keys.
        
        Returns:
            List of cache keys
        """
        if self.use_global:
            return list(_global_cache.keys())
        else:
            return list(self.local_cache.keys())
    
    def size(self) -> int:
        """Get cache size.
        
        Returns:
            Number of cached items
        """
        if self.use_global:
            return len(_global_cache)
        else:
            return len(self.local_cache)


def get_cached_model(key: str) -> Optional[Any]:
    """Get model from global cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached model or None
    """
    return _global_cache.get(key)


def cache_model(key: str, model: Any) -> None:
    """Cache model in global cache.
    
    Args:
        key: Cache key
        model: Model to cache
    """
    _global_cache[key] = model
    logger.debug(f"Cached model globally with key: {key}")


def clear_model_cache(key: Optional[str] = None) -> None:
    """Clear global model cache.
    
    Args:
        key: Specific key to clear, or None to clear all
    """
    if key:
        _global_cache.pop(key, None)
        logger.info(f"Cleared model cache key: {key}")
    else:
        _global_cache.clear()
        logger.info("Cleared all model cache")