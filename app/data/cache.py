"""Redis caching utilities."""

import json
import hashlib
import logging
from typing import Optional, Any, Dict
from datetime import timedelta

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache for search results and other data."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 7200,
        prefix: str = "shopping:"
    ):
        """Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            default_ttl: Default TTL in seconds
            prefix: Key prefix for all cache entries
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.client = None
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            logger.info("Cache will be disabled")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key.
        
        Args:
            key: Base key
            
        Returns:
            Prefixed key
        """
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            return None
        
        try:
            full_key = self._make_key(key)
            value = self.client.get(full_key)
            
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache miss: {key}")
                return None
                
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            ttl = ttl or self.default_ttl
            
            self.client.setex(
                full_key,
                timedelta(seconds=ttl),
                serialized
            )
            logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            return True
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if not self.client:
            return False
        
        try:
            full_key = self._make_key(key)
            deleted = self.client.delete(full_key) > 0
            if deleted:
                logger.debug(f"Deleted from cache: {key}")
            return deleted
            
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists
        """
        if not self.client:
            return False
        
        try:
            full_key = self._make_key(key)
            return self.client.exists(full_key) > 0
            
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., "search:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching: {pattern}")
                return deleted
            return 0
            
        except RedisError as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if not exists
        """
        if not self.client:
            return -2
        
        try:
            full_key = self._make_key(key)
            return self.client.ttl(full_key)
            
        except RedisError as e:
            logger.error(f"Cache TTL error: {e}")
            return -2
    
    @staticmethod
    def hash_key(value: str) -> str:
        """Generate hash key from string.
        
        Args:
            value: String to hash
            
        Returns:
            Hash string
        """
        return hashlib.md5(value.encode()).hexdigest()


def create_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    **kwargs
) -> Optional[RedisCache]:
    """Create Redis cache instance.
    
    Args:
        host: Redis host
        port: Redis port
        **kwargs: Additional parameters
        
    Returns:
        RedisCache instance or None if connection fails
    """
    try:
        cache = RedisCache(host, port, **kwargs)
        if cache.client:
            return cache
        return None
    except Exception as e:
        logger.error(f"Failed to create Redis cache: {e}")
        return None