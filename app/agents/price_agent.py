"""Price Agent for managing product pricing and price tracking."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
from contextlib import contextmanager

import redis
from redis.exceptions import RedisError

from .schemas import (
    ProductPrice,
    PriceHistory,
    PriceUpdate,
    PriceSearchResult,
    Currency
)

logger = logging.getLogger(__name__)


class PriceAgent:
    """Agent for managing product pricing and price history."""
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 7200  # 2 hours
    ):
        """Initialize the Price Agent.
        
        Args:
            db_path: Path to SQLite database file
            redis_client: Redis client for caching
            cache_ttl: Cache TTL in seconds
        """
        self.db_path = db_path or Path("app/databases/pricing.db")
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Try to connect to Redis if client provided
        if self.redis_client:
            try:
                self.redis_client.ping()
                logger.info("Connected to Redis for price caching")
            except RedisError:
                logger.warning("Redis connection failed, continuing without cache")
                self.redis_client = None
    
    def _init_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create products_pricing table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products_pricing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT UNIQUE NOT NULL,
                    product_title TEXT NOT NULL,
                    base_price DECIMAL(10,2) NOT NULL,
                    current_price DECIMAL(10,2) NOT NULL,
                    min_price DECIMAL(10,2),
                    max_price DECIMAL(10,2),
                    currency TEXT DEFAULT 'USD',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    price_history JSON,
                    retailer_prices JSON
                )
            """)
            
            # Create price_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    retailer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products_pricing(product_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_id ON products_pricing(product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_product ON price_history(product_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)")
            
            conn.commit()
            logger.info(f"Price database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _get_cache_key(self, product_id: str) -> str:
        """Generate cache key for a product."""
        return f"price:{product_id}"
    
    def _cache_price(self, product_price: ProductPrice):
        """Cache price data in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(product_price.product_id)
            price_data = product_price.model_dump_json()
            self.redis_client.setex(cache_key, self.cache_ttl, price_data)
        except RedisError as e:
            logger.warning(f"Failed to cache price: {e}")
    
    def _get_cached_price(self, product_id: str) -> Optional[ProductPrice]:
        """Get cached price from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(product_id)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return ProductPrice.model_validate_json(cached_data)
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get cached price: {e}")
        
        return None
    
    def add_product_price(self, product_price: ProductPrice) -> ProductPrice:
        """Add a new product price entry.
        
        Args:
            product_price: Product price data
            
        Returns:
            Created product price with ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert retailer_prices to JSON
            retailer_prices_json = json.dumps(
                {k: str(v) for k, v in product_price.retailer_prices.items()}
            ) if product_price.retailer_prices else None
            
            # Convert price_history to JSON
            price_history_json = json.dumps(product_price.price_history) if product_price.price_history else None
            
            cursor.execute("""
                INSERT INTO products_pricing (
                    product_id, product_title, base_price, current_price,
                    min_price, max_price, currency, price_history, retailer_prices
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product_price.product_id,
                product_price.product_title,
                str(product_price.base_price),
                str(product_price.current_price),
                str(product_price.min_price) if product_price.min_price else None,
                str(product_price.max_price) if product_price.max_price else None,
                product_price.currency.value,
                price_history_json,
                retailer_prices_json
            ))
            
            conn.commit()
            product_price.id = cursor.lastrowid
            
            # Cache the price
            self._cache_price(product_price)
            
            # Add to price history
            self._add_price_history(
                product_price.product_id,
                product_price.current_price,
                retailer="system"
            )
            
            logger.info(f"Added price for product {product_price.product_id}")
            return product_price
    
    def get_product_price(self, product_id: str) -> Optional[ProductPrice]:
        """Get current price for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product price data or None if not found
        """
        # Check cache first
        cached_price = self._get_cached_price(product_id)
        if cached_price:
            return cached_price
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM products_pricing WHERE product_id = ?
            """, (product_id,))
            
            row = cursor.fetchone()
            if row:
                # Parse JSON fields
                price_data = dict(row)
                price_data['base_price'] = Decimal(price_data['base_price']).quantize(Decimal("0.01"))
                price_data['current_price'] = Decimal(price_data['current_price']).quantize(Decimal("0.01"))
                
                if price_data['min_price']:
                    price_data['min_price'] = Decimal(price_data['min_price']).quantize(Decimal("0.01"))
                if price_data['max_price']:
                    price_data['max_price'] = Decimal(price_data['max_price']).quantize(Decimal("0.01"))
                
                if price_data['retailer_prices']:
                    retailer_prices = json.loads(price_data['retailer_prices'])
                    price_data['retailer_prices'] = {
                        k: Decimal(v).quantize(Decimal("0.01")) for k, v in retailer_prices.items()
                    }
                
                if price_data['price_history']:
                    price_data['price_history'] = json.loads(price_data['price_history'])
                
                product_price = ProductPrice(**price_data)
                
                # Cache the result
                self._cache_price(product_price)
                
                return product_price
        
        return None
    
    def update_product_price(self, price_update: PriceUpdate) -> Optional[ProductPrice]:
        """Update product price.
        
        Args:
            price_update: Price update request
            
        Returns:
            Updated product price or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current price data
            current_price = self.get_product_price(price_update.product_id)
            if not current_price:
                return None
            
            # Update min/max if necessary
            new_min = min(current_price.min_price or price_update.new_price, price_update.new_price)
            new_max = max(current_price.max_price or price_update.new_price, price_update.new_price)
            
            # Update retailer prices if retailer specified
            if price_update.retailer and current_price.retailer_prices:
                current_price.retailer_prices[price_update.retailer] = price_update.new_price
                retailer_prices_json = json.dumps(
                    {k: str(v) for k, v in current_price.retailer_prices.items()}
                )
            else:
                retailer_prices_json = None
            
            cursor.execute("""
                UPDATE products_pricing
                SET current_price = ?, min_price = ?, max_price = ?,
                    retailer_prices = ?, last_updated = CURRENT_TIMESTAMP
                WHERE product_id = ?
            """, (
                str(price_update.new_price),
                str(new_min),
                str(new_max),
                retailer_prices_json,
                price_update.product_id
            ))
            
            conn.commit()
            
            # Add to price history
            self._add_price_history(
                price_update.product_id,
                price_update.new_price,
                price_update.retailer
            )
            
            # Invalidate cache
            if self.redis_client:
                try:
                    cache_key = self._get_cache_key(price_update.product_id)
                    self.redis_client.delete(cache_key)
                except RedisError:
                    pass
            
            logger.info(f"Updated price for product {price_update.product_id} to {price_update.new_price}")
            return self.get_product_price(price_update.product_id)
    
    def _add_price_history(self, product_id: str, price: Decimal, retailer: Optional[str] = None):
        """Add price history entry.
        
        Args:
            product_id: Product ID
            price: Price value
            retailer: Retailer name
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO price_history (product_id, price, retailer)
                VALUES (?, ?, ?)
            """, (product_id, str(price), retailer))
            conn.commit()
    
    def get_price_history(
        self,
        product_id: str,
        days: int = 30,
        limit: Optional[int] = None
    ) -> List[PriceHistory]:
        """Get price history for a product.
        
        Args:
            product_id: Product ID
            days: Number of days of history
            limit: Maximum number of records
            
        Returns:
            List of price history records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT * FROM price_history
                WHERE product_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (product_id, since_date))
            
            history = []
            for row in cursor.fetchall():
                history_data = dict(row)
                history_data['price'] = Decimal(history_data['price'])
                history.append(PriceHistory(**history_data))
            
            return history
    
    def get_products_by_price_range(
        self,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        limit: int = 50
    ) -> List[ProductPrice]:
        """Get products within a price range.
        
        Args:
            min_price: Minimum price
            max_price: Maximum price
            limit: Maximum number of results
            
        Returns:
            List of products in price range
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM products_pricing WHERE 1=1"
            params = []
            
            if min_price is not None:
                query += " AND current_price >= ?"
                params.append(str(min_price))
            
            if max_price is not None:
                query += " AND current_price <= ?"
                params.append(str(max_price))
            
            query += " ORDER BY current_price ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            products = []
            for row in cursor.fetchall():
                price_data = dict(row)
                price_data['base_price'] = Decimal(price_data['base_price']).quantize(Decimal("0.01"))
                price_data['current_price'] = Decimal(price_data['current_price']).quantize(Decimal("0.01"))
                
                if price_data['min_price']:
                    price_data['min_price'] = Decimal(price_data['min_price']).quantize(Decimal("0.01"))
                if price_data['max_price']:
                    price_data['max_price'] = Decimal(price_data['max_price']).quantize(Decimal("0.01"))
                
                if price_data['retailer_prices']:
                    retailer_prices = json.loads(price_data['retailer_prices'])
                    price_data['retailer_prices'] = {
                        k: Decimal(v).quantize(Decimal("0.01")) for k, v in retailer_prices.items()
                    }
                
                if price_data['price_history']:
                    price_data['price_history'] = json.loads(price_data['price_history'])
                
                products.append(ProductPrice(**price_data))
            
            return products
    
    def get_products_on_sale(self, min_discount: float = 10.0) -> List[ProductPrice]:
        """Get products currently on sale.
        
        Args:
            min_discount: Minimum discount percentage
            
        Returns:
            List of products on sale
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM products_pricing
                WHERE current_price < base_price
                AND ((base_price - current_price) / base_price * 100) >= ?
                ORDER BY ((base_price - current_price) / base_price * 100) DESC
            """, (min_discount,))
            
            products = []
            for row in cursor.fetchall():
                price_data = dict(row)
                price_data['base_price'] = Decimal(price_data['base_price']).quantize(Decimal("0.01"))
                price_data['current_price'] = Decimal(price_data['current_price']).quantize(Decimal("0.01"))
                
                if price_data['min_price']:
                    price_data['min_price'] = Decimal(price_data['min_price']).quantize(Decimal("0.01"))
                if price_data['max_price']:
                    price_data['max_price'] = Decimal(price_data['max_price']).quantize(Decimal("0.01"))
                
                if price_data['retailer_prices']:
                    retailer_prices = json.loads(price_data['retailer_prices'])
                    price_data['retailer_prices'] = {
                        k: Decimal(v).quantize(Decimal("0.01")) for k, v in retailer_prices.items()
                    }
                
                if price_data['price_history']:
                    price_data['price_history'] = json.loads(price_data['price_history'])
                
                products.append(ProductPrice(**price_data))
            
            return products
    
    def bulk_update_prices(self, price_updates: List[PriceUpdate]) -> int:
        """Bulk update product prices.
        
        Args:
            price_updates: List of price updates
            
        Returns:
            Number of products updated
        """
        updated_count = 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for update in price_updates:
                cursor.execute("""
                    UPDATE products_pricing
                    SET current_price = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE product_id = ?
                """, (str(update.new_price), update.product_id))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    
                    # Add to price history
                    cursor.execute("""
                        INSERT INTO price_history (product_id, price, retailer)
                        VALUES (?, ?, ?)
                    """, (update.product_id, str(update.new_price), update.retailer))
            
            conn.commit()
            
            # Invalidate cache for all updated products
            if self.redis_client:
                try:
                    for update in price_updates:
                        cache_key = self._get_cache_key(update.product_id)
                        self.redis_client.delete(cache_key)
                except RedisError:
                    pass
        
        logger.info(f"Bulk updated {updated_count} product prices")
        return updated_count