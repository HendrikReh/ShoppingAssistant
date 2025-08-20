"""Initialize product pricing and stock databases with realistic data."""

import json
import random
import logging
from pathlib import Path
from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .price_agent import PriceAgent
from .stock_agent import StockAgent
from .schemas import (
    ProductPrice,
    StockLevel,
    Currency,
    WarehouseLocation,
    TransactionType,
    StockUpdate
)

# Import for web price fetching
import sys
sys.path.append(str(Path(__file__).parent.parent))
from web_search_agent import TavilyWebSearchAgent, WebSearchConfig

logger = logging.getLogger(__name__)

# Realistic price ranges by category
PRICE_RANGES = {
    "Amazon Devices": {
        "Fire TV": (29.99, 179.99),
        "Echo": (24.99, 249.99),
        "Kindle": (79.99, 349.99),
        "Ring": (29.99, 249.99),
        "default": (19.99, 199.99)
    },
    "Electronics": {
        "Headphones": (19.99, 399.99),
        "Speakers": (29.99, 499.99),
        "Chargers": (9.99, 79.99),
        "Cables": (4.99, 49.99),
        "Cases": (9.99, 59.99),
        "default": (14.99, 299.99)
    },
    "Computers": {
        "Laptops": (299.99, 2999.99),
        "Monitors": (99.99, 1499.99),
        "Keyboards": (19.99, 249.99),
        "Mice": (9.99, 149.99),
        "default": (29.99, 999.99)
    },
    "default": (9.99, 199.99)
}

# Known products with researched prices
KNOWN_PRICES = {
    "Fire TV Stick 4K": 49.99,
    "Fire TV Stick 4K Max": 59.99,
    "Echo Dot (3rd Gen)": 39.99,
    "Echo Dot (5th Gen)": 49.99,
    "Fire TV Stick": 39.99,
    "Echo Show": 84.99,
    "Kindle Paperwhite": 139.99,
    "Ring Video Doorbell": 99.99
}


class DatabaseInitializer:
    """Initialize and populate price and stock databases."""
    
    def __init__(
        self,
        products_path: Path,
        use_web_search: bool = False,
        web_api_key: Optional[str] = None
    ):
        """Initialize the database initializer.
        
        Args:
            products_path: Path to products JSONL file
            use_web_search: Whether to fetch real prices from web
            web_api_key: Tavily API key for web search
        """
        self.products_path = products_path
        self.use_web_search = use_web_search
        self.web_api_key = web_api_key
        
        # Initialize agents
        self.price_agent = PriceAgent()
        self.stock_agent = StockAgent()
        
        # Initialize web search if enabled
        self.web_agent = None
        if use_web_search and web_api_key:
            try:
                import os
                os.environ['TAVILY_API_KEY'] = web_api_key
                config = WebSearchConfig(
                    api_key=web_api_key,
                    enable_web_search=True,
                    max_results=5
                )
                self.web_agent = TavilyWebSearchAgent(config)
                logger.info("Web search agent initialized for price research")
            except Exception as e:
                logger.warning(f"Failed to initialize web search: {e}")
                self.web_agent = None
    
    def load_products(self) -> List[Dict[str, Any]]:
        """Load products from JSONL file.
        
        Returns:
            List of product dictionaries
        """
        products = []
        with open(self.products_path, 'r') as f:
            for line in f:
                products.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(products)} products from {self.products_path}")
        return products
    
    def get_product_category(self, product: Dict[str, Any]) -> str:
        """Extract product category from product data.
        
        Args:
            product: Product dictionary
            
        Returns:
            Product category
        """
        # Try main_category first
        if 'main_category' in product:
            return product['main_category']
        
        # Try categories list
        if 'categories' in product and product['categories']:
            return product['categories'][0]
        
        return "default"
    
    def get_product_subcategory(self, product: Dict[str, Any]) -> str:
        """Extract product subcategory from title.
        
        Args:
            product: Product dictionary
            
        Returns:
            Product subcategory
        """
        title = product.get('title', '').lower()
        
        # Amazon devices
        if 'fire tv' in title:
            return "Fire TV"
        elif 'echo' in title:
            return "Echo"
        elif 'kindle' in title:
            return "Kindle"
        elif 'ring' in title:
            return "Ring"
        
        # Electronics
        elif 'headphone' in title or 'earbuds' in title or 'airpods' in title:
            return "Headphones"
        elif 'speaker' in title:
            return "Speakers"
        elif 'charger' in title:
            return "Chargers"
        elif 'cable' in title or 'cord' in title:
            return "Cables"
        elif 'case' in title or 'cover' in title:
            return "Cases"
        
        # Computers
        elif 'laptop' in title or 'notebook' in title:
            return "Laptops"
        elif 'monitor' in title or 'display' in title:
            return "Monitors"
        elif 'keyboard' in title:
            return "Keyboards"
        elif 'mouse' in title or 'mice' in title:
            return "Mice"
        
        return "default"
    
    def generate_realistic_price(self, product: Dict[str, Any]) -> Decimal:
        """Generate a realistic price for a product.
        
        Args:
            product: Product dictionary
            
        Returns:
            Generated price
        """
        title = product.get('title', '')
        
        # Check if we have a known price
        for known_product, known_price in KNOWN_PRICES.items():
            if known_product.lower() in title.lower():
                # Add small variation
                variation = random.uniform(0.95, 1.05)
                return Decimal(str(round(known_price * variation, 2)))
        
        # Get price range based on category
        category = self.get_product_category(product)
        subcategory = self.get_product_subcategory(product)
        
        if category in PRICE_RANGES:
            if subcategory in PRICE_RANGES[category]:
                min_price, max_price = PRICE_RANGES[category][subcategory]
            else:
                min_price, max_price = PRICE_RANGES[category]["default"]
        else:
            min_price, max_price = PRICE_RANGES["default"]
        
        # Adjust based on rating and reviews
        rating = product.get('average_rating', 3.5)
        review_count = product.get('review_count', 100)
        
        # Higher rating -> slightly higher price
        rating_factor = 0.8 + (rating / 5.0) * 0.4  # 0.8 to 1.2
        
        # More reviews -> competitive pricing (slightly lower)
        if review_count > 10000:
            popularity_factor = 0.95
        elif review_count > 1000:
            popularity_factor = 1.0
        else:
            popularity_factor = 1.05
        
        # Generate base price
        base_price = random.uniform(min_price, max_price)
        adjusted_price = base_price * rating_factor * popularity_factor
        
        # Round to realistic price points
        if adjusted_price < 10:
            adjusted_price = round(adjusted_price, 2)
        elif adjusted_price < 100:
            adjusted_price = round(adjusted_price / 0.99) * 0.99  # .99 pricing
        else:
            adjusted_price = round(adjusted_price / 10) * 10 - 0.01  # 149.99, 199.99, etc.
        
        return Decimal(str(adjusted_price))
    
    def fetch_web_price(self, product: Dict[str, Any]) -> Optional[Decimal]:
        """Fetch real price from web search.
        
        Args:
            product: Product dictionary
            
        Returns:
            Web price or None if not found
        """
        if not self.web_agent:
            return None
        
        try:
            title = product.get('title', '')
            # Simplify title for search
            search_title = ' '.join(title.split()[:10])  # First 10 words
            
            results = self.web_agent.search_product_price(search_title)
            
            if results and 'price' in results:
                # Extract numeric price
                price_str = results['price'].replace('$', '').replace(',', '')
                try:
                    return Decimal(price_str)
                except:
                    pass
        except Exception as e:
            logger.debug(f"Failed to fetch web price for {product.get('title')}: {e}")
        
        return None
    
    def generate_stock_quantity(self, product: Dict[str, Any]) -> int:
        """Generate realistic stock quantity based on product popularity.
        
        Args:
            product: Product dictionary
            
        Returns:
            Stock quantity
        """
        review_count = product.get('review_count', 0)
        rating = product.get('average_rating', 3.5)
        
        # Popular products (many reviews, high rating)
        if review_count > 10000 and rating >= 4.0:
            return random.randint(200, 500)
        elif review_count > 5000:
            return random.randint(100, 300)
        elif review_count > 1000:
            return random.randint(50, 200)
        elif review_count > 500:
            return random.randint(20, 100)
        elif review_count > 100:
            return random.randint(10, 50)
        else:
            # Low popularity or new products
            return random.randint(0, 30)
    
    def initialize_prices(self, products: List[Dict[str, Any]], top_k: int = 100):
        """Initialize product prices.
        
        Args:
            products: List of products
            top_k: Number of top products to fetch web prices for
        """
        logger.info("Initializing product prices...")
        
        # Sort products by popularity (review count)
        sorted_products = sorted(
            products,
            key=lambda p: p.get('review_count', 0),
            reverse=True
        )
        
        price_count = 0
        web_price_count = 0
        
        for i, product in enumerate(sorted_products):
            product_id = product.get('parent_asin', '') or str(i)
            title = product.get('title', f'Product {i}')
            
            # Try to fetch web price for top products
            web_price = None
            if self.use_web_search and i < top_k:
                web_price = self.fetch_web_price(product)
                if web_price:
                    web_price_count += 1
                    logger.debug(f"Fetched web price for {title}: ${web_price}")
            
            # Use web price or generate realistic price
            if web_price:
                current_price = web_price
                base_price = web_price * Decimal("1.1")  # Assume 10% markup from base
            else:
                current_price = self.generate_realistic_price(product)
                base_price = current_price * Decimal("1.05")  # 5% markup
            
            # Create some sale prices
            if random.random() < 0.2:  # 20% chance of sale
                discount = random.uniform(0.1, 0.3)  # 10-30% discount
                current_price = base_price * Decimal(str(1 - discount))
                current_price = current_price.quantize(Decimal("0.01"))
            
            # Create retailer prices
            retailer_prices = {
                "Amazon": current_price,
                "BestBuy": current_price * Decimal("1.02"),
                "Walmart": current_price * Decimal("0.98")
            }
            
            # Create price entry
            product_price = ProductPrice(
                product_id=product_id,
                product_title=title,
                base_price=base_price.quantize(Decimal("0.01")),
                current_price=current_price.quantize(Decimal("0.01")),
                min_price=current_price.quantize(Decimal("0.01")),
                max_price=base_price.quantize(Decimal("0.01")),
                currency=Currency.USD,
                retailer_prices=retailer_prices
            )
            
            try:
                self.price_agent.add_product_price(product_price)
                price_count += 1
            except Exception as e:
                logger.warning(f"Failed to add price for {product_id}: {e}")
        
        logger.info(f"Initialized {price_count} product prices ({web_price_count} from web)")
    
    def initialize_stock(self, products: List[Dict[str, Any]]):
        """Initialize product stock levels.
        
        Args:
            products: List of products
        """
        logger.info("Initializing product stock levels...")
        
        stock_count = 0
        warehouses = list(WarehouseLocation)
        
        for i, product in enumerate(products):
            product_id = product.get('parent_asin', '') or str(i)
            title = product.get('title', f'Product {i}')
            
            # Generate stock for main warehouse
            main_quantity = self.generate_stock_quantity(product)
            
            stock_level = StockLevel(
                product_id=product_id,
                product_title=title,
                quantity_available=main_quantity,
                quantity_reserved=0,
                reorder_level=max(10, main_quantity // 10),  # 10% of stock or 10 minimum
                max_stock_level=main_quantity * 2,  # Can hold double current stock
                warehouse_location=WarehouseLocation.MAIN,
                last_restocked=datetime.now() - timedelta(days=random.randint(1, 30))
            )
            
            try:
                self.stock_agent.add_product_stock(stock_level)
                stock_count += 1
            except Exception as e:
                logger.warning(f"Failed to add stock for {product_id}: {e}")
            
            # Add stock to other warehouses for popular products
            if product.get('review_count', 0) > 5000:
                for warehouse in [WarehouseLocation.EAST, WarehouseLocation.WEST]:
                    secondary_quantity = main_quantity // 3  # 1/3 of main warehouse
                    
                    if secondary_quantity > 0:
                        secondary_stock = StockLevel(
                            product_id=product_id,
                            product_title=title,
                            quantity_available=secondary_quantity,
                            quantity_reserved=0,
                            reorder_level=max(5, secondary_quantity // 10),
                            max_stock_level=secondary_quantity * 2,
                            warehouse_location=warehouse,
                            last_restocked=datetime.now() - timedelta(days=random.randint(5, 45))
                        )
                        
                        try:
                            self.stock_agent.add_product_stock(secondary_stock)
                            stock_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to add secondary stock for {product_id}: {e}")
        
        logger.info(f"Initialized {stock_count} stock entries across warehouses")
    
    def run(self, max_products: Optional[int] = None):
        """Run the complete initialization process.
        
        Args:
            max_products: Maximum number of products to process
        """
        logger.info("Starting database initialization...")
        
        # Load products
        products = self.load_products()
        
        if max_products:
            products = products[:max_products]
            logger.info(f"Processing first {max_products} products")
        
        # Initialize prices
        self.initialize_prices(products, top_k=min(100, len(products)))
        
        # Initialize stock
        self.initialize_stock(products)
        
        logger.info("Database initialization complete!")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print initialization summary."""
        # Get some stats
        low_stock = self.stock_agent.get_low_stock_products()
        out_of_stock = self.stock_agent.get_out_of_stock_products()
        
        print("\n" + "="*50)
        print("DATABASE INITIALIZATION SUMMARY")
        print("="*50)
        print(f"Price database: {self.price_agent.db_path}")
        print(f"Stock database: {self.stock_agent.db_path}")
        print(f"\nStock Status:")
        print(f"  - Products with low stock: {len(low_stock)}")
        print(f"  - Products out of stock: {len(out_of_stock)}")
        print("="*50)


def main():
    """Main entry point for database initialization."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Initialize pricing and stock databases")
    parser.add_argument(
        "--products-path",
        type=Path,
        default=Path("data/top_1000_products.jsonl"),
        help="Path to products JSONL file"
    )
    parser.add_argument(
        "--max-products",
        type=int,
        help="Maximum number of products to process"
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Fetch real prices from web (requires TAVILY_API_KEY)"
    )
    parser.add_argument(
        "--web-api-key",
        help="Tavily API key for web search"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    web_api_key = args.web_api_key or os.environ.get('TAVILY_API_KEY')
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize databases
    initializer = DatabaseInitializer(
        products_path=args.products_path,
        use_web_search=args.use_web_search,
        web_api_key=web_api_key
    )
    
    initializer.run(max_products=args.max_products)


if __name__ == "__main__":
    main()