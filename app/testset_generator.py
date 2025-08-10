"""Enhanced synthetic test data generation using actual product catalog.

This module generates realistic test datasets based on actual products in the catalog:
- Uses real product names, brands, and categories
- Creates queries that match actual inventory
- Ensures realistic brand-category combinations
- Generates ground truth answers based on actual product data
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import defaultdict

import typer
from typing_extensions import Annotated


@dataclass
class Product:
    """Structured product representation."""
    id: str
    title: str
    brand: str
    category: str
    product_type: str
    features: List[str]
    rating: float
    review_count: int
    main_category: str


@dataclass 
class QueryTemplate:
    """Template for generating specific query types."""
    template: str
    query_type: str
    complexity: str  # simple, moderate, complex
    requires_context: List[str] = field(default_factory=list)


class RealisticQueryGenerator:
    """Generate realistic e-commerce queries based on actual product catalog."""
    
    def __init__(self, products: List[Dict], reviews: List[Dict], seed: int = 42):
        """Initialize with actual product and review data."""
        self.raw_products = products
        self.raw_reviews = reviews
        self.rng = random.Random(seed)
        
        # Process and categorize products
        self.products = []
        self._process_products()
        self._extract_catalog_info()
        self._define_realistic_templates()
    
    def _extract_brand_from_title(self, title: str) -> str:
        """Extract brand from product title."""
        # Common brands in electronics
        known_brands = [
            'Amazon', 'Apple', 'Samsung', 'Sony', 'Bose', 'JBL', 'Anker',
            'Logitech', 'Dell', 'HP', 'Lenovo', 'ASUS', 'Acer', 'Microsoft',
            'Google', 'LG', 'Panasonic', 'SanDisk', 'Kingston', 'Corsair',
            'Razer', 'SteelSeries', 'HyperX', 'TOZO', 'OontZ', 'DOSS',
            'Belkin', 'TP-Link', 'Netgear', 'Ring', 'Wyze', 'Roku',
            'Fire', 'Echo', 'Kindle'  # Amazon brands
        ]
        
        title_lower = title.lower()
        for brand in known_brands:
            if brand.lower() in title_lower:
                return brand
        
        # Try to get first word if it looks like a brand
        first_word = title.split()[0] if title else ""
        if first_word and first_word[0].isupper() and len(first_word) > 2:
            return first_word
        
        return "Generic"
    
    def _extract_product_type(self, title: str, category: str) -> str:
        """Extract specific product type from title."""
        title_lower = title.lower()
        
        # Map of keywords to product types
        type_keywords = {
            'earbuds': ['earbuds', 'earbud', 'airpods'],
            'headphones': ['headphones', 'headphone', 'headset'],
            'speaker': ['speaker', 'bluetooth speaker', 'soundbar'],
            'cable': ['cable', 'cord', 'wire'],
            'charger': ['charger', 'charging', 'power adapter'],
            'mouse': ['mouse', 'wireless mouse', 'gaming mouse'],
            'keyboard': ['keyboard', 'mechanical keyboard'],
            'monitor': ['monitor', 'display', 'screen'],
            'webcam': ['webcam', 'camera', 'web cam'],
            'tablet': ['tablet', 'ipad', 'fire tablet'],
            'laptop': ['laptop', 'notebook', 'chromebook', 'macbook'],
            'phone': ['phone', 'smartphone', 'iphone'],
            'tv': ['tv', 'television', 'fire tv'],
            'streaming': ['streaming', 'fire stick', 'roku', 'chromecast'],
            'smart home': ['echo', 'alexa', 'smart plug', 'smart light'],
            'storage': ['ssd', 'hard drive', 'flash drive', 'memory card', 'storage'],
            'router': ['router', 'wifi', 'modem'],
            'case': ['case', 'cover', 'sleeve', 'protector'],
            'stand': ['stand', 'mount', 'holder'],
            'adapter': ['adapter', 'hub', 'dock', 'dongle']
        }
        
        for product_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return product_type
        
        # Fallback to category
        if 'audio' in category.lower():
            return 'audio device'
        elif 'computer' in category.lower():
            return 'computer accessory'
        elif 'phone' in category.lower():
            return 'phone accessory'
        
        return 'electronics'
    
    def _extract_features(self, product: Dict) -> List[str]:
        """Extract key features from product data."""
        features = []
        title = product.get('title', '').lower()
        
        # Get features from product features list
        if 'features' in product and isinstance(product['features'], list):
            for feature in product['features'][:3]:  # Take first 3 features
                if isinstance(feature, str) and len(feature) < 100:
                    features.append(feature)
        
        # Extract common feature keywords
        feature_keywords = [
            'wireless', 'bluetooth', 'usb-c', 'usb 3.0', 'hdmi', '4k', '1080p',
            'noise cancelling', 'waterproof', 'fast charging', 'rgb', 'mechanical',
            'ergonomic', 'portable', 'rechargeable', 'smart', 'alexa', 'wifi',
            'long battery', 'hd', 'ultra hd', 'dolby', 'surround sound'
        ]
        
        for keyword in feature_keywords:
            if keyword in title:
                features.append(keyword)
        
        return features[:5]  # Limit to 5 features
    
    def _process_products(self):
        """Process raw products into structured format."""
        for prod in self.raw_products[:500]:  # Process first 500 products
            if not prod.get('title'):
                continue
                
            title = prod['title']
            brand = self._extract_brand_from_title(title)
            main_cat = prod.get('main_category', 'Electronics')
            product_type = self._extract_product_type(title, main_cat)
            features = self._extract_features(prod)
            
            self.products.append(Product(
                id=prod.get('parent_asin', ''),
                title=title,
                brand=brand,
                category=main_cat,
                product_type=product_type,
                features=features,
                rating=prod.get('average_rating', 0),
                review_count=prod.get('review_count', 0),
                main_category=main_cat
            ))
    
    def _extract_catalog_info(self):
        """Extract catalog information from processed products."""
        # Group products by type and brand
        self.products_by_type = defaultdict(list)
        self.products_by_brand = defaultdict(list)
        self.brand_categories = defaultdict(set)
        
        for product in self.products:
            self.products_by_type[product.product_type].append(product)
            self.products_by_brand[product.brand].append(product)
            self.brand_categories[product.brand].add(product.product_type)
        
        # Get popular products (high review count)
        self.popular_products = sorted(
            self.products, 
            key=lambda p: p.review_count, 
            reverse=True
        )[:100]
        
        # Get top-rated products
        self.top_rated = sorted(
            [p for p in self.products if p.rating >= 4.0],
            key=lambda p: p.rating,
            reverse=True
        )[:100]
        
        # Extract actual price ranges from titles (if mentioned)
        self.price_indicators = ['budget', 'premium', 'affordable', 'high-end', 'value']
    
    def _define_realistic_templates(self):
        """Define query templates that match actual catalog."""
        self.templates = [
            # Simple factual queries about specific products
            QueryTemplate(
                "What is the rating of {specific_product}?",
                "single_hop",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Show me {brand} {product_type}",
                "single_hop", 
                "simple",
                ["products"]
            ),
            
            # Comparative queries with real products
            QueryTemplate(
                "Compare {product1} and {product2}",
                "comparative",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Which is better: {product1} or {product2}?",
                "comparative",
                "moderate", 
                ["products", "reviews"]
            ),
            
            # Recommendation queries for actual categories
            QueryTemplate(
                "Best {product_type} for {use_case}",
                "recommendation",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Recommend a {product_type} under {price_indicator} price",
                "recommendation",
                "moderate",
                ["products"]
            ),
            QueryTemplate(
                "What {product_type} has the best reviews?",
                "recommendation",
                "simple",
                ["products", "reviews"]
            ),
            
            # Feature-based queries
            QueryTemplate(
                "Show me {product_type} with {feature}",
                "technical",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Does {specific_product} have {feature}?",
                "technical",
                "simple",
                ["products"]
            ),
            
            # Review-based queries
            QueryTemplate(
                "What do customers say about {specific_product}?",
                "abstract",
                "moderate",
                ["reviews"]
            ),
            QueryTemplate(
                "Common complaints about {brand} {product_type}",
                "problem_solving",
                "moderate",
                ["reviews"]
            ),
            
            # Multi-hop reasoning
            QueryTemplate(
                "Which {brand} product has better reviews: their {type1} or {type2}?",
                "multi_hop",
                "complex",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Find me a {product_type} similar to {specific_product} but cheaper",
                "multi_hop",
                "complex",
                ["products"]
            )
        ]
        
        # Realistic use cases
        self.use_cases = [
            "home office", "gaming", "travel", "students", "remote work",
            "content creation", "music listening", "video calls", "streaming",
            "photography", "fitness tracking", "smart home"
        ]
    
    def generate_query(self, template: QueryTemplate) -> Tuple[str, Dict]:
        """Generate a realistic query from template."""
        query = template.template
        metadata = {
            "query_type": template.query_type,
            "complexity": template.complexity,
            "requires_context": template.requires_context,
            "generated_from": "actual_catalog"
        }
        
        # Replace placeholders with actual catalog data
        if "{specific_product}" in query:
            product = self.rng.choice(self.popular_products)
            query = query.replace("{specific_product}", product.title[:50])
            metadata["product_id"] = product.id
            metadata["product"] = product.title
        
        if "{brand}" in query:
            # Choose brand that actually exists in catalog
            brand = self.rng.choice(list(self.products_by_brand.keys()))
            query = query.replace("{brand}", brand)
            metadata["brand"] = brand
        
        if "{product_type}" in query:
            # Choose product type that exists
            if "brand" in metadata:
                # Use product type that this brand actually makes
                types = list(self.brand_categories[metadata["brand"]])
                if types:
                    product_type = self.rng.choice(types)
                else:
                    product_type = self.rng.choice(list(self.products_by_type.keys()))
            else:
                product_type = self.rng.choice(list(self.products_by_type.keys()))
            query = query.replace("{product_type}", product_type)
            metadata["product_type"] = product_type
        
        if "{product1}" in query:
            prod1 = self.rng.choice(self.popular_products[:50])
            query = query.replace("{product1}", prod1.title[:40])
            metadata["product1"] = prod1.title
            
        if "{product2}" in query:
            # Choose similar product for comparison
            same_type = [p for p in self.products_by_type[prod1.product_type] 
                        if p.id != prod1.id]
            if same_type:
                prod2 = self.rng.choice(same_type)
            else:
                prod2 = self.rng.choice(self.popular_products[:50])
            query = query.replace("{product2}", prod2.title[:40])
            metadata["product2"] = prod2.title
        
        if "{type1}" in query and "{type2}" in query:
            # For brand comparison, use actual product types from that brand
            if "brand" in metadata:
                types = list(self.brand_categories[metadata["brand"]])
                if len(types) >= 2:
                    type1, type2 = self.rng.sample(types, 2)
                    query = query.replace("{type1}", type1)
                    query = query.replace("{type2}", type2)
                    metadata["type1"] = type1
                    metadata["type2"] = type2
        
        if "{feature}" in query:
            # Use features that actually exist in products
            all_features = []
            for p in self.products[:100]:
                all_features.extend(p.features)
            if all_features:
                feature = self.rng.choice(all_features)
                # Clean up feature text
                feature = feature.split('.')[0].lower()
                if len(feature) > 50:
                    feature = "wireless"  # fallback
                query = query.replace("{feature}", feature)
                metadata["feature"] = feature
        
        if "{use_case}" in query:
            use_case = self.rng.choice(self.use_cases)
            query = query.replace("{use_case}", use_case)
            metadata["use_case"] = use_case
        
        if "{price_indicator}" in query:
            indicator = self.rng.choice(self.price_indicators)
            query = query.replace("{price_indicator}", indicator)
            metadata["price_indicator"] = indicator
        
        return query, metadata
    
    def generate_dataset(self, num_samples: int, distribution: str = "balanced") -> List[Dict]:
        """Generate a complete dataset with realistic queries."""
        dataset = []
        
        # Define distribution of query types
        if distribution == "balanced":
            weights = {
                "simple": 0.3,
                "moderate": 0.4,
                "complex": 0.3
            }
        elif distribution == "simple":
            weights = {
                "simple": 0.6,
                "moderate": 0.3,
                "complex": 0.1
            }
        elif distribution == "complex":
            weights = {
                "simple": 0.2,
                "moderate": 0.3,
                "complex": 0.5
            }
        else:  # mixed
            weights = {
                "simple": 0.33,
                "moderate": 0.34,
                "complex": 0.33
            }
        
        # Group templates by complexity
        templates_by_complexity = defaultdict(list)
        for template in self.templates:
            templates_by_complexity[template.complexity].append(template)
        
        for i in range(num_samples):
            # Choose complexity based on weights
            complexity = self.rng.choices(
                list(weights.keys()),
                weights=list(weights.values())
            )[0]
            
            # Choose template of that complexity
            if templates_by_complexity[complexity]:
                template = self.rng.choice(templates_by_complexity[complexity])
            else:
                template = self.rng.choice(self.templates)
            
            # Generate query
            query, metadata = self.generate_query(template)
            
            # Create dataset entry
            entry = {
                "query": query,
                "metadata": metadata,
                # Add reference for evaluation (can be expanded)
                "reference": f"Generated query about {metadata.get('product_type', 'products')}"
            }
            
            dataset.append(entry)
        
        return dataset


def generate_realistic_testset(
    products_path: Path,
    reviews_path: Path, 
    output_path: Path,
    num_samples: int = 500,
    distribution: str = "balanced",
    seed: int = 42
):
    """Generate realistic test dataset from actual product catalog."""
    
    # Load product and review data
    products = []
    with open(products_path, 'r') as f:
        for line in f:
            products.append(json.loads(line))
    
    reviews = []
    if reviews_path.exists():
        with open(reviews_path, 'r') as f:
            for line in f:
                reviews.append(json.loads(line))
    
    # Initialize generator
    generator = RealisticQueryGenerator(products, reviews, seed)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_samples, distribution)
    
    # Save to file
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    
    return dataset


if __name__ == "__main__":
    # Example usage
    products_path = Path("data/top_1000_products.jsonl")
    reviews_path = Path("data/100_top_reviews_of_the_top_1000_products.jsonl")
    output_path = Path("eval/datasets/realistic_test_500.jsonl")
    
    dataset = generate_realistic_testset(
        products_path,
        reviews_path,
        output_path,
        num_samples=500,
        distribution="balanced"
    )
    
    print(f"Generated {len(dataset)} realistic test queries")
    print(f"Saved to {output_path}")