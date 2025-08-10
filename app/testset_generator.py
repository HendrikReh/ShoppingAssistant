"""Synthetic test data generation for RAG evaluation using RAGAS.

This module generates diverse test datasets with various query types:
- Single-hop queries (simple factual questions)
- Multi-hop queries (requiring reasoning across multiple documents)
- Abstract queries (interpretive questions)
- Comparative queries (comparing products/features)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

import typer
from typing_extensions import Annotated


@dataclass
class QueryTemplate:
    """Template for generating specific query types."""
    
    template: str
    query_type: str
    complexity: str  # simple, moderate, complex
    requires_context: List[str] = field(default_factory=list)  # products, reviews, both


class EcommerceQueryGenerator:
    """Generate diverse e-commerce queries for testing."""
    
    def __init__(self, products: List[Dict], reviews: List[Dict], seed: int = 42):
        """Initialize with product and review data.
        
        Args:
            products: List of product documents
            reviews: List of review documents  
            seed: Random seed for reproducibility
        """
        self.products = products
        self.reviews = reviews
        self.rng = random.Random(seed)
        
        # Extract key entities for query generation
        self._extract_entities()
        self._define_templates()
    
    def _extract_entities(self):
        """Extract product categories, brands, features for query generation."""
        self.categories = set()
        self.brands = set()
        self.features = set()
        self.price_ranges = []
        
        for product in self.products:
            # Extract categories from title
            title = product.get('title', '').lower()
            if 'laptop' in title:
                self.categories.add('laptop')
            elif 'headphone' in title or 'earbud' in title:
                self.categories.add('headphones')
            elif 'mouse' in title:
                self.categories.add('mouse')
            elif 'keyboard' in title:
                self.categories.add('keyboard')
            elif 'monitor' in title or 'display' in title:
                self.categories.add('monitor')
            elif 'speaker' in title:
                self.categories.add('speaker')
            elif 'cable' in title or 'charger' in title:
                self.categories.add('cable')
            elif 'webcam' in title or 'camera' in title:
                self.categories.add('webcam')
            elif 'drive' in title or 'storage' in title:
                self.categories.add('storage')
            elif 'tablet' in title:
                self.categories.add('tablet')
            
            # Extract brands (common electronics brands)
            for brand in ['apple', 'samsung', 'sony', 'bose', 'logitech', 'dell', 
                         'hp', 'lenovo', 'asus', 'microsoft', 'jbl', 'anker']:
                if brand in title:
                    self.brands.add(brand.capitalize())
            
            # Extract features from description
            desc_data = product.get('description', '')
            if isinstance(desc_data, list):
                desc = ' '.join(desc_data).lower()
            else:
                desc = str(desc_data).lower()
            for feature in ['wireless', 'bluetooth', 'usb-c', 'noise cancelling',
                          'waterproof', 'rgb', 'mechanical', '4k', 'portable',
                          'fast charging', 'long battery', 'ergonomic']:
                if feature in desc or feature in title:
                    self.features.add(feature)
        
        # Set defaults if no entities found
        if not self.categories:
            self.categories = {'laptop', 'headphones', 'mouse', 'keyboard', 'speaker'}
        if not self.brands:
            self.brands = {'Apple', 'Samsung', 'Sony', 'Logitech', 'Dell'}
        if not self.features:
            self.features = {'wireless', 'bluetooth', 'portable', 'ergonomic'}
    
    def _define_templates(self):
        """Define query templates for different types."""
        self.templates = [
            # Single-hop factual queries
            QueryTemplate(
                "What is the best {category} for {use_case}?",
                "single_hop_factual",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Show me {brand} {category}",
                "single_hop_factual",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Find {category} with {feature}",
                "single_hop_factual",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "What do customers say about {brand} {category}?",
                "single_hop_factual",
                "simple",
                ["reviews"]
            ),
            
            # Multi-hop reasoning queries
            QueryTemplate(
                "Compare {brand1} and {brand2} {category} in terms of {aspect}",
                "multi_hop_reasoning",
                "complex",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Which {category} has the best {feature} according to reviews?",
                "multi_hop_reasoning",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "What are the pros and cons of {feature} {category} based on user experiences?",
                "multi_hop_reasoning",
                "complex",
                ["reviews"]
            ),
            
            # Abstract/interpretive queries
            QueryTemplate(
                "How has {category} technology evolved in recent products?",
                "abstract_interpretive",
                "complex",
                ["products"]
            ),
            QueryTemplate(
                "What makes a good {category} for {use_case}?",
                "abstract_interpretive",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Why do users prefer {brand} over competitors?",
                "abstract_interpretive",
                "complex",
                ["reviews"]
            ),
            
            # Comparative queries
            QueryTemplate(
                "{category} under ${price} with good reviews",
                "comparative",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Best value {category} with {feature}",
                "comparative",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Most reliable {brand} {category}",
                "comparative",
                "simple",
                ["products", "reviews"]
            ),
            
            # Recommendation queries
            QueryTemplate(
                "Recommend a {category} for {use_case}",
                "recommendation",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "What {category} should I buy for {scenario}?",
                "recommendation",
                "moderate",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Suggest alternatives to {brand} {category}",
                "recommendation",
                "complex",
                ["products"]
            ),
            
            # Technical specification queries
            QueryTemplate(
                "What are the specifications of {brand} {category}?",
                "technical",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Does {brand} {category} support {feature}?",
                "technical",
                "simple",
                ["products"]
            ),
            QueryTemplate(
                "Battery life of {category} with {feature}",
                "technical",
                "moderate",
                ["products", "reviews"]
            ),
            
            # Problem-solving queries
            QueryTemplate(
                "How to choose between {category} options?",
                "problem_solving",
                "complex",
                ["products", "reviews"]
            ),
            QueryTemplate(
                "Common issues with {brand} {category}",
                "problem_solving",
                "moderate",
                ["reviews"]
            ),
            QueryTemplate(
                "Is {feature} worth it in {category}?",
                "problem_solving",
                "moderate",
                ["products", "reviews"]
            )
        ]
        
        # Use cases and scenarios for templates
        self.use_cases = [
            "gaming", "work", "travel", "home office", "students",
            "video editing", "music production", "coding", "casual use",
            "professional use", "outdoor activities", "fitness"
        ]
        
        self.scenarios = [
            "working from home", "traveling frequently", "on a budget",
            "starting college", "content creation", "competitive gaming",
            "remote meetings", "streaming", "photography"
        ]
        
        self.aspects = [
            "performance", "battery life", "build quality", "price",
            "features", "durability", "comfort", "ease of use", "value"
        ]
        
        self.price_points = ["100", "200", "300", "500", "1000", "1500"]
    
    def generate_query(self, template: QueryTemplate) -> Tuple[str, Dict]:
        """Generate a query from a template.
        
        Returns:
            Tuple of (query_string, metadata_dict)
        """
        # Fill in template placeholders
        query = template.template
        metadata = {
            "query_type": template.query_type,
            "complexity": template.complexity,
            "requires_context": template.requires_context
        }
        
        # Replace placeholders with random entities
        if "{category}" in query:
            category = self.rng.choice(list(self.categories))
            query = query.replace("{category}", category)
            metadata["category"] = category
        
        if "{brand}" in query or "{brand1}" in query:
            brand = self.rng.choice(list(self.brands))
            query = query.replace("{brand}", brand)
            query = query.replace("{brand1}", brand)
            metadata["brand"] = brand
        
        if "{brand2}" in query:
            brands = list(self.brands)
            if metadata.get("brand"):
                brands.remove(metadata["brand"])
            if brands:
                brand2 = self.rng.choice(brands)
                query = query.replace("{brand2}", brand2)
                metadata["brand2"] = brand2
        
        if "{feature}" in query:
            feature = self.rng.choice(list(self.features))
            query = query.replace("{feature}", feature)
            metadata["feature"] = feature
        
        if "{use_case}" in query:
            use_case = self.rng.choice(self.use_cases)
            query = query.replace("{use_case}", use_case)
            metadata["use_case"] = use_case
        
        if "{scenario}" in query:
            scenario = self.rng.choice(self.scenarios)
            query = query.replace("{scenario}", scenario)
            metadata["scenario"] = scenario
        
        if "{aspect}" in query:
            aspect = self.rng.choice(self.aspects)
            query = query.replace("{aspect}", aspect)
            metadata["aspect"] = aspect
        
        if "{price}" in query:
            price = self.rng.choice(self.price_points)
            query = query.replace("{price}", price)
            metadata["price_limit"] = price
        
        return query, metadata
    
    def generate_dataset(
        self, 
        num_samples: int = 100,
        distribution: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """Generate a diverse test dataset.
        
        Args:
            num_samples: Number of test samples to generate
            distribution: Distribution of query types (should sum to 1.0)
                         Default: balanced across all types
        
        Returns:
            List of test samples with queries and metadata
        """
        if distribution is None:
            # Default balanced distribution
            query_types = list(set(t.query_type for t in self.templates))
            distribution = {qt: 1.0/len(query_types) for qt in query_types}
        
        # Calculate samples per query type
        samples_per_type = {}
        remaining = num_samples
        for query_type, weight in distribution.items():
            count = int(num_samples * weight)
            samples_per_type[query_type] = count
            remaining -= count
        
        # Add remaining samples to largest category
        if remaining > 0:
            max_type = max(distribution.keys(), key=lambda k: distribution[k])
            samples_per_type[max_type] += remaining
        
        # Generate samples
        dataset = []
        for query_type, count in samples_per_type.items():
            # Get templates for this query type
            type_templates = [t for t in self.templates if t.query_type == query_type]
            if not type_templates:
                continue
            
            for _ in range(count):
                template = self.rng.choice(type_templates)
                query, metadata = self.generate_query(template)
                
                # Create unique ID for the query
                query_id = hashlib.md5(query.encode()).hexdigest()[:8]
                
                sample = {
                    "query_id": query_id,
                    "query": query,
                    "metadata": metadata
                }
                
                # Add expected context hints for evaluation
                if "products" in template.requires_context:
                    sample["expected_context_type"] = "products"
                if "reviews" in template.requires_context:
                    if sample.get("expected_context_type"):
                        sample["expected_context_type"] = "both"
                    else:
                        sample["expected_context_type"] = "reviews"
                
                dataset.append(sample)
        
        # Shuffle for variety
        self.rng.shuffle(dataset)
        
        return dataset
    
    def generate_with_reference_answers(
        self,
        num_samples: int = 50,
        use_llm: bool = False
    ) -> List[Dict]:
        """Generate test data with reference answers.
        
        Args:
            num_samples: Number of samples to generate
            use_llm: Whether to use LLM to generate reference answers
        
        Returns:
            List of samples with queries and reference answers
        """
        dataset = self.generate_dataset(num_samples)
        
        for sample in dataset:
            # Generate reference answer based on query type
            query_type = sample["metadata"]["query_type"]
            
            if query_type == "single_hop_factual":
                # Simple factual template
                sample["reference_answer"] = self._generate_factual_reference(sample)
            elif query_type == "multi_hop_reasoning":
                # Complex reasoning template
                sample["reference_answer"] = self._generate_reasoning_reference(sample)
            elif query_type == "comparative":
                # Comparative analysis template
                sample["reference_answer"] = self._generate_comparative_reference(sample)
            elif query_type == "recommendation":
                # Recommendation template
                sample["reference_answer"] = self._generate_recommendation_reference(sample)
            else:
                # Generic template
                sample["reference_answer"] = self._generate_generic_reference(sample)
        
        return dataset
    
    def _generate_factual_reference(self, sample: Dict) -> str:
        """Generate factual reference answer."""
        metadata = sample["metadata"]
        category = metadata.get("category", "product")
        
        templates = [
            f"The best {category} depends on your specific needs, but popular options include models with the requested features.",
            f"For {category}, consider factors like performance, price, and user reviews when making your selection.",
            f"Top-rated {category} products typically offer good value and reliability based on customer feedback."
        ]
        
        return self.rng.choice(templates)
    
    def _generate_reasoning_reference(self, sample: Dict) -> str:
        """Generate reasoning reference answer."""
        metadata = sample["metadata"]
        
        templates = [
            "Based on analysis of multiple products and reviews, the key differences include build quality, feature set, and price point. Users generally prefer options that balance these factors well.",
            "Comparing across different sources shows varied perspectives. Professional reviews highlight technical specifications while user reviews focus on real-world experience and reliability.",
            "The evaluation should consider both objective specifications and subjective user experiences to provide a comprehensive comparison."
        ]
        
        return self.rng.choice(templates)
    
    def _generate_comparative_reference(self, sample: Dict) -> str:
        """Generate comparative reference answer."""
        templates = [
            "When comparing options, consider factors like price-to-performance ratio, feature availability, and long-term reliability based on user feedback.",
            "The best value typically comes from products that balance essential features with reasonable pricing, as indicated by positive review trends.",
            "Comparative analysis shows that mid-range options often provide the best balance of features and affordability for most users."
        ]
        
        return self.rng.choice(templates)
    
    def _generate_recommendation_reference(self, sample: Dict) -> str:
        """Generate recommendation reference answer."""
        metadata = sample["metadata"]
        use_case = metadata.get("use_case", "general use")
        
        return f"For {use_case}, I recommend considering products that prioritize the specific features needed for this use case, while maintaining good overall quality and value based on user reviews."
    
    def _generate_generic_reference(self, sample: Dict) -> str:
        """Generate generic reference answer."""
        return "Based on the available information, the answer depends on specific requirements and preferences. Consider reviewing product specifications and user feedback to make an informed decision."


def generate_testset_from_documents(
    products_path: Path,
    reviews_path: Path,
    output_path: Path,
    num_samples: int = 500,
    include_reference: bool = True,
    seed: int = 42
) -> None:
    """Generate synthetic test dataset from product and review documents.
    
    Args:
        products_path: Path to products JSONL file
        reviews_path: Path to reviews JSONL file
        output_path: Path to save generated test dataset
        num_samples: Number of test samples to generate
        include_reference: Whether to include reference answers
        seed: Random seed for reproducibility
    """
    print(f"Loading documents from {products_path} and {reviews_path}...")
    
    # Load documents
    products = []
    with open(products_path, 'r') as f:
        for line in f:
            products.append(json.loads(line))
    
    reviews = []
    with open(reviews_path, 'r') as f:
        for line in f:
            reviews.append(json.loads(line))
    
    print(f"Loaded {len(products)} products and {len(reviews)} reviews")
    
    # Initialize generator
    generator = EcommerceQueryGenerator(products, reviews, seed)
    
    # Define distribution for diverse query types
    distribution = {
        "single_hop_factual": 0.25,      # 25% simple factual queries
        "multi_hop_reasoning": 0.20,     # 20% complex reasoning
        "abstract_interpretive": 0.10,   # 10% abstract questions
        "comparative": 0.15,             # 15% comparative queries
        "recommendation": 0.15,          # 15% recommendations
        "technical": 0.10,               # 10% technical specs
        "problem_solving": 0.05          # 5% problem-solving
    }
    
    print(f"Generating {num_samples} test samples with distribution:")
    for query_type, weight in distribution.items():
        print(f"  - {query_type}: {weight*100:.0f}% ({int(num_samples*weight)} samples)")
    
    # Generate dataset
    if include_reference:
        dataset = generator.generate_with_reference_answers(num_samples)
    else:
        dataset = generator.generate_dataset(num_samples, distribution)
    
    # Add dataset metadata
    metadata = {
        "total_samples": len(dataset),
        "generation_seed": seed,
        "distribution": distribution,
        "query_types": list(set(s["metadata"]["query_type"] for s in dataset)),
        "complexity_distribution": {}
    }
    
    # Calculate complexity distribution
    for level in ["simple", "moderate", "complex"]:
        count = sum(1 for s in dataset if s["metadata"]["complexity"] == level)
        metadata["complexity_distribution"][level] = count
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write metadata as first line
        f.write(json.dumps({"_metadata": metadata}) + "\n")
        
        # Write samples
        for sample in dataset:
            # Format for RAGAS compatibility
            output_sample = {
                "question": sample["query"],
                "query": sample["query"],  # Include both for compatibility
                "query_id": sample["query_id"],
                "metadata": sample["metadata"]
            }
            
            if "reference_answer" in sample:
                output_sample["reference_answer"] = sample["reference_answer"]
                output_sample["ground_truth"] = sample["reference_answer"]
            
            if "expected_context_type" in sample:
                output_sample["expected_context_type"] = sample["expected_context_type"]
            
            f.write(json.dumps(output_sample) + "\n")
    
    print(f"\n✅ Generated {len(dataset)} test samples")
    print(f"📁 Saved to {output_path}")
    print(f"\nDataset statistics:")
    print(f"  - Query types: {len(metadata['query_types'])}")
    print(f"  - Complexity distribution:")
    for level, count in metadata["complexity_distribution"].items():
        print(f"    - {level}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Save separate metadata file
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📊 Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # CLI interface
    app = typer.Typer(help="Generate synthetic test datasets for RAG evaluation")
    
    @app.command()
    def generate(
        products_path: Annotated[Path, typer.Option(help="Path to products JSONL")] = Path("data/top_1000_products.jsonl"),
        reviews_path: Annotated[Path, typer.Option(help="Path to reviews JSONL")] = Path("data/100_top_reviews_of_the_top_1000_products.jsonl"),
        output_path: Annotated[Path, typer.Option(help="Output path for test dataset")] = Path("eval/datasets/synthetic_test_500.jsonl"),
        num_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 500,
        include_reference: Annotated[bool, typer.Option(help="Include reference answers")] = True,
        seed: Annotated[int, typer.Option(help="Random seed")] = 42
    ):
        """Generate synthetic test dataset from documents."""
        generate_testset_from_documents(
            products_path,
            reviews_path,
            output_path,
            num_samples,
            include_reference,
            seed
        )
    
    app()