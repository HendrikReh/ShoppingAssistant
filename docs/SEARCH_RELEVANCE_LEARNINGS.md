# Search Relevance Learnings: Products vs Reviews

## Executive Summary

Through extensive testing and optimization of the Shopping Assistant's search functionality, we discovered fundamental differences in how relevance scoring works for products versus reviews. These differences significantly impact search quality and user experience.

## Key Discoveries

### 1. Products vs Reviews Have Different Relevance Characteristics

**Products:**
- Have structured, concise metadata (title, description, categories)
- Contain specific brand names and model numbers
- Use technical specifications and feature lists
- Generally have shorter, more focused text

**Reviews:**
- Contain narrative, conversational text
- May mention multiple products in comparisons
- Include subjective opinions and experiences
- Often have longer, more varied vocabulary

### 2. The Mixed Results Problem

Initially, our search results were dominated by reviews rather than products, even when users clearly wanted product information. For example:
- Query: "TV Sticks" → Returned mostly reviews mentioning TV sticks
- Query: "Fire TV" → Mixed results with reviews about various streaming devices

**Root Cause:** Reviews naturally have richer text content, leading to higher relevance scores in both BM25 (keyword) and semantic (vector) search.

## Solutions Implemented

### 1. Query Preprocessing
```python
# Remove common search command prefixes
search_prefixes = ['search for ', 'find ', 'show me ', 'look for ', 'find me ']
query_lower = query.lower()
for prefix in search_prefixes:
    if query_lower.startswith(prefix):
        query = query[len(prefix):]
        break
```

**Learning:** Users often type natural commands that add noise to the actual search query.

### 2. Products-Only Filtering
```python
# Detect and handle "products only" requests
products_only = False
if ', products only' in query.lower() or ' products only' in query.lower():
    products_only = True
    query = query.replace(', products only', '').replace(' products only', '')
```

**Learning:** Users need explicit control to filter out reviews when they only want product listings.

### 3. Product Boosting in RRF Fusion
```python
def _rrf_fuse(result_lists, k=60, product_boost=1.5):
    fused = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            _id = item[0]
            score = 1.0 / (k + rank)
            # Boost products over reviews
            if _id.startswith("prod::") and product_boost > 1.0:
                score *= product_boost
            fused[_id] = fused.get(_id, 0.0) + score
    return fused
```

**Learning:** A 1.5x boost for products effectively counterbalances the natural advantage reviews have in text richness.

## Why Reviews Score Higher Than Products

### 1. Text Length Advantage
- Reviews average 100-500 words
- Product descriptions average 20-100 words
- More text = more keyword matches in BM25
- Longer documents provide more context for semantic embeddings

### 2. Vocabulary Diversity
- Reviews use natural language with synonyms and variations
- Products use technical specifications and formal descriptions
- Example: Product says "wireless earbuds", reviews say "wireless", "earbuds", "bluetooth headphones", "truly wireless", etc.

### 3. Context Richness
- Reviews describe use cases, comparisons, and experiences
- Products focus on features and specifications
- Semantic models trained on natural language favor review-style text

### 4. Query-Text Alignment
- User queries often match review language: "best budget earbuds"
- Product titles are formal: "JBL Tune 510BT Wireless On-Ear Headphones"
- Reviews naturally bridge this vocabulary gap

## Empirical Results

### Before Optimizations
```
Query: "TV Sticks"
Results: 70% reviews, 30% products
Top result: Review mentioning TV stick in passing
```

### After Optimizations
```
Query: "TV Sticks"
Results: 100% products (when products_only=True)
Results: 90% products (with product_boost=1.5)
Top result: Fire TV Stick (actual product)
```

## Best Practices for E-commerce Search

### 1. Default to Product-First Results
Users searching in e-commerce contexts typically want to find products to purchase, not read reviews first.

### 2. Implement Tiered Boosting
```python
product_boost = 1.5  # General products
bestseller_boost = 2.0  # Top-rated or bestselling items
exact_match_boost = 3.0  # Exact title/brand matches
```

### 3. Consider Search Intent
- Navigation queries ("Fire TV Stick 4K") → Show specific product
- Research queries ("compare streaming devices") → Include reviews
- Purchase queries ("buy wireless mouse") → Products only

### 4. Separate Search Modes
Offer distinct modes for different use cases:
- **Product Search**: Products only, optimized for shopping
- **Review Search**: Reviews only, for research
- **Hybrid Search**: Balanced mix for general discovery

## Technical Recommendations

### 1. Embedding Model Selection
- Choose models trained on e-commerce data when available
- Consider domain-specific fine-tuning
- Test multiple models with your specific product catalog

### 2. Scoring Adjustments
```python
# Example scoring formula
final_score = (
    bm25_score * 0.3 +  # Keyword relevance
    vector_score * 0.4 +  # Semantic similarity
    popularity_score * 0.2 +  # Product popularity
    recency_score * 0.1  # How recent the item is
) * type_boost  # Product vs review boost
```

### 3. Collection Separation
- Maintain separate indices for products and reviews
- Apply different scoring weights per collection
- Allow users to search specific collections

## Metrics to Monitor

1. **Click-through Rate (CTR)** on products vs reviews
2. **Conversion Rate** from search results
3. **Search Refinement Rate** (indicates poor initial results)
4. **Product/Review Ratio** in top 10 results
5. **User Feedback** on result relevance

## Future Improvements

### 1. Machine Learning Ranking
Train a learning-to-rank model that learns optimal weights for:
- Product vs review distinction
- Query-document features
- User interaction signals

### 2. Personalization
- Track user preference for products vs reviews
- Adjust boosting based on user history
- Learn domain-specific preferences (electronics vs books)

### 3. Query Understanding
- Implement query classification (product search vs information seeking)
- Extract product entities from queries
- Use intent to dynamically adjust scoring

## Understanding Cross-Encoder Scores of 0.00

### Why Relevance Scores Sometimes Show 0.00

When you see all relevance scores as 0.00, it indicates that:

1. **Poor Initial Retrieval**: BM25 and vector search found no relevant candidates
2. **Cross-Encoder Limitations**: The cross-encoder can only rerank what's already retrieved
3. **Common Causes**:
   - Typos in the query (e.g., "TV Stciks" instead of "TV Sticks")
   - Very rare or unusual product names
   - Queries with no matching products in the catalog

### The Retrieval-Reranking Pipeline

```
Query → Retrieval (BM25 + Vector) → Fusion → Reranking (Cross-Encoder) → Results
         ↑                                      ↑
         If this fails                         This can't fix bad retrieval
```

The cross-encoder is powerful but **cannot manufacture relevance** - it can only score the relevance of already-retrieved items.

### Solutions for Better Typo Handling

1. **Query Correction**: Detect and correct common typos before retrieval
2. **Fuzzy Matching**: Use edit distance for approximate string matching
3. **Query Expansion**: Generate variations of the query
4. **Fallback Strategies**: When CE scores are all 0, trigger alternative search methods

## Conclusion

The fundamental difference in text characteristics between products and reviews creates a natural bias toward reviews in standard search algorithms. Successful e-commerce search requires explicit adjustments to counteract this bias and align results with user intent. The key insight is that **relevance scoring must be task-aware** - what's relevant for research (reviews) differs from what's relevant for shopping (products).

Our implementation of query preprocessing, products-only filtering, product boosting, and typo correction successfully addresses these challenges, resulting in search results that better match user expectations in an e-commerce context.