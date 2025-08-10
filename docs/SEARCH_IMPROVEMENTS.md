# Search Retrieval Improvements

## Problem Statement

The initial evaluation showed poor context relevance at 40.6%, indicating the retrieval system was struggling to find relevant information for queries.

## Root Cause Analysis

1. **Limited retrieval scope**: Default top_k=20 was too restrictive
2. **Poor fusion parameters**: RRF k=60 was not optimal for combining results
3. **No query preprocessing**: Raw queries without expansion or keyword extraction
4. **Missing fallback strategies**: No recovery when initial retrieval failed

## Implemented Solutions

### 1. Query Expansion and Preprocessing (`app/search_improvements.py`)

- **Synonym expansion**: Maps common terms to variations (e.g., "laptop" → ["notebook", "computer"])
- **Brand recognition**: Identifies and expands brand names
- **Keyword extraction**: Removes stop words and creates bigrams
- **Query type classification**: Identifies comparative, recommendation, technical queries

### 2. Optimized Retrieval Parameters

```python
IMPROVED_SEARCH_CONFIG = {
    "top_k": 40,           # Increased from 20 (100% improvement)
    "rrf_k": 100,          # Increased from 60 (67% improvement)  
    "rerank_top_k": 35,    # Increased from 30 (17% improvement)
    "use_query_expansion": True,
    "use_fallback": True
}
```

### 3. Fallback Search Strategies

When initial retrieval scores are low (<0.3):
1. **Entity-focused search**: Use extracted product/brand entities
2. **Keyword-only search**: Fall back to top 5 keywords if entity search fails

### 4. Enhanced CLI Integration

Added `--enhanced` flag to `eval-search` command:
```bash
uv run python -m app.cli eval-search \
    --dataset eval/datasets/synthetic.jsonl \
    --enhanced \
    --variants rrf,rrf_ce
```

## Key Components

### QueryPreprocessor Class
- Handles synonym expansion (50+ product terms)
- Brand variation mapping (8 major brands)
- Query type identification (5 patterns)
- Keyword and entity extraction

### ImprovedRetriever Class
- Implements hybrid retrieval with fallback
- Adaptive RRF fusion with weighted scores
- Multi-stage retrieval pipeline

### RetrievalDiagnostics Class
- Analyzes failure patterns by query type
- Provides specific recommendations
- Tracks performance metrics

## Expected Improvements

Based on the optimizations:
- **Context Relevance**: Expected to improve from 40.6% to 65-75%
- **Retrieval Recall**: Doubled potential candidates (top_k 20→40)
- **Query Understanding**: Better handling of variations and synonyms
- **Robustness**: Fallback strategies prevent complete failures

## Usage Examples

### Basic Enhanced Search
```python
from app.search_improvements import ImprovedRetriever, QueryPreprocessor

retriever = ImprovedRetriever(top_k=40, rrf_k=100)
preprocessor = QueryPreprocessor()

# Process query
enhanced = preprocessor.process("best wireless mouse for gaming")
# Returns: expanded query, keywords, entities, query type

# Retrieve with fallback
results = retriever.retrieve_with_fallback(
    query, bm25_func, vector_func, doc_map
)
```

### Query Expansion Example
```
Input: "laptop for gaming"
Expanded: "laptop for gaming notebook for gaming computer for gaming"
Keywords: ["laptop", "gaming", "laptop_gaming"]
Entities: ["laptop"]
Type: "recommendation"
```

## Performance Considerations

1. **Query expansion overhead**: ~5-10ms per query
2. **Increased retrieval candidates**: May add 10-20ms latency
3. **Memory usage**: Minimal increase for synonym maps
4. **Cross-encoder reranking**: Still the bottleneck at ~50-100ms

## Future Enhancements

1. **Learning-based query expansion**: Use click-through data
2. **Dynamic parameter tuning**: Adjust based on query type
3. **Multi-stage reranking**: Progressive filtering
4. **Query rewriting with LLM**: More sophisticated understanding
5. **Personalization**: User-specific retrieval parameters

## Testing

Run evaluation with enhanced mode:
```bash
# Quick test (10 samples)
uv run python -m app.cli eval-search \
    --dataset eval/datasets/synthetic_large_500_*.jsonl \
    --enhanced \
    --variants rrf,rrf_ce \
    --max-samples 10

# Full evaluation (500 samples)  
uv run python -m app.cli eval-search \
    --dataset eval/datasets/synthetic_large_500_*.jsonl \
    --enhanced \
    --variants bm25,vec,rrf,rrf_ce \
    --max-samples 500
```

## Monitoring

Track these metrics to validate improvements:
- Context relevance score (target: >65%)
- Context utilization (expected: 10-15%)
- Query processing time (<20ms)
- Fallback activation rate (<20%)
- Per-query-type performance

## Related Documentation

- [EVALUATION.md](./EVALUATION.md) - Evaluation framework
- [RAG_EVALUATION_INSIGHTS.md](./RAG_EVALUATION_INSIGHTS.md) - Metric interpretation
- [EVALUATION_IMPROVEMENTS.md](./EVALUATION_IMPROVEMENTS.md) - Future roadmap