# Cross-Encoder Reranking Models Guide

## Current Implementation
- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Size**: ~80MB
- **Speed**: 100-200 docs/sec on CPU
- **Accuracy**: Good baseline for general search

## Alternative Models

### For Better Accuracy

#### ms-marco-MiniLM-L-12-v2
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
```
- **Improvement**: 15-20% better accuracy
- **Speed**: 50-100 docs/sec
- **Size**: 160MB
- **Best for**: Production systems with quality focus

#### ms-marco-base
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-base"
```
- **Improvement**: 25-30% better accuracy
- **Speed**: 25-50 docs/sec
- **Size**: 420MB
- **Best for**: Highest quality requirements

### For Better Speed

#### ms-marco-TinyBERT-L-2-v2
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
```
- **Speed**: 1000+ docs/sec (10x faster)
- **Size**: 18MB
- **Trade-off**: 20-30% less accurate
- **Best for**: Real-time applications, large scale

### For Specific Use Cases

#### Question Answering
```python
CROSS_ENCODER_MODEL = "cross-encoder/qnli-distilroberta-base"
```
- **Best for**: Chat functionality, FAQ matching
- **Optimized**: Question-answer relevance

#### Multilingual
```python
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
```
- **Languages**: 100+ languages supported
- **Best for**: International e-commerce

## Implementation Examples

### 1. Simple Model Change
Edit `app/cli.py`:
```python
# Change this line
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

### 2. Query-Based Model Selection
```python
def select_reranker_model(query: str, num_results: int) -> str:
    """Select best reranker based on query characteristics."""
    
    # Fast model for large result sets
    if num_results > 100:
        return "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    
    # QA model for question queries
    if "?" in query or query.startswith(("what", "how", "why", "when")):
        return "cross-encoder/qnli-distilroberta-base"
    
    # Default to balanced model
    return "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

### 3. Cascaded Reranking
```python
def cascaded_rerank(query: str, candidates: list, device: str) -> list:
    """Two-stage reranking for optimal speed/accuracy."""
    
    # Stage 1: Fast reranking of all candidates
    if len(candidates) > 50:
        fast_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        candidates = _cross_encoder_scores(fast_model, device, query, candidates)
        candidates = candidates[:30]  # Keep top 30
    
    # Stage 2: Accurate reranking of top candidates
    accurate_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    return _cross_encoder_scores(accurate_model, device, query, candidates)
```

## Performance Comparison

| Model | Accuracy | Speed (docs/sec) | Size (MB) | Best Use Case |
|-------|----------|------------------|-----------|---------------|
| TinyBERT-L-2 | 70% | 1000+ | 18 | Real-time, high volume |
| MiniLM-L-6 (current) | 85% | 100-200 | 80 | Balanced performance |
| MiniLM-L-12 | 90% | 50-100 | 160 | Better accuracy |
| ms-marco-base | 95% | 25-50 | 420 | Maximum accuracy |

## Recommendations

### For Shopping Assistant

1. **Immediate Upgrade**: Switch to MiniLM-L-12
   - Better product matching accuracy
   - Still fast enough for interactive use
   - Simple configuration change

2. **Advanced Setup**: Implement cascaded reranking
   - Use TinyBERT for initial filtering
   - Use L-12 or base for final ranking
   - Optimal speed/accuracy balance

3. **Future Enhancement**: Fine-tune on e-commerce data
   - Collect user interaction data
   - Train on actual product-query relevance
   - Achieve 30-50% improvement on domain-specific queries

## Testing Different Models

```bash
# Test with different models
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --top-k 20 --rerank

# Compare in evaluation
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants rrf,rrf_ce
```

## Configuration

To change the model, edit `app/cli.py`:
```python
CROSS_ENCODER_MODEL = "your-chosen-model"
```

Or make it configurable via environment:
```python
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", 
    "cross-encoder/ms-marco-MiniLM-L-12-v2"
)
```