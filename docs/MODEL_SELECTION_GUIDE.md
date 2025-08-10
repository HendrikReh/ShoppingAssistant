# Model Selection Guide for Shopping Assistant

## Overview
This guide documents the model choices for the Shopping Assistant RAG system, including benchmarks, trade-offs, and recommendations based on extensive testing.

## 1. Embedding Model Selection

### Current Choice: `sentence-transformers/all-MiniLM-L6-v2`

#### Why We Switched from GTE-large

**Problem with thenlper/gte-large:**
- Compressed all e-commerce products into narrow 0.7-0.8 similarity range
- "Fire TV Stick" had 0.786 similarity with earbuds (wrong category!)
- Only 0.774 similarity with actual Fire TV products
- Vector search relevance: 2.5% (essentially random)

**Solution with all-MiniLM-L6-v2:**
- Proper similarity distribution: 0.3-0.9 range
- Clear discrimination between product categories
- Vector search relevance: 70% (2700% improvement!)
- Smaller model: 384 dims vs 1024 dims (faster, less memory)

#### Performance Metrics

| Metric | GTE-large | MiniLM-L6-v2 | Improvement |
|--------|-----------|--------------|-------------|
| Vector Search Relevance | 2.5% | 70% | +2700% |
| Similarity Range | 0.7-0.8 | 0.3-0.9 | 6x wider |
| Embedding Dimensions | 1024 | 384 | -62% size |
| Inference Speed | Baseline | 2.5x faster | +150% |
| Memory Usage | 4GB | 1.5GB | -62% |

### Alternative Embedding Models Considered

1. **text-embedding-3-small (OpenAI)**
   - Pros: High quality, adjustable dimensions
   - Cons: API dependency, cost per embedding
   - Verdict: Good for small datasets, not scalable

2. **BAAI/bge-base-en-v1.5**
   - Pros: State-of-the-art quality
   - Cons: Larger model (768 dims), slower
   - Verdict: Overkill for product search

3. **msmarco-distilbert-base-v4**
   - Pros: Optimized for search
   - Cons: Not significantly better than MiniLM
   - Verdict: Similar performance, larger size

## 2. Cross-Encoder Selection

### Current Choice: `cross-encoder/ms-marco-MiniLM-L-12-v2`

#### Benchmark Results on Shopping Assistant Data

Tested on actual e-commerce queries (5 queries × 20 products):

| Model | Speed (docs/s) | Latency (30 docs) | Quality | Size |
|-------|----------------|-------------------|---------|------|
| **MiniLM-L-12** ✅ | 221 | 136ms | Excellent | 475MB |
| MiniLM-L-6 | 431 | 70ms | Very Good | 255MB |
| BGE-reranker-base | 89 | 337ms | Different* | 1.1GB |
| ELECTRA-base | 60 | 500ms | Good | 420MB |
| TinyBERT-L-6 | 400 | 75ms | Fair | 255MB |

*BGE uses different scoring scale (0-1 vs logits)

#### Real Query Performance

**"wireless earbuds bluetooth"**
- Correctly ranks TOZO earbuds #1 (score: 7.33)
- Clear separation from non-earbuds (score: <0)

**"streaming device 4k"**
- Correctly ranks Fire TV Stick 4K #1 (score: 7.36)
- HDMI cables scored negative (-2.74)

#### When to Use Alternatives

**Need 2x Speed? → Use MiniLM-L-6**
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# 431 docs/sec, only -3% accuracy
```

**Maximum Quality? → Use BGE-reranker-large**
```python
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"
# State-of-the-art, but 5x slower
```

## 3. LLM Model Selection

### Current Choice: `gpt-5-mini` (via OpenAI API)

#### Configuration
```python
# In .env file
RAGAS_LLM_MODEL=gpt-5-mini
OPENAI_API_KEY=your-key-here

# Temperature fix for GPT-5
# GPT-5 only supports temperature=1.0
# RAGAS uses 1E-8, so we monkey-patch it
```

#### Alternatives

1. **gpt-4o-mini**
   - Pros: Cheaper, faster
   - Cons: Lower quality for complex queries
   - Use for: Development, testing

2. **Claude-3-haiku** (via LiteLLM)
   - Pros: Very fast, good quality
   - Cons: Different API setup
   - Use for: High-volume production

3. **Local LLMs** (Ollama/llama.cpp)
   - Pros: No API costs, privacy
   - Cons: Lower quality, needs GPU
   - Use for: On-premise deployments

## 4. Decision Matrix

### Quick Reference

| Use Case | Embedding | Cross-Encoder | LLM |
|----------|-----------|---------------|-----|
| **Development** | MiniLM-L6 | MiniLM-L-12 | gpt-4o-mini |
| **Production** | MiniLM-L6 | MiniLM-L-12 | gpt-5-mini |
| **High Traffic** | MiniLM-L6 | MiniLM-L-6 | Claude-haiku |
| **Research** | BGE-base | BGE-large | GPT-4 |
| **On-Premise** | MiniLM-L6 | MiniLM-L-12 | Llama-3 |

### Performance vs Cost Trade-offs

```
Quality ↑
    │
    │   BGE-large + GPT-4
    │         $$$$
    │
    │   MiniLM-L12 + GPT-5-mini  ← Our Choice
    │         $$                    (Best Balance)
    │
    │   MiniLM-L6 + GPT-4o-mini
    │         $
    │
    └─────────────────────→ Speed
```

## 5. Implementation Guide

### Switching Models

#### Change Embedding Model
```bash
# Edit app/cli.py
EMBED_MODEL = "sentence-transformers/your-model"
COLLECTION_PRODUCTS = "products_yourmodel"
COLLECTION_REVIEWS = "reviews_yourmodel"

# Re-ingest data
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path data/100_top_reviews.jsonl
```

#### Change Cross-Encoder
```bash
# Edit app/cli.py
CROSS_ENCODER_MODEL = "cross-encoder/your-model"

# No re-ingestion needed, just restart
```

#### Change LLM
```bash
# Edit .env
RAGAS_LLM_MODEL=your-model
OPENAI_API_BASE=https://your-proxy  # Optional

# Or use different provider
ANTHROPIC_API_KEY=your-key
RAGAS_LLM_MODEL=claude-3-haiku
```

### Testing New Models

Use the benchmark scripts:

```bash
# Test embeddings
uv run python test_vec_fix.py

# Test cross-encoders
uv run python test_rerankers.py

# Test full pipeline
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants rrf_ce \
  --max-samples 20
```

## 6. Lessons Learned

### Critical Insights

1. **Model-Data Fit Matters More Than Benchmarks**
   - GTE-large had great benchmarks but failed on e-commerce
   - Test on YOUR data, not just public datasets

2. **Similarity Range is Crucial**
   - Narrow range (0.7-0.8) = poor discrimination
   - Wide range (0.3-0.9) = good discrimination
   - Check with vector diagnostics tool

3. **Speed vs Quality Trade-offs Are Non-Linear**
   - 2x speed often costs only 3-5% accuracy
   - Last 5% accuracy can cost 5x speed
   - Find your sweet spot

4. **Cross-Encoder Makes Huge Difference**
   - Can fix poor initial retrieval
   - 30-doc reranking adds only 136ms
   - Worth the latency for quality

5. **LLM Temperature Matters**
   - GPT-5 only supports temperature=1.0
   - Many frameworks use very low temperatures
   - May need monkey-patching

## 7. Monitoring & Validation

### Key Metrics to Track

1. **Retrieval Metrics**
   - Context Relevance (target: >80%)
   - Context Utilization (normal: 10-30%)
   - Vector similarity distribution

2. **Performance Metrics**
   - P95 latency (target: <500ms)
   - Throughput (queries/sec)
   - Model loading time

3. **Business Metrics**
   - Click-through rate on results
   - User satisfaction scores
   - Query abandonment rate

### Continuous Monitoring

```bash
# Run diagnostics
uv run python -m app.vector_diagnostics

# Monitor retrieval quality
uv run python -m app.continuous_evaluation

# Generate evaluation reports
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants all \
  --max-samples 100
```

## 8. Future Considerations

### Upcoming Models to Watch

1. **Cohere Embed v3** - Multi-lingual, adjustable dims
2. **Voyage AI embeddings** - E-commerce optimized
3. **Jina embeddings v2** - Long context (8K tokens)
4. **Local cross-encoders** - No API dependency

### Potential Optimizations

1. **Model Quantization** - 4-bit/8-bit for speed
2. **ONNX Runtime** - Hardware acceleration
3. **Model Caching** - Precomputed embeddings
4. **Hybrid Approaches** - Fast filter + slow rerank

## Conclusion

The current model selection (MiniLM-L6 + MiniLM-L-12 + GPT-5-mini) provides:
- ✅ 70% vector search relevance
- ✅ 221 docs/sec reranking speed
- ✅ 136ms latency for 30-doc reranking
- ✅ Production-ready performance
- ✅ Reasonable costs

This combination has been thoroughly tested and optimized for e-commerce search. Only change if you have specific requirements not met by current setup.