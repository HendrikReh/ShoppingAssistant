# Evaluation Sample Size Guide: Benefits, Costs & Trade-offs

## Executive Summary

Choosing the right sample size for evaluation is crucial for balancing statistical reliability with time and cost constraints. This guide provides detailed analysis of benefits and costs at different sample sizes.

## 📊 Statistical Confidence by Sample Size

### Confidence Intervals for Metrics (95% confidence level)

| Sample Size | Margin of Error | Confidence in Results | Statistical Power |
|------------|-----------------|----------------------|-------------------|
| 10         | ±31%           | Very Low             | ~20%             |
| 20         | ±22%           | Low                  | ~35%             |
| 50         | ±14%           | Moderate             | ~60%             |
| 100        | ±10%           | Good                 | ~80%             |
| 200        | ±7%            | High                 | ~90%             |
| 500        | ±4.5%          | Very High            | ~95%             |
| 1000       | ±3.2%          | Excellent            | ~99%             |

**Example**: With 100 samples showing 85% relevance, true relevance is between 75-95% (±10%)

## 💰 API Cost Analysis

### Cost Breakdown per Sample

Each evaluation sample requires:
- **2 RAGAS metric calls** (context_relevance + context_utilization)
- **Each metric call** includes context analysis (~1000-2000 tokens)

### GPT-5-mini Pricing (as of 2025)
- Input: $0.0001 per 1K tokens
- Output: $0.0003 per 1K tokens
- Average per evaluation: ~3K tokens input, 0.5K tokens output

### Total API Costs by Sample Size

| Sample Size | API Calls | Token Usage | Estimated Cost | Cost Range |
|------------|-----------|-------------|----------------|------------|
| 10         | 20        | ~35K        | $0.005         | $0.003-0.008 |
| 20         | 40        | ~70K        | $0.01          | $0.006-0.015 |
| 50         | 100       | ~175K       | $0.025         | $0.015-0.04 |
| 100        | 200       | ~350K       | $0.05          | $0.03-0.08 |
| 200        | 400       | ~700K       | $0.10          | $0.06-0.16 |
| 500        | 1000      | ~1.75M      | $0.25          | $0.15-0.40 |
| 1000       | 2000      | ~3.5M       | $0.50          | $0.30-0.80 |
| 5000       | 10000     | ~17.5M      | $2.50          | $1.50-4.00 |

**Note**: Costs vary based on query complexity and context length

## ⏱️ Runtime Performance

### Execution Time Breakdown

Runtime includes:
1. **Data loading**: 1-5 seconds
2. **Retrieval per query**: 0.5-1 second
3. **RAGAS evaluation**: 1-2 seconds per sample
4. **Report generation**: 2-10 seconds

### Total Runtime by Sample Size

| Sample Size | Retrieval Time | Evaluation Time | Total Runtime | Parallelization Benefit |
|------------|---------------|-----------------|---------------|------------------------|
| 10         | ~10s          | ~15s            | **30s**       | Minimal                |
| 20         | ~20s          | ~30s            | **1 min**     | Minimal                |
| 50         | ~50s          | ~75s            | **2-3 min**   | Moderate               |
| 100        | ~100s         | ~150s           | **4-5 min**   | Significant            |
| 200        | ~200s         | ~300s           | **8-10 min**  | Significant            |
| 500        | ~500s         | ~750s           | **20-25 min** | Critical               |
| 1000       | ~1000s        | ~1500s          | **40-50 min** | Critical               |

**Performance Tips**:
- Use `--max-workers 10` for parallel processing
- Pre-load models to avoid repeated downloads
- Use caching for repeated evaluations

## 📈 Benefits by Sample Size

### 10-20 Samples: Quick Validation
**Benefits**:
- ✅ Immediate feedback (< 1 minute)
- ✅ Minimal cost (< $0.02)
- ✅ Good for smoke testing
- ✅ Catches major breaks

**Limitations**:
- ❌ High variance (±22-31% error)
- ❌ May miss edge cases
- ❌ Not representative of all query types
- ❌ Cannot detect subtle regressions

**Use Cases**:
```bash
# Quick development check
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --max-samples 20 \
  --variants rrf_ce
```

### 50-100 Samples: Standard Testing
**Benefits**:
- ✅ Reasonable confidence (±10-14% error)
- ✅ Covers main query patterns
- ✅ Acceptable runtime (3-5 minutes)
- ✅ Cost-effective ($0.05-0.10)
- ✅ Detects significant changes

**Limitations**:
- ⚠️ May miss rare query types
- ⚠️ Limited statistical power for A/B testing

**Use Cases**:
```bash
# PR validation
uv run python -m app.cli eval-search \
  --dataset eval/datasets/realistic_catalog_100_*.jsonl \
  --max-samples 100 \
  --variants bm25,vec,rrf,rrf_ce
```

### 200-500 Samples: Comprehensive Evaluation
**Benefits**:
- ✅ High confidence (±4.5-7% error)
- ✅ Statistically significant results
- ✅ Covers edge cases
- ✅ Reliable for decision-making
- ✅ Detects small improvements (>5%)

**Trade-offs**:
- ⚠️ Moderate runtime (10-25 minutes)
- ⚠️ Noticeable cost ($0.10-0.25)
- ⚠️ Requires planning

**Use Cases**:
```bash
# Pre-production validation
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --max-samples 500 \
  --variants rrf_ce
```

### 1000+ Samples: Research Quality
**Benefits**:
- ✅ Excellent confidence (±3.2% error)
- ✅ Publication-ready results
- ✅ Complete query coverage
- ✅ Detects tiny improvements (>2%)
- ✅ Robust A/B testing

**Trade-offs**:
- ❌ Long runtime (40-50+ minutes)
- ❌ Higher cost ($0.50+)
- ❌ Requires dedicated resources

**Use Cases**:
```bash
# Benchmark establishment
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_1000_*.jsonl \
  --max-samples 1000 \
  --variants all
```

## 🎯 Query Coverage Analysis

### Coverage by Sample Size

| Sample Size | Query Types Covered | Edge Cases | Confidence in Coverage |
|------------|-------------------|------------|----------------------|
| 10         | 2-3 types         | None       | 20%                  |
| 20         | 3-4 types         | Few        | 40%                  |
| 50         | 4-5 types         | Some       | 65%                  |
| 100        | 5-6 types         | Many       | 80%                  |
| 200        | 6-7 types         | Most       | 90%                  |
| 500        | All types         | Extensive  | 95%                  |
| 1000       | All + variations  | Complete   | 99%                  |

### Query Type Distribution
- **Simple lookups**: 30% (e.g., "wireless mouse")
- **Feature-based**: 25% (e.g., "bluetooth headphones noise cancelling")
- **Comparisons**: 15% (e.g., "iPad vs Samsung tablet")
- **Recommendations**: 15% (e.g., "best budget laptop")
- **Problem-solving**: 10% (e.g., "fix slow computer")
- **Complex/Multi-hop**: 5% (e.g., "gaming laptop under $1000 with RTX graphics")

## 📊 Decision Matrix

### Quick Reference Table

| Scenario | Recommended Size | Runtime | Cost | Confidence |
|----------|-----------------|---------|------|------------|
| Dev testing | 10-20 | < 1 min | < $0.02 | Low |
| Daily CI/CD | 50 | 2-3 min | $0.03 | Moderate |
| PR validation | 100 | 4-5 min | $0.05 | Good |
| Feature launch | 200 | 8-10 min | $0.10 | High |
| Major release | 500 | 20-25 min | $0.25 | Very High |
| Benchmarking | 1000+ | 40-50 min | $0.50+ | Excellent |

## 💡 Cost Optimization Strategies

### 1. Progressive Evaluation
```bash
# Start small
eval_samples="10 50 100"
for n in $eval_samples; do
  uv run python -m app.cli eval-search \
    --max-samples $n \
    --variants rrf_ce
  
  # Check if metrics are acceptable
  # Break if issues found early
done
```

### 2. Focused Testing
```bash
# Test only changed components
# If only vector search changed:
uv run python -m app.cli eval-search \
  --max-samples 100 \
  --variants vec  # Only test vector variant
```

### 3. Synthetic Data Generation
```bash
# Generate once, use multiple times
uv run python -m app.cli generate-testset \
  --num-samples 1000 \
  --distribution balanced

# Reuse for multiple evaluations
```

### 4. Sampling Strategies
```python
# Stratified sampling for better coverage
# Instead of random 100, take:
# - 20 simple queries
# - 20 feature-based
# - 20 comparisons
# - 20 recommendations
# - 20 edge cases
```

## 📈 ROI Analysis

### Return on Investment by Sample Size

| Size | Investment | Detection Capability | ROI Rating |
|------|------------|---------------------|------------|
| 20 | $0.01 + 1min | Major bugs only | ⭐⭐ |
| 100 | $0.05 + 5min | 10% regressions | ⭐⭐⭐⭐ |
| 500 | $0.25 + 25min | 5% regressions | ⭐⭐⭐⭐⭐ |
| 1000 | $0.50 + 50min | 2% regressions | ⭐⭐⭐ |

**Best ROI**: 100-200 samples for most use cases

## 🚀 Recommendations

### Development Phase
- **Daily work**: 10-20 samples
- **Feature complete**: 50-100 samples
- **Before PR**: 100 samples

### Production Pipeline
- **CI/CD checks**: 50 samples
- **Staging deployment**: 200 samples
- **Production release**: 500 samples

### Research & Benchmarking
- **A/B testing**: 500+ samples
- **Paper submission**: 1000+ samples
- **Public benchmarks**: 5000+ samples

## 📝 Example Commands

### Quick Test (30 seconds, $0.005)
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --max-samples 10 \
  --variants rrf_ce
```

### Standard Evaluation (5 minutes, $0.05)
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/realistic_catalog_100_*.jsonl \
  --max-samples 100 \
  --variants bm25,vec,rrf,rrf_ce
```

### Comprehensive Test (25 minutes, $0.25)
```bash
uv run python -m app.cli generate-testset --num-samples 500
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --max-samples 500 \
  --variants rrf_ce \
  --enhanced
```

### Maximum Coverage (50 minutes, $0.50)
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_1000_*.jsonl \
  --max-samples 1000 \
  --variants all
```

## 🎯 Key Takeaways

1. **Start with 100 samples** - Best balance of cost, time, and confidence
2. **Use 500+ samples** only for production decisions
3. **10-20 samples** are only for quick smoke tests
4. **Progressive evaluation** saves time and money
5. **Statistical significance** requires at least 50 samples
6. **Edge cases** need 200+ samples to surface reliably

Remember: The cost of a missed bug in production far exceeds the cost of proper evaluation!