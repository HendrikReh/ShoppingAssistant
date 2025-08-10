# Evaluation Guide for Shopping Assistant RAG System

## Overview

The Shopping Assistant includes a comprehensive evaluation framework that combines synthetic test data generation with RAGAS-based metrics to assess both retrieval and generation quality. This guide covers test data generation, evaluation execution, and result interpretation.

## Table of Contents

1. [Synthetic Test Data Generation](#synthetic-test-data-generation)
2. [Evaluation Commands](#evaluation-commands)
3. [Metrics and Interpretation](#metrics-and-interpretation)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Synthetic Test Data Generation

### Overview

The system includes a powerful synthetic test data generator (`app/testset_generator.py`) that creates diverse, realistic queries from your actual product and review data. This enables comprehensive evaluation without manual test set creation.

### Key Features

- **10x Test Coverage**: Generate 500+ test samples vs. ~50 manual samples
- **7 Query Types**: Covering different complexity levels and user intents
- **4 Distribution Presets**: Balanced, simple, complex, and mixed
- **Entity Extraction**: Uses actual products, brands, and features from your data
- **Reference Answers**: Includes ground truth for evaluation metrics
- **RAGAS Compatible**: Output format works directly with evaluation pipeline

### Query Types and Examples

#### 1. Single-Hop Factual (25% in balanced)
Simple lookups requiring information from a single source.
- "What is the best laptop for gaming?"
- "Show me Sony headphones"
- "Find keyboards with RGB lighting"

#### 2. Multi-Hop Reasoning (20% in balanced)
Complex queries requiring information synthesis across multiple documents.
- "Compare Apple and Samsung tablets based on user reviews"
- "Which mouse has the best wireless performance according to reviews?"
- "What are the pros and cons of mechanical keyboards based on user experiences?"

#### 3. Abstract/Interpretive (10% in balanced)
Questions requiring interpretation and analysis.
- "How has laptop technology evolved in recent products?"
- "Why do users prefer Bose over competitors?"
- "What makes a good gaming mouse?"

#### 4. Comparative (15% in balanced)
Queries involving comparisons and rankings.
- "Headphones under $200 with good reviews"
- "Best value laptop with long battery life"
- "Most reliable Dell keyboards"

#### 5. Recommendation (15% in balanced)
Personalized suggestion requests.
- "Recommend a webcam for remote meetings"
- "What speaker should I buy for outdoor activities?"
- "Suggest alternatives to Apple AirPods"

#### 6. Technical (10% in balanced)
Specification and feature queries.
- "What are the specifications of Logitech mouse?"
- "Does Samsung tablet support fast charging?"
- "Battery life of headphones with noise cancelling"

#### 7. Problem-Solving (5% in balanced)
Troubleshooting and decision-making queries.
- "How to choose between laptop options?"
- "Common issues with wireless earbuds"
- "Is RGB lighting worth it in keyboards?"

### Generation Command

```bash
# Basic usage - generate 500 balanced samples
uv run python -m app.cli generate-testset --num-samples 500

# Generate with different distributions
uv run python -m app.cli generate-testset \
  --num-samples 200 \
  --distribution-preset complex \
  --output-name stress_test

# Available distribution presets:
# - balanced: Equal distribution across all query types
# - simple: 50% single-hop, 25% technical, focus on straightforward queries
# - complex: 35% multi-hop, 20% abstract, focus on reasoning tasks
# - mixed: Realistic production distribution
```

### Distribution Presets

#### Balanced Distribution
```
single_hop_factual:    25%
multi_hop_reasoning:   20%
abstract_interpretive: 10%
comparative:           15%
recommendation:        15%
technical:             10%
problem_solving:        5%
```

#### Simple Distribution
```
single_hop_factual:    50%
technical:             25%
comparative:           15%
recommendation:        10%
multi_hop_reasoning:    0%
abstract_interpretive:  0%
problem_solving:        0%
```

#### Complex Distribution
```
single_hop_factual:    10%
multi_hop_reasoning:   35%
abstract_interpretive: 20%
comparative:           10%
recommendation:        10%
technical:              5%
problem_solving:       10%
```

#### Mixed Distribution (Production-like)
```
single_hop_factual:    30%
multi_hop_reasoning:   25%
comparative:           20%
recommendation:        15%
technical:              5%
abstract_interpretive:  3%
problem_solving:        2%
```

### Output Format

Generated datasets are saved in JSONL format with:
- First line: Metadata about the dataset
- Subsequent lines: Test samples

Sample structure:
```json
{
  "question": "What are the best wireless earbuds under $200?",
  "query": "What are the best wireless earbuds under $200?",
  "query_id": "a3f28b91",
  "metadata": {
    "query_type": "comparative",
    "complexity": "moderate",
    "requires_context": ["products", "reviews"],
    "category": "headphones",
    "price_limit": "200"
  },
  "reference_answer": "When comparing options, consider factors like...",
  "ground_truth": "When comparing options, consider factors like...",
  "expected_context_type": "both"
}
```

### Dataset Statistics

After generation, you'll see statistics like:
```
📈 Dataset Statistics:
  Complexity Distribution:
    simple      142 (28.4%)
    moderate    202 (40.4%)
    complex     156 (31.2%)

  Query Types Generated:
    single_hop_factual      125 (25.0%)
    multi_hop_reasoning     100 (20.0%)
    comparative              75 (15.0%)
    ...
```

---

## Evaluation Commands

### Search Evaluation

Evaluates retrieval quality across different search strategies.

```bash
# Evaluate with synthetic data
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --top-k 20 \
  --rrf-k 60 \
  --rerank-top-k 30 \
  --variants bm25,vec,rrf,rrf_ce \
  --max-samples 500

# Quick evaluation with subset
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --variants rrf,rrf_ce \
  --max-samples 100
```

**Variants:**
- `bm25`: Keyword-based BM25 search
- `vec`: Pure vector/semantic search
- `rrf`: Reciprocal Rank Fusion (BM25 + vectors)
- `rrf_ce`: RRF with cross-encoder reranking

### Chat Evaluation

Evaluates end-to-end chat response quality.

```bash
# Evaluate chat responses
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --top-k 8 \
  --max-samples 50

# Full evaluation with more context
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --top-k 12 \
  --max-samples 200
```

---

## Metrics and Interpretation

### Search Metrics

#### Context Relevance (0-1)
Measures how relevant retrieved contexts are to the query.
- **Good**: > 0.8
- **Acceptable**: 0.6-0.8
- **Poor**: < 0.6

#### Context Utilization (0-1)
Percentage of retrieved context used in the answer.
- **Normal for e-commerce**: 9-27%
- **Not a quality indicator**: Low utilization means comprehensive retrieval
- **See**: [RAG_EVALUATION_INSIGHTS.md](./RAG_EVALUATION_INSIGHTS.md)

### Chat Metrics

#### Faithfulness (0-1)
Whether the answer is grounded in retrieved contexts.
- **Good**: > 0.85
- **Acceptable**: 0.7-0.85
- **Poor**: < 0.7 (indicates hallucination)

#### Answer Relevancy (0-1)
How well the answer addresses the question.
- **Good**: > 0.8
- **Acceptable**: 0.65-0.8
- **Poor**: < 0.65

#### Context Precision (0-1)
Ranking quality of retrieved contexts.
- **Good**: > 0.75
- **Acceptable**: 0.6-0.75
- **Poor**: < 0.6

#### Context Recall (0-1)
Coverage of required information in contexts.
- **Good**: > 0.8
- **Acceptable**: 0.65-0.8
- **Poor**: < 0.65

### Expected Performance by Query Type

Different query types have different expected performance:

| Query Type | Expected Context Relevance | Expected Faithfulness | Expected Utilization |
|------------|---------------------------|----------------------|---------------------|
| Single-hop factual | 0.85-0.95 | 0.90-0.95 | 20-30% |
| Multi-hop reasoning | 0.70-0.85 | 0.75-0.85 | 10-20% |
| Abstract/interpretive | 0.65-0.80 | 0.70-0.80 | 5-15% |
| Comparative | 0.75-0.90 | 0.80-0.90 | 15-25% |
| Recommendation | 0.70-0.85 | 0.75-0.85 | 10-20% |
| Technical | 0.85-0.95 | 0.90-0.95 | 25-35% |
| Problem-solving | 0.65-0.80 | 0.70-0.80 | 5-15% |

---

## Best Practices

### 1. Test Data Generation

**Start with balanced distribution:**
```bash
# Initial comprehensive test
uv run python -m app.cli generate-testset \
  --num-samples 500 \
  --distribution-preset balanced
```

**Then test edge cases:**
```bash
# Stress test with complex queries
uv run python -m app.cli generate-testset \
  --num-samples 200 \
  --distribution-preset complex \
  --output-name stress_test
```

### 2. Incremental Evaluation

Start small and scale up:
```bash
# Quick smoke test (5 minutes)
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_*.jsonl \
  --max-samples 20 \
  --variants rrf_ce

# Full evaluation (30+ minutes)
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --max-samples 500 \
  --variants bm25,vec,rrf,rrf_ce
```

### 3. Regression Testing

After changes, compare with baseline:
```bash
# Save baseline
cp eval/results/search_20250810_*.json eval/baseline_search.json

# After changes, compare
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl

# Check for regression (>5% drop is concerning)
```

### 4. Regular Evaluation Schedule

- **Daily**: Quick smoke test with 50 samples
- **Weekly**: Full evaluation with 500 samples
- **Before deployment**: Comprehensive test with 1000+ samples

### 5. Custom Test Sets

For specific scenarios, create targeted test sets:
```bash
# Focus on complex reasoning
uv run python -m app.cli generate-testset \
  --num-samples 100 \
  --distribution-preset complex \
  --output-name reasoning_test

# Focus on simple lookups for speed testing
uv run python -m app.cli generate-testset \
  --num-samples 1000 \
  --distribution-preset simple \
  --output-name speed_test
```

---

## Troubleshooting

### Issue: Low Context Relevance (<0.6)

**Possible causes:**
- Embedding model vocabulary mismatch
- Query types not in training data
- Insufficient retrieval top-k

**Solutions:**
```bash
# Increase retrieval top-k
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_*.jsonl \
  --top-k 30  # Increase from 20

# Test different embedding models (requires code change)
```

### Issue: Low Faithfulness (<0.7)

**Possible causes:**
- LLM hallucination
- Insufficient context
- Model temperature too high

**Solutions:**
```bash
# Increase context retrieval
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_*.jsonl \
  --top-k 12  # Increase from 8

# Check LLM configuration in app/llm_config.py
# Ensure eval_temperature = 0.0 for deterministic evaluation
```

### Issue: HTTP 429 Errors (Too Many Requests)

**Possible causes:**
- Model being downloaded repeatedly from HuggingFace
- Rate limiting on model hub

**Solutions:**
- Fixed in latest version with model caching
- Models are loaded once and reused
- If still occurring, wait a few minutes for rate limit reset

### Issue: Evaluation Timeout

**Possible causes:**
- Too many samples
- Slow LLM API
- Network issues
- Model download on first run

**Solutions:**
```bash
# Reduce sample size
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_*.jsonl \
  --max-samples 20  # Start small

# Use local LLM if available
export OPENAI_API_BASE="http://localhost:4000/v1"

# Pre-download models (first run only)
# Models are cached after first download
```

### Issue: Inconsistent Results

**Possible causes:**
- Non-deterministic generation
- Different random seeds
- Data changes

**Solutions:**
```bash
# Use fixed seed for reproducibility
uv run python -m app.cli generate-testset \
  --num-samples 100 \
  --seed 42  # Always use same seed

# Set temperature to 0 for evaluation
# Check app/llm_config.py: eval_temperature = 0.0
```

---

## Output Files

### Generated Test Data
- Location: `eval/datasets/`
- Format: `{name}_{samples}_{timestamp}.jsonl`
- Metadata: `{name}_{samples}_{timestamp}_metadata.json`

### Evaluation Results
- Location: `eval/results/`
- JSON: `{command}_{timestamp}.json` (detailed metrics and raw data)
- Markdown: `{command}_{timestamp}.md` (human-readable report with interpretations)

#### Enhanced Report Features (NEW!)

Reports now include detailed interpretations:

1. **Executive Summary**: Overall system health status (Excellent/Good/Acceptable/Needs Attention)
2. **Variant Performance Comparison**: Side-by-side analysis of search strategies
3. **Metric-by-Metric Analysis**: Individual metric interpretations with context
4. **Critical Issues Identification**: Prioritized list of problems requiring attention
5. **Strategic Recommendations**: Actionable improvement plans
6. **Success Criteria Assessment**: Pass/fail status for quality thresholds
7. **Configuration Details**: Complete record of evaluation parameters

Example interpretation snippet:
```markdown
## Executive Summary
✅ **System Health: GOOD** (Average score: 78.5%)
Evaluated 500 samples across 4 search variants.

## Critical Issues & Priorities
🟡 **MEDIUM**: Context Recall at 68.0% (target: >80%)
- Increase retrieval top-k parameter
- Implement query expansion techniques

## Strategic Recommendations
1. **Improve Answer Relevancy** (Priority: HIGH)
   - Review and optimize prompt templates
   - Add explicit instructions to address all query aspects
```

### MLflow Tracking
- Location: `mlruns/`
- View: `mlflow ui`
- Tracks all evaluation runs with parameters and metrics

---

## Advanced Usage

### Custom Query Distribution

Edit `app/testset_generator.py` to add custom distributions:
```python
distributions = {
    "custom": {
        "single_hop_factual": 0.40,    # Your distribution
        "multi_hop_reasoning": 0.30,
        "comparative": 0.20,
        "recommendation": 0.10,
        # ...
    }
}
```

### Adding New Query Types

Extend the `QueryTemplate` list in `app/testset_generator.py`:
```python
QueryTemplate(
    "Your new query template with {placeholder}",
    "new_query_type",
    "complexity_level",  # simple, moderate, complex
    ["required_context"]  # products, reviews, or both
)
```

### Batch Evaluation

For multiple datasets:
```bash
for dataset in eval/datasets/synthetic_*.jsonl; do
    echo "Evaluating $dataset"
    uv run python -m app.cli eval-search \
        --dataset "$dataset" \
        --variants rrf_ce \
        --max-samples 100
done
```

### Environment Configuration

For LiteLLM proxy or custom endpoints:
```bash
# Use LiteLLM proxy
export OPENAI_API_BASE="http://localhost:4000/v1"
export OPENAI_API_KEY="your-proxy-key"

# Specify evaluation model
export EVAL_MODEL="gpt-4o-mini"

# Run evaluation
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_500_*.jsonl
```

---

## See Also

- [RAG_EVALUATION_INSIGHTS.md](./RAG_EVALUATION_INSIGHTS.md) - Understanding metrics in depth
- [CLI.md](./CLI.md) - Complete CLI command reference
- [EVALUATION_IMPROVEMENTS.md](./EVALUATION_IMPROVEMENTS.md) - Planned improvements and roadmap
- [README.md](../README.md) - Project overview

---

## Summary

The synthetic test data generation system provides:

1. **Scale**: 10x increase in test coverage (50 → 500+ samples)
2. **Diversity**: 7 query types with 3 complexity levels
3. **Realism**: Entity extraction from actual product data
4. **Flexibility**: 4 distribution presets + custom options
5. **Integration**: RAGAS-compatible format with reference answers

This enables comprehensive, reproducible evaluation of both retrieval and generation components, ensuring the Shopping Assistant maintains high quality across diverse user queries.