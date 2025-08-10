# Enhanced Evaluation Reporting Guide

## Overview
The enhanced eval-search command now provides detailed query-by-query analysis, making it much easier to debug retrieval issues and improve the system.

## What's Captured

### For Each Query:
1. **Request**: The exact search query text
2. **Response**: Complete list of retrieved items with:
   - Rank position
   - Item type (product or review)
   - Title and category
   - Rating information
   - Content snippet
3. **Metrics**: Individual scores for that query:
   - Context relevance
   - Context utilization
   - Any other configured metrics

## Report Structure

### JSON Report (`eval/results/search_*.json`)
```json
{
  "call_parameters": { ... },
  "config": { ... },
  "aggregates": { 
    "rrf_ce": {
      "context_relevance": 0.82,
      "context_utilization": 0.15
    }
  },
  "detailed_results": {
    "rrf_ce": [
      {
        "query": "mechanical keyboard quiet switches",
        "retrieved_items": [
          {
            "rank": 1,
            "type": "product",
            "title": "GREAT keyboard",
            "category": "Electronics",
            "rating": 5.0,
            "snippet": "Silent mechanical keyboard..."
          }
        ],
        "num_retrieved": 20,
        "metrics": {
          "context_relevance": 0.95,
          "context_utilization": 0.12
        }
      }
    ]
  }
}
```

### Markdown Report (`eval/results/search_*.md`)
- Includes visual samples of top queries
- Shows top 3 retrieved items per query
- Displays metrics inline with each query
- Easy to read for manual inspection

## How This Helps

### 1. Identify Retrieval Failures
```bash
# Look for queries with low relevance scores
uv run python -c "
import json
with open('eval/results/search_latest.json') as f:
    data = json.load(f)
    
for variant, results in data['detailed_results'].items():
    print(f'\\n{variant} - Low relevance queries:')
    for r in results:
        metrics = r.get('metrics', {})
        relevance = metrics.get('context_relevance', 0)
        if relevance < 0.5:
            print(f'  Query: {r[\"query\"]}')
            print(f'  Relevance: {relevance:.2f}')
            print(f'  Top result: {r[\"retrieved_items\"][0][\"title\"]}')
"
```

### 2. Analyze Query Patterns
```bash
# Find common patterns in failing queries
uv run python -c "
import json
with open('eval/results/search_latest.json') as f:
    data = json.load(f)

failed_queries = []
for variant, results in data['detailed_results'].items():
    for r in results:
        relevance = r.get('metrics', {}).get('context_relevance', 0)
        if relevance < 0.5:
            failed_queries.append(r['query'])

print('Failed query patterns:')
# Check for common words
from collections import Counter
words = ' '.join(failed_queries).lower().split()
common = Counter(words).most_common(5)
for word, count in common:
    print(f'  {word}: {count} occurrences')
"
```

### 3. Debug Specific Queries
```bash
# Examine why a specific query failed
uv run python -c "
import json
target_query = 'wireless mouse gaming'

with open('eval/results/search_latest.json') as f:
    data = json.load(f)

for variant, results in data['detailed_results'].items():
    for r in results:
        if target_query in r['query'].lower():
            print(f'Query: {r[\"query\"]}')
            print(f'Metrics: {r.get(\"metrics\", {})}')
            print('\\nRetrieved items:')
            for item in r['retrieved_items'][:5]:
                print(f'  #{item[\"rank\"]} {item[\"title\"]} (Rating: {item[\"rating\"]})')
                print(f'     Category: {item[\"category\"]}')
"
```

## Usage Examples

### Basic Evaluation with Details
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants rrf_ce \
  --top-k 10
```

### Compare Multiple Variants
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants bm25,vec,rrf,rrf_ce \
  --top-k 20
```

### Large-Scale Evaluation
```bash
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --variants rrf_ce \
  --max-samples 500 \
  --top-k 10
```

## Improvement Workflow

1. **Run Evaluation**: Generate detailed reports
2. **Identify Issues**: Find queries with poor metrics
3. **Analyze Patterns**: Look for common failure modes
4. **Debug Retrieval**: Examine what was actually retrieved
5. **Implement Fixes**: Target specific query types or retrieval issues
6. **Re-evaluate**: Verify improvements with same dataset

## Key Insights from Details

- **Relevance Mismatch**: When top results don't match query intent
- **Category Confusion**: Wrong product categories being retrieved
- **Keyword Dominance**: BM25 overweighting certain terms
- **Semantic Gaps**: Vector search missing related concepts
- **Reranking Issues**: Cross-encoder not fixing order properly

This enhanced reporting makes the entire retrieval system transparent and debuggable!