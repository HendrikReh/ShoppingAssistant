# Evaluation Datasets

This directory contains datasets for evaluating the ShoppingAssistant system.

## Datasets

### search_eval.jsonl
- **Purpose**: Evaluate retrieval quality across different search methods (BM25, vector, RRF, RRF+CE)
- **Format**: JSONL with `{"query": "search query"}` per line
- **Size**: 20 queries covering common e-commerce searches
- **Categories**: Electronics, accessories, computing devices

### chat_eval.jsonl (optional)
- **Purpose**: Evaluate chat/QA performance with DSPy RAG
- **Format**: JSONL with `{"question": "...", "reference_answer": "..."}` per line
- Can include ground truth answers for more comprehensive evaluation

## Usage

### Search Evaluation
```bash
export OPENAI_API_KEY="your-key"
uv run python -m app.cli eval-search \
    --dataset eval/datasets/search_eval.jsonl \
    --top-k 20 \
    --variants bm25,vec,rrf,rrf_ce
```

### Chat Evaluation
```bash
export OPENAI_API_KEY="your-key"
uv run python -m app.cli eval-chat \
    --dataset eval/datasets/chat_eval.jsonl \
    --top-k 8
```

## Creating New Datasets

Datasets should be in JSONL format (one JSON object per line):

```jsonl
{"query": "your search query"}
{"query": "another search query"}
```

For chat evaluation with ground truth:
```jsonl
{"question": "What are the best earbuds?", "reference_answer": "The best earbuds depend on..."}
```

## Metrics

### Search Metrics
- **Context Relevance**: How relevant are retrieved contexts to the query
- **Context Utilization**: What percentage of retrieved contexts are useful

### Chat Metrics  
- **Faithfulness**: How grounded is the answer in retrieved contexts
- **Answer Relevancy**: How relevant is the answer to the question
- **Context Precision**: Precision of retrieved contexts
- **Context Recall**: Recall of relevant information