# Shopping Assistant CLI Reference

## Overview

The Shopping Assistant CLI provides a comprehensive set of commands for data ingestion, search, chat, and evaluation. All commands are accessed through the `app.cli` module using the `uv run` command.

```bash
# General syntax
uv run python -m app.cli [COMMAND] [OPTIONS]

# Get help
uv run python -m app.cli --help
uv run python -m app.cli [COMMAND] --help
```

## Commands Summary

| Command | Description | Primary Use Case |
|---------|-------------|------------------|
| `ingest` | Load products and reviews into Qdrant | Initial setup and data refresh |
| `search` | Search products using various strategies | Testing retrieval quality |
| `chat` | Interactive Q&A with RAG | User interaction and testing |
| `eval-search` | Evaluate search quality with metrics | Performance benchmarking |
| `eval-chat` | Evaluate chat responses with RAGAS | Response quality assessment |

---

## Command Details

### 1. `ingest` - Data Ingestion

Loads product and review data into Qdrant vector database with embeddings.

```bash
uv run python -m app.cli ingest [OPTIONS]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--products-path` | Path | `data/top_1000_products.jsonl` | Path to products JSONL file |
| `--reviews-path` | Path | `data/100_top_reviews_of_the_top_1000_products.jsonl` | Path to reviews JSONL file |
| `--batch-size` | int | 32 | Number of documents to process in each batch |
| `--device` | str | auto-detect | Device for embeddings: 'cuda', 'mps', or 'cpu' |

#### Details

- **Embedding Model**: Uses `thenlper/gte-large` (1024-dimensional)
- **Collections Created**: 
  - `products_gte_large`: Product embeddings
  - `reviews_gte_large`: Review embeddings
- **ID Format**: 
  - Products: `prod::{parent_asin}`
  - Reviews: `rev::{parent_asin}::{index}`
- **Payload Storage**: Full document metadata stored with vectors

#### Example Usage

```bash
# Default ingestion
uv run python -m app.cli ingest

# Custom data paths
uv run python -m app.cli ingest \
  --products-path custom/products.jsonl \
  --reviews-path custom/reviews.jsonl

# Larger batch size for faster processing
uv run python -m app.cli ingest --batch-size 64

# Force CPU usage
uv run python -m app.cli ingest --device cpu
```

#### Performance Notes

- Batch size affects memory usage and speed
- MPS (Apple Silicon) typically 2-3x faster than CPU
- CUDA (NVIDIA GPU) typically 5-10x faster than CPU
- Initial ingestion takes ~5-10 minutes for default dataset

---

### 2. `search` - Product Search

Performs hybrid search across products and reviews with multiple retrieval strategies.

```bash
uv run python -m app.cli search [OPTIONS]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--query` | str | Required | Search query text |
| `--top-k` | int | 20 | Number of results per retrieval method |
| `--rrf-k` | int | 60 | RRF fusion parameter (higher = more weight to top results) |
| `--rerank` | bool | False | Enable cross-encoder reranking |
| `--rerank-top-k` | int | 30 | Number of candidates to rerank |
| `--device` | str | auto-detect | Device for embeddings and reranking |

#### Retrieval Strategies

1. **BM25**: Traditional keyword matching
   - Fast, interpretable
   - Good for exact term matches
   
2. **Vector Search**: Semantic similarity using embeddings
   - Understands synonyms and concepts
   - Language-agnostic matching
   
3. **RRF (Reciprocal Rank Fusion)**: Combines multiple rankings
   - Formula: `score = Σ(1/(k + rank))`
   - Balances keyword and semantic signals
   
4. **Cross-Encoder Reranking**: Neural reranking of top candidates
   - Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
   - Significantly improves relevance

#### Example Usage

```bash
# Basic search
uv run python -m app.cli search --query "wireless earbuds"

# Search with more results
uv run python -m app.cli search \
  --query "laptop for programming" \
  --top-k 30

# Hybrid search with reranking (best quality)
uv run python -m app.cli search \
  --query "gaming mouse" \
  --top-k 20 \
  --rerank \
  --rerank-top-k 40

# Adjust RRF fusion parameter
uv run python -m app.cli search \
  --query "budget smartphone" \
  --rrf-k 100  # Higher k = less aggressive fusion
```

#### Output Format

```
=== Results ===
1. [prod::B09ABC123] Apple AirPods Pro (Score: 0.8534)
   Rating: 4.5 | Reviews: 15234
   
2. [rev::B09ABC123::42] "Perfect for workouts..." (Score: 0.7892)
   Rating: 5.0 | Helpful: 127
```

---

### 3. `chat` - Interactive Chat

Provides Q&A interface using RAG with retrieved context.

```bash
uv run python -m app.cli chat [OPTIONS]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--question` | str | None | Single question (non-interactive mode) |
| `--top-k` | int | 8 | Number of context documents to retrieve |
| `--model` | str | from config | LLM model name |
| `--temperature` | float | 0.0 | Response randomness (0.0-1.0) |
| `--max-tokens` | int | 1000 | Maximum response length |
| `--device` | str | auto-detect | Device for embeddings |

#### Modes

1. **Interactive Mode** (default): 
   - Continuous conversation
   - Type 'quit' or 'exit' to stop
   - Context maintained across turns

2. **Single Question Mode**:
   - Provide `--question` parameter
   - Returns answer and exits

#### Example Usage

```bash
# Interactive chat
uv run python -m app.cli chat

# Single question
uv run python -m app.cli chat \
  --question "What's the best laptop for data science?"

# More context for complex questions
uv run python -m app.cli chat \
  --question "Compare wireless earbuds under $200" \
  --top-k 15

# Creative responses
uv run python -m app.cli chat \
  --temperature 0.7 \
  --max-tokens 2000
```

#### Context Management

- Retrieves from both products and reviews
- Uses RRF to combine BM25 and vector search
- Applies cross-encoder reranking by default
- Context formatted with clear boundaries

---

### 4. `eval-search` - Search Evaluation

Evaluates search quality using RAGAS metrics across different retrieval variants.

```bash
uv run python -m app.cli eval-search [OPTIONS]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | Path | Required | Path to evaluation dataset (JSONL) |
| `--products-path` | Path | `data/top_1000_products.jsonl` | Products data |
| `--reviews-path` | Path | `data/100_top_reviews_of_the_top_1000_products.jsonl` | Reviews data |
| `--top-k` | int | 20 | Results per retrieval method |
| `--rrf-k` | int | 60 | RRF fusion parameter |
| `--rerank-top-k` | int | 30 | Candidates for reranking |
| `--variants` | str | "bm25,vec,rrf,rrf_ce" | Comma-separated variants to test |
| `--max-samples` | int | None | Limit evaluation samples |
| `--seed` | int | 42 | Random seed for sampling |
| `--device` | str | auto-detect | Computation device |

#### Variants

- `bm25`: BM25 keyword search only
- `vec`: Vector search only
- `rrf`: Reciprocal Rank Fusion (BM25 + vector)
- `rrf_ce`: RRF with cross-encoder reranking

#### Metrics

1. **Context Relevance** (0-1):
   - Measures how relevant retrieved documents are
   - Target: >0.8 for good performance

2. **Context Utilization** (0-1):
   - Percentage of context used in ideal answer
   - Normal range for e-commerce: 0.09-0.27
   - Higher isn't always better

#### Dataset Format

```jsonl
{"query": "wireless earbuds for sports", "reference_answer": "..."}
{"query": "budget gaming laptop", "reference_answer": "..."}
```

#### Example Usage

```bash
# Evaluate all variants
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl

# Test specific variants
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants "rrf,rrf_ce"

# Quick test with few samples
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --max-samples 10 \
  --variants "bm25,vec"

# Production evaluation
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --top-k 30 \
  --rerank-top-k 50 \
  --max-samples 100
```

#### Output Files

- `eval/results/search_YYYYMMDD_HHMMSS.json`: Detailed metrics
- `eval/results/search_YYYYMMDD_HHMMSS.md`: Human-readable report

---

### 5. `eval-chat` - Chat Evaluation

Evaluates chat response quality using RAGAS metrics for RAG systems.

```bash
uv run python -m app.cli eval-chat [OPTIONS]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | Path | Required | Path to Q&A evaluation dataset |
| `--top-k` | int | 8 | Context documents per question |
| `--max-samples` | int | None | Limit evaluation samples |
| `--seed` | int | 42 | Random seed for sampling |
| `--model` | str | from config | LLM for generating answers |
| `--eval-model` | str | from config | LLM for evaluation metrics |
| `--device` | str | auto-detect | Computation device |

#### Metrics

1. **Faithfulness** (0-1):
   - Whether answer is grounded in context
   - Target: >0.8

2. **Answer Relevancy** (0-1):
   - How well answer addresses the question
   - Target: >0.9

3. **Context Precision** (0-1):
   - Ranking quality of retrieved context
   - Target: >0.7

4. **Context Recall** (0-1):
   - Coverage of required information
   - Target: >0.8

#### Dataset Format

```jsonl
{
  "question": "What makes a good gaming laptop?",
  "answer": "A good gaming laptop needs...",
  "contexts": ["Context 1...", "Context 2..."]  // Optional
}
```

#### Example Usage

```bash
# Basic evaluation
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl

# Quick test
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --max-samples 5

# Full evaluation with more context
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --top-k 15 \
  --max-samples 50

# Use specific models
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --model "gpt-4o" \
  --eval-model "gpt-4o-mini"
```

#### Output Files

- `eval/results/chat_YYYYMMDD_HHMMSS.json`: Metrics and parameters
- `eval/results/chat_YYYYMMDD_HHMMSS.md`: Formatted report

---

## Environment Variables

Configure the CLI behavior using environment variables:

```bash
# OpenAI API Configuration
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="http://localhost:4000/v1"  # Optional: LiteLLM proxy

# Model Selection (optional)
export CHAT_MODEL="gpt-4o-mini"
export EVAL_MODEL="gpt-4o-mini"
export EMBED_MODEL="thenlper/gte-large"
export CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-12-v2"

# Infrastructure
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

---

## Performance Tips

### 1. Device Selection
- **CUDA**: Fastest for large batches
- **MPS**: Good for Apple Silicon Macs
- **CPU**: Universal but slower

### 2. Batch Sizes
- Ingestion: 32-64 for GPU, 8-16 for CPU
- Search: Not applicable (single query)
- Evaluation: Process in parallel when possible

### 3. Caching
- Redis caches embeddings and search results
- Speeds up repeated queries significantly
- 2GB limit with LRU eviction

### 4. Model Selection
- Larger cross-encoders = better accuracy, slower speed
- Consider ms-marco-TinyBERT-L-2 for real-time needs
- GPT-4o-mini balances cost and quality

---

## Common Workflows

### Initial Setup
```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Ingest data
uv run python -m app.cli ingest

# 3. Test search
uv run python -m app.cli search --query "test query"

# 4. Test chat
uv run python -m app.cli chat --question "test question"
```

### Development Iteration
```bash
# 1. Make code changes

# 2. Quick search test
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --rerank

# 3. Evaluate specific variant
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants "rrf_ce" \
  --max-samples 10
```

### Production Evaluation
```bash
# Full search evaluation
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --variants "bm25,vec,rrf,rrf_ce" \
  --top-k 30 \
  --max-samples 100

# Full chat evaluation
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --top-k 10 \
  --max-samples 50
```

---

## Troubleshooting

### Issue: "No API key configured"
```bash
export OPENAI_API_KEY="your-key-here"
```

### Issue: "Collection not found"
```bash
# Re-run ingestion
uv run python -m app.cli ingest
```

### Issue: Low evaluation scores
- Check if Docker services are running
- Verify data was ingested successfully
- Review metrics guide: `docs/RAG_EVALUATION_INSIGHTS.md`

### Issue: Slow performance
- Use GPU/MPS instead of CPU
- Reduce batch size if OOM
- Enable Redis caching

---

## See Also

- [RAG Evaluation Insights](./RAG_EVALUATION_INSIGHTS.md) - Understanding metrics
- [Cross-Encoder Models](./RERANKING_MODELS.md) - Model selection guide
- [README](../README.md) - Project overview