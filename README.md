# Shopping Assistant

An advanced e-commerce search and recommendation system that combines data analysis, retrieval augmented generation (RAG), and machine learning capabilities to provide intelligent search, ranking, and business insights.

## Features

- **Semantic Search**: Vector search using Qdrant with gte-large embeddings (1024-dim)
- **Hybrid Retrieval**: Combines BM25 keyword search with semantic search using Reciprocal Rank Fusion (RRF)
- **Cross-encoder Reranking**: Uses ms-marco-MiniLM-L-12-v2 for improved result reranking
- **Interactive Modes**: Three interactive modes for seamless user experience
  - 🔍 **Interactive Search**: Continuous product searching with built-in help
  - 💬 **Interactive Chat**: Natural conversations with the shopping assistant
  - 🛍️ **Combined Mode**: Switch between search and chat seamlessly
- **Visual CLI**: Colored outputs, emojis, and progress indicators for better UX
- **Interactive Analysis**: Marimo notebooks for data exploration and visualization
- **Command-Line Interface**: Typer-based CLI for all core functionality
- **Evaluation Framework**: RAGAS metrics for search and chat quality assessment
- **Flexible LLM Configuration**: Centralized config supporting OpenAI and LiteLLM proxy
- **Caching Layer**: Redis for performance optimization with LRU eviction
- **ML Tracking**: MLflow for experiment tracking and model management
- **Web Integration**: Tavily for external web search capabilities

## Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- uv (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ShoppingAssistant
```

2. Install dependencies:
```bash
uv sync
```

3. Start infrastructure services:
```bash
docker-compose up -d
```

This starts:
- **Qdrant** on port 6333 (vector database)
- **Redis** on port 6379 (caching, 2GB memory limit)

### Try It Out!

After setup, try the interactive mode:
```bash
# Start the interactive shopping assistant
uv run python -m app.cli interactive

# Or jump directly into search or chat
uv run python -m app.cli search    # Interactive search
uv run python -m app.cli chat      # Interactive chat
```

### Configuration

1. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

Or use LiteLLM proxy:
```bash
export OPENAI_API_BASE="http://localhost:4000/v1"
export OPENAI_API_KEY="your-proxy-key"
```

2. Ingest data into vector database:
```bash
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path data/100_top_reviews_of_the_top_1000_products.jsonl
```

## Command-Line Interface

### Interactive Mode (NEW!)
```bash
# Combined interactive mode - switch between search and chat
uv run python -m app.cli interactive
```

### Search
```bash
# Interactive search mode - continuous searching
uv run python -m app.cli search

# Single query search
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --top-k 20 --rrf-k 60 --rerank
```

### Chat
```bash
# Interactive chat mode - have a conversation
uv run python -m app.cli chat

# Single question
uv run python -m app.cli chat \
  --question "What are the best budget earbuds?"
```

### Interactive Features
- **Built-in Commands**: `help`, `settings`, `context`, `clear`, `exit`
- **Color-Coded Results**: Green (high relevance), Yellow (medium), White (low)
- **Persistent Sessions**: Data loaded once for faster responses
- **Visual Feedback**: Thinking indicators, progress bars, formatted outputs

### Evaluation
```bash
# Evaluate search quality
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --top-k 20 --variants bm25,vec,rrf,rrf_ce

# Evaluate chat quality
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --top-k 8 --max-samples 50
```

Results are saved to `eval/results/` with timestamps (e.g., `search_20250809_174328.json`).

## Interactive Notebooks

### Data Analysis
```bash
uv run marimo run notebooks/ecommerce_analysis.py
```

### Embedding Ingestion
```bash
uv run marimo run notebooks/ingest_embeddings.py
```

### Retrieval Fusion
```bash
uv run marimo run notebooks/retrieval_fusion.py
```

Edit notebooks interactively:
```bash
uv run marimo edit notebooks/<notebook_name>.py
```

## Data

The project uses Amazon Reviews dataset (2023):
- **Products**: `data/top_1000_products.jsonl` - Top 1000 electronics with ratings and review counts
- **Reviews**: `data/100_top_reviews_of_the_top_1000_products.jsonl` - Top 100 reviews per product

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

## Development

### Code Quality

Format code:
```bash
uv run black .
```

Lint code:
```bash
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues
```

### Testing Components

```bash
# Test vector search
uv run python test_vec_fix.py

# Test LLM configuration
uv run python test_llm_config.py

# Test RAGAS integration
uv run python test_ragas_fix.py
```

## Project Structure

```
ShoppingAssistant/
├── app/
│   ├── cli.py         # Typer CLI implementation
│   └── llm_config.py  # Centralized LLM configuration
├── notebooks/         # Interactive Marimo notebooks
│   ├── ecommerce_analysis.py
│   ├── ingest_embeddings.py
│   └── retrieval_fusion.py
├── data/             # Dataset files (JSONL format)
│   ├── top_1000_products.jsonl
│   └── 100_top_reviews_of_the_top_1000_products.jsonl
├── eval/
│   ├── datasets/     # Evaluation datasets
│   │   ├── search_eval.jsonl
│   │   └── chat_eval.jsonl
│   └── results/      # Evaluation results
├── docs/             # Documentation
├── docker-compose.yml
└── pyproject.toml
```

## Technology Stack

- **Data Processing**: Polars (fast dataframe operations)
- **RAG Framework**: DSPy (retrieval-augmented generation)
- **Vector Search**: Qdrant (semantic search)
- **Keyword Search**: BM25 (traditional text matching)
- **Embeddings**: Sentence Transformers (thenlper/gte-large, 1024-dim)
- **Reranking**: Cross-encoder (ms-marco-MiniLM-L-12-v2)
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **CLI**: Typer (command-line interface)
- **Caching**: Redis (LRU eviction, 2GB limit)
- **ML Tracking**: MLflow
- **Evaluation**: RAGAS (context relevance, faithfulness, etc.)
- **Visualization**: Plotly
- **Notebooks**: Marimo (interactive Python)
- **Web API**: FastAPI
- **Package Management**: uv (fast Python package manager)

## Documentation

- [CLI Reference](docs/CLI.md) - Complete command-line interface documentation
- [RAG Evaluation Insights](docs/RAG_EVALUATION_INSIGHTS.md) - Understanding evaluation metrics and results
- [Cross-Encoder Models Guide](docs/RERANKING_MODELS.md) - Choosing and configuring reranking models

## Known Issues & Fixes

- **Vector Search ID Mapping**: Fixed - Qdrant UUIDs now properly map to original IDs
- **RAGAS Score Extraction**: Fixed - Handles list-based score format
- **LiteLLM Dependencies**: All required packages included (apscheduler, cryptography, python-multipart, email-validator)
- **Timestamp Format**: Evaluation results use human-readable format (YYYYMMDD_HHMMSS)
- **Low Context Utilization**: This is normal for RAG systems (see [documentation](docs/RAG_EVALUATION_INSIGHTS.md))

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]