# Shopping Assistant

An advanced e-commerce search and recommendation system that combines data analysis, retrieval augmented generation (RAG), and machine learning capabilities to provide intelligent search, ranking, and business insights.

## Features

- **Semantic Search**: Vector search using Qdrant with all-MiniLM-L6-v2 embeddings (384-dim)
- **Hybrid Retrieval**: Combines BM25 keyword search with semantic search using Reciprocal Rank Fusion (RRF)
- **Cross-encoder Reranking**: Uses ms-marco-MiniLM-L-12-v2 for improved result reranking
- **Web Search Integration** (NEW): Real-time product information via Tavily API
  - Current prices and availability across major retailers
  - Latest professional reviews and comparisons
  - Alternative product suggestions
  - Intelligent query routing (local vs web based on intent)
  - Redis caching for performance
- **Unified Interactive Mode**: Natural language interface that automatically detects search vs chat intent
  - **Automatic Search**: Type product names to search
  - **Automatic Chat**: Ask questions for recommendations
  - **No Mode Selection**: System intelligently routes queries
- **Visual CLI**: Colored outputs and progress indicators for better UX
- **Interactive Analysis**: Marimo notebooks for data exploration and visualization
- **Command-Line Interface**: Typer-based CLI for all core functionality
- **Evaluation Framework**: RAGAS metrics for search and chat quality assessment
- **Flexible LLM Configuration**: Centralized config supporting OpenAI and LiteLLM proxy
- **Caching Layer**: Redis for performance optimization with LRU eviction
- **ML Tracking**: MLflow for experiment tracking and model management

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

After setup, try the unified interactive mode:
```bash
# Start the shopping assistant - no mode selection needed!
uv run python -m app.cli interactive

# Examples of what you can type:
# "wireless mouse"              → Triggers search
# "what's the best laptop?"     → Triggers chat
# "gaming keyboard"             → Triggers search
# "compare tablets under $500"  → Triggers chat

# Or use specific modes directly:
uv run python -m app.cli search    # Search-only mode
uv run python -m app.cli chat      # Chat-only mode
```

### Configuration

1. Set up your environment variables:

Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY='your-api-key'
TAVILY_API_KEY='your-tavily-key'  # Optional: for web search
```

The CLI automatically loads `.env` files using python-dotenv.

2. Ingest data into vector database:
```bash
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path data/100_top_reviews_of_the_top_1000_products.jsonl
```

## Command-Line Interface

### Interactive Mode (Enhanced!)
```bash
# Unified interactive mode - just type naturally!
uv run python -m app.cli interactive

# The system automatically detects your intent:
# - Product names → Search
# - Questions → Chat  
# - Type /help for tips and examples
```

### Search
```bash
# Interactive search mode - continuous searching
uv run python -m app.cli search

# Single query search
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --top-k 20 --rrf-k 60 --rerank

# Web-enhanced search (combines local + web results)
uv run python -m app.cli search \
  --query "latest macbook pro prices" \
  --web

# Web-only search (no local data)
uv run python -m app.cli search \
  --query "rtx 4090 availability" \
  --web-only
```

### Web Search Commands (NEW)
```bash
# Check current prices and availability
uv run python -m app.cli check-price "Fire TV Stick 4K"

# Find alternative products
uv run python -m app.cli find-alternatives "Apple AirPods Pro" \
  --max-results 5
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
- **Smart Intent Detection**: Automatically routes to search or chat based on input
- **Built-in Commands**: `/help` (tips), `/exit` (quit)
- **Color-Coded Results**: Green (high relevance), Yellow (medium), White (low)
- **Persistent Sessions**: Models and data loaded once for faster responses
- **Visual Feedback**: Thinking indicators, progress bars, formatted outputs
- **Execution Tracking**: All evaluation reports include the exact command used for reproducibility

### Evaluation

#### Generate Realistic Test Data (Catalog-Based)
```bash
# Generate realistic test samples based on actual product catalog
uv run python -m app.cli generate-testset \
  --num-samples 100  # Uses actual products from catalog

# Generate with different complexity distributions
uv run python -m app.cli generate-testset \
  --num-samples 200 \
  --distribution-preset complex
```

#### Run Evaluations with Optimized Parameters
```bash
# Evaluate search performance (uses simple retrieval metrics)
uv run python -m app.cli eval-search \
  --dataset eval/datasets/realistic_catalog_*.jsonl \
  --top-k 25 --rrf-k 80 \
  --variants bm25,vec,rrf,rrf_ce \
  --max-samples 50

# Evaluate chat quality (uses RAGAS metrics)
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/realistic_catalog_*.jsonl \
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
│   ├── cli.py                  # Typer CLI implementation  
│   ├── llm_config.py           # Centralized LLM configuration
│   ├── eval_interpreter.py     # Evaluation metrics interpretation
│   └── testset_generator.py    # Realistic catalog-based test data generator
├── notebooks/                  # Interactive Marimo notebooks
│   ├── ecommerce_analysis.py
│   ├── ingest_embeddings.py
│   └── retrieval_fusion.py
├── data/                       # Dataset files (JSONL format)
│   ├── top_1000_products.jsonl
│   └── 100_top_reviews_of_the_top_1000_products.jsonl
├── eval/
│   ├── datasets/               # Evaluation datasets
│   │   ├── search_eval.jsonl   # Hand-crafted search queries
│   │   ├── chat_eval.jsonl     # Hand-crafted Q&A pairs
│   │   └── realistic_*.jsonl   # Generated from actual catalog
│   └── results/                # Evaluation results with timestamps
├── docs/                       # Documentation
├── docker-compose.yml
├── .env                        # Environment variables (API keys)
└── pyproject.toml
```

## Technology Stack

- **Data Processing**: Polars (fast dataframe operations)
- **RAG Framework**: DSPy (retrieval-augmented generation)
- **Vector Search**: Qdrant (semantic search)
- **Keyword Search**: BM25 (traditional text matching)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, 384-dim)
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
- [Search Relevance Learnings](docs/SEARCH_RELEVANCE_LEARNINGS.md) - Why products vs reviews score differently

## Recent Improvements

### Latest Updates (2025-08-11)
- **Improved Product Search**: Products now prioritized over reviews with 1.5x boost
- **Query Preprocessing**: Automatically removes "search for" and similar prefixes
- **Products-Only Mode**: Filter out reviews when you only want products
- **Search Relevance Documentation**: New guide explaining products vs reviews scoring

### Previous Updates (2025-08-10)
- **Unified Interactive Mode**: No mode selection needed - just type naturally
- **Fixed Evaluation Metrics**: Search evaluation now uses appropriate retrieval metrics (no RAGAS)
- **Better Embeddings**: Switched to all-MiniLM-L6-v2 for better product discrimination
- **Smart Intent Detection**: Automatically routes queries to search or chat
- **Enhanced CLI**: Fixed parameter passing issues in interactive modes

### Previous Fixes
- **Vector Search ID Mapping**: Fixed - Qdrant UUIDs now properly map to original IDs
- **RAGAS Score Extraction**: Fixed - Handles list-based score format
- **LiteLLM Dependencies**: All required packages included
- **Timestamp Format**: Evaluation results use human-readable format (YYYYMMDD_HHMMSS)
- **Low Context Utilization**: This is normal for RAG systems (see [documentation](docs/RAG_EVALUATION_INSIGHTS.md))

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]