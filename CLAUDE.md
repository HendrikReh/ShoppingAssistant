# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General instructions

During the development phase of this project I will give you examples which are stored under the folder named cohort. Most of the time I will ask you to migrate specific files from the cohort folder to the actually used tech stack of this project (check pyproject.toml).
You will never change code under the folder cohort neither will you mention its content in any documentation.

## Used Data Sets

The data stored under data/100_top_reviews_of_the_top_1000_products.jsonl and /Volumes/Halle4/projects/ShoppingAssistant/data/top_1000_products.jsonl were extracted from 'Amazon Reviews dataset, collected in 2023' (https://amazon-reviews-2023.github.io/)

## Project Overview

Shopping Assistant is an e-commerce search and recommendation system that combines:
- Data analysis with Polars for product and review datasets
- Retrieval Augmented Generation (RAG) using DSPy
- Vector search with Qdrant
- Caching with Redis
- ML experiment tracking with MLflow
- Interactive data analysis with Marimo notebooks
- Command-line interface with Typer

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Run Python commands
uv run python main.py

# Start infrastructure services
docker-compose up -d
```

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues

# Run type checking (if mypy is added)
# uv run mypy .
```

### CLI Commands

#### Ingestion
```bash
# Ingest products and reviews into Qdrant
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path data/100_top_reviews_of_the_top_1000_products.jsonl
```

#### Search
```bash
# Hybrid search with cross-encoder reranking
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --top-k 20 \
  --rrf-k 60 \
  --rerank
```

#### Chat
```bash
# Interactive chat
uv run python -m app.cli chat

# Single question
uv run python -m app.cli chat \
  --question "What are the best budget earbuds?"
```

#### Evaluation
```bash
# Evaluate search variants
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --top-k 20 \
  --variants bm25,vec,rrf,rrf_ce

# Evaluate chat with RAGAS metrics
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --top-k 8 \
  --max-samples 50
```

### Marimo Notebooks
```bash
# Run interactive notebook
uv run marimo run notebooks/ecommerce_analysis.py
uv run marimo run notebooks/ingest_embeddings.py
uv run marimo run notebooks/retrieval_fusion.py

# Edit notebook
uv run marimo edit notebooks/ecommerce_analysis.py
```

### Testing
```bash
# Test vector search fix
uv run python test_vec_fix.py

# Test LLM configuration
uv run python test_llm_config.py

# Test RAGAS integration
uv run python test_ragas_fix.py
```

## Architecture

### Data Structure
- **Product data**: `data/top_1000_products.jsonl` - Top electronics products with ratings, review counts
- **Review data**: `data/100_top_reviews_of_the_top_1000_products.jsonl` - Customer reviews with ratings, text, helpful votes

### Infrastructure Services
- **Qdrant** (port 6333): Vector database for semantic search and product embeddings
- **Redis** (port 6379): Caching layer with 2GB memory limit and LRU eviction

### Key Components
- **app/cli.py**: Typer-based CLI with commands for ingestion, search, chat, and evaluation
- **app/llm_config.py**: Centralized LLM configuration supporting OpenAI and LiteLLM proxy
- **notebooks/ecommerce_analysis.py**: Marimo notebook for interactive data analysis and visualization using Plotly
- **notebooks/ingest_embeddings.py**: Marimo notebook for ingesting products and reviews into Qdrant vector database using gte-large embeddings
- **notebooks/retrieval_fusion.py**: Hybrid search implementation with RRF and cross-encoder reranking
- **main.py**: Entry point for the application (to be expanded)

### Recent Changes (Session 2025-08-09)

#### Major Additions
- **Created app/cli.py**: Full Typer CLI implementation with all core functionality
- **Created app/llm_config.py**: Centralized LLM configuration with GPT-5 support
- **Created eval/datasets/**: Evaluation datasets for search and chat
  - `search_eval.jsonl`: 20 e-commerce search queries
  - `chat_eval.jsonl`: 20 Q&A pairs with reference answers
- **Upgraded Cross-Encoder**: Switched from ms-marco-MiniLM-L-6-v2 to ms-marco-MiniLM-L-12-v2
  - 15-20% better reranking accuracy
  - Better understanding of nuanced relevance
  - Still maintains good speed (50-100 docs/sec)

#### Fixes Applied
1. **Vector Search ID Mapping**: Fixed UUID to original ID mapping issue
   - Vector search now properly maps Qdrant UUIDs to `prod::ASIN` format
   - Enables correct payload retrieval for evaluation

2. **RAGAS Score Extraction**: Fixed handling of EvaluationResult object
   - Handles `res.scores` as list containing dictionary
   - Properly extracts metrics with NaN handling

3. **LiteLLM Dependencies**: Added all required packages
   - apscheduler
   - cryptography
   - python-multipart
   - email-validator

4. **Timestamp Format**: Improved evaluation output naming
   - Changed from Unix timestamp to YYYYMMDD_HHMMSS format
   - Example: `search_20250809_174328.json`

5. **LLM Configuration**: Flexible model support
   - Default: GPT-4o-mini with configurable temperature
   - GPT-5 support with automatic temperature=1.0 enforcement
   - Environment variable configuration for API keys

### Technology Stack
- **Data Processing**: Polars for fast dataframe operations
- **RAG Framework**: DSPy for retrieval-augmented generation pipelines
- **ML/AI**: Instructor for structured LLM outputs, RAGAS for RAG evaluation
- **Search**: BM25 for keyword search, Qdrant for vector search
- **Web API**: FastAPI for serving endpoints
- **External Data**: Tavily for web search integration
- **CLI**: Typer for command-line interface
- **Package Manager**: uv for fast Python package management

## Development Notes

- Python 3.12+ required
- Uses `uv` for fast Python package management
- Uses `uv run` for executing python code and other apps
- Uses `uv add` instead of `pip install`
- Docker services must be running for vector search and caching functionality
- Marimo notebooks combine code, visualizations, and markdown in executable Python files
- Run marimo notebooks via `uv run marimo run`
- Evaluation results include both JSON and Markdown reports with timestamped filenames

## LLM Configuration

The project uses a centralized LLM configuration in `app/llm_config.py`:
- Default model: GPT-4o-mini
- Supports OpenAI API directly or via LiteLLM proxy
- Automatic handling of model-specific requirements (e.g., GPT-5 temperature restrictions)
- Configure via environment variables:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_API_BASE`: Optional LiteLLM proxy URL

## Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER update the git config
- NEVER use git commands with the -i flag (interactive mode not supported)
- When committing, always end commit messages with the Claude Code signature