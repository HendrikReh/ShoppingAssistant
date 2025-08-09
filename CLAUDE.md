# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General instructions

During the development phase of this project I will give you examples whicha re stored under the folder named cohort. Most of the time I will ask you to migrate specif files from the cohort folder to the actually used tech stack of this project (check pyproject.toml).
You will never change coder under the folder cohort neither will you mention its content in any documentation.

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

### Marimo Notebooks
```bash
# Run interactive notebook
uv run marimo run notebooks/ecommerce_analysis.py
uv run marimo run notebooks/ingest_embeddings.py

# Edit notebook
uv run marimo edit notebooks/ecommerce_analysis.py
uv run marimo edit notebooks/ingest_embeddings.py
```

### Testing
```bash
# Run tests (when test framework is added)
# uv run pytest tests/
```

## Architecture

### Data Structure
- **Product data**: `data/top_1000_products.jsonl` - Top electronics products with ratings, review counts
- **Review data**: `data/100_top_reviews_of_the_top_1000_products.jsonl` - Customer reviews with ratings, text, helpful votes

### Infrastructure Services
- **Qdrant** (port 6333): Vector database for semantic search and product embeddings
- **Redis** (port 6379): Caching layer with 2GB memory limit and LRU eviction

### Key Components
- **notebooks/ecommerce_analysis.py**: Marimo notebook for interactive data analysis and visualization using Plotly
- **notebooks/ingest_embeddings.py**: Marimo notebook for ingesting products and reviews into Qdrant vector database using gte-large embeddings
- **main.py**: Entry point for the application (to be expanded)

### Recent Changes
- Added `notebooks/ingest_embeddings.py` for vector embeddings ingestion - embeds products and reviews using thenlper/gte-large model (1024-dim) and stores in Qdrant collections
- Added `notebooks/ecommerce_analysis.py` as a Marimo notebook that reads `data/top_1000_products.jsonl` and `data/100_top_reviews_of_the_top_1000_products.jsonl`, using Polars and Plotly for analysis and charts.
- Added cross-encoder reranking to `notebooks/retrieval_fusion.py` using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Toggle via "Use cross-encoder rerank" and control "Rerank top-K after fusion".
- Added `app/cli.py` Typer CLI for ingestion, search (BM25+vectors+RRF+CE), and DSPy-based chat.
- Added docs: `docs/CLI.md`, `docs/EVALUATION.md`, and dataset guidance under `eval/datasets/`.

### Technology Stack
- **Data Processing**: Polars for fast dataframe operations
- **RAG Framework**: DSPy for retrieval-augmented generation pipelines
- **ML/AI**: Instructor for structured LLM outputs, RAGAS for RAG evaluation
- **Search**: BM25 for keyword search, Qdrant for vector search
- **Web API**: FastAPI for serving endpoints
- **External Data**: Tavily for web search integration

## Development Notes

- Python 3.12+ required
- Uses `uv` for fast Python package management
- Uses `uv run` for executing python code and other apps
- Uses `uv add` instead of `pip install`
- Docker services must be running for vector search and caching functionality
- Marimo notebooks combine code, visualizations, and markdown in executable Python files
- Run marimo notebooks via `uv run marimo run`