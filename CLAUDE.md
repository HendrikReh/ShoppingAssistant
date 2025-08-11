# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Instructions

During the development phase of this project I will give you examples which are stored under the folder named cohort. Most of the time I will ask you to migrate specific files from the cohort folder to the actually used tech stack of this project (check pyproject.toml).
You will never change code under the folder cohort neither will you mention its content in any documentation.

## Used Data Sets

The data stored under data/100_top_reviews_of_the_top_1000_products.jsonl and /Volumes/Halle4/projects/ShoppingAssistant/data/top_1000_products.jsonl were extracted from 'Amazon Reviews dataset, collected in 2023' (https://amazon-reviews-2023.github.io/)

## Project Overview

Shopping Assistant is an advanced e-commerce search and recommendation system that combines:
- Hybrid search (BM25 + semantic vectors) with Reciprocal Rank Fusion
- Cross-encoder reranking for improved relevance (ms-marco-MiniLM-L-12-v2)
- Retrieval Augmented Generation (RAG) using DSPy framework
- Vector search with Qdrant (384-dim all-MiniLM-L6-v2 embeddings)
- Redis caching layer with 2GB limit and LRU eviction
- Interactive data analysis with Marimo notebooks
- Comprehensive evaluation with RAGAS metrics
- Full-featured command-line interface with Typer

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Load environment variables (OpenAI API key, etc.)
set -a
source .env

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

#### Interactive Mode (ENHANCED!)
```bash
# Unified interactive mode - just type naturally, no mode selection needed
uv run python -m app.cli interactive

# Features:
# - Automatically detects search vs chat intent
# - Type product names for search (e.g., "wireless mouse")
# - Ask questions for recommendations (e.g., "what's the best laptop?")
# - Type /help for tips and examples
# - Type /exit or Ctrl+C to quit
```

#### Ingestion
```bash
# Ingest products and reviews into Qdrant
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path data/100_top_reviews_of_the_top_1000_products.jsonl
```

#### Search
```bash
# Interactive search mode - continuous searching with commands
uv run python -m app.cli search

# Single query search
uv run python -m app.cli search \
  --query "wireless earbuds" \
  --top-k 20 \
  --rrf-k 60 \
  --rerank
```

#### Chat
```bash
# Interactive chat mode - have a conversation
uv run python -m app.cli chat

# Single question
uv run python -m app.cli chat \
  --question "What are the best budget earbuds?"
```

#### Interactive Features
- **Unified Mode**: No need to choose between search and chat - system auto-detects intent
- **Search Mode Commands**: `/help` (tips), `/settings` (config), `/exit`
- **Chat Mode Commands**: `/help` (examples), `/context` (settings), `/clear` (screen), `/exit`
- **Visual Feedback**: Color-coded results, emojis, progress indicators
- **Persistent Sessions**: Models and data loaded once for faster responses
- **Smart Routing**: Simple product names trigger search, questions trigger chat

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

### Testing & Evaluation
```bash
# Generate synthetic test data (NEW!)
uv run python -m app.cli generate-testset --num-samples 500

# Run evaluation with generated data
uv run python -m app.cli eval-search \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --variants bm25,vec,rrf,rrf_ce

uv run python -m app.cli eval-chat \
  --dataset eval/datasets/synthetic_500_*.jsonl \
  --top-k 8 --max-samples 50

# Test specific components
uv run python test_vec_fix.py  # Test vector search
uv run python test_llm_config.py  # Test LLM configuration
uv run python test_ragas_fix.py  # Test RAGAS integration
```

## Architecture

### Data Structure
- **Product data**: `data/top_1000_products.jsonl` - Top electronics products with ratings, review counts
- **Review data**: `data/100_top_reviews_of_the_top_1000_products.jsonl` - Customer reviews with ratings, text, helpful votes

### Infrastructure Services
- **Qdrant** (port 6333): Vector database for semantic search and product embeddings
- **Redis** (port 6379): Caching layer with 2GB memory limit and LRU eviction

### Model Choices & Performance

#### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384 (efficient, fast)
- **Why chosen**: GTE-large compressed all products to 0.7-0.8 similarity range
- **Performance**: Vector search relevance improved from 2.5% to 70%
- **Collections**: `products_minilm`, `reviews_minilm`

#### Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Speed**: 221 docs/sec on MPS, 136ms latency for 30 docs
- **Accuracy**: Excellent ranking quality with clear score discrimination
- **Size**: 475MB model, 125M parameters
- **Alternative for speed**: `ms-marco-MiniLM-L-6-v2` (2x faster, -3% accuracy)
- **Benchmarked against**: BGE-reranker, ELECTRA, TinyBERT, MxBai
- **See**: `docs/CROSS_ENCODER_COMPARISON.md` for detailed benchmarks

### Key Components
- **app/cli.py**: Typer-based CLI with commands for ingestion, search, chat, and evaluation
- **app/llm_config.py**: Centralized LLM configuration supporting OpenAI and LiteLLM proxy
- **notebooks/ecommerce_analysis.py**: Marimo notebook for interactive data analysis and visualization using Plotly
- **notebooks/ingest_embeddings.py**: Marimo notebook for ingesting products and reviews into Qdrant vector database using all-MiniLM-L6-v2 embeddings
- **notebooks/retrieval_fusion.py**: Hybrid search implementation with RRF and cross-encoder reranking
- **main.py**: Entry point for the application (to be expanded)

### Recent Changes

#### Session 2025-08-11 Updates

**Improved Product Search:**
- **Query preprocessing**: Automatically removes command prefixes like "search for", "find", "show me"
- **Products-only filtering**: Detects "products only" suffix and excludes reviews from results
- **Product boosting**: RRF fusion now applies 1.5x weight boost to products over reviews
- **Better relevance**: Product searches now consistently return products as top results
- **Fixed search results**: "TV Sticks" and similar queries now correctly return relevant products
- **LLM-based typo correction**: Intelligent query correction using LLM with caching for performance
  - Pre-filtering with heuristics to avoid unnecessary LLM calls
  - Cached common corrections for instant response
  - Fallback to original query if correction fails

#### Session 2025-08-10 Updates (Part 4)

**Unified Interactive Mode:**
- **No mode selection required**: Users can type naturally without choosing search or chat
- **Intelligent routing**: System automatically detects search vs chat intent
  - Simple product names → Search mode
  - Questions with "?" or question words → Chat mode
  - Search keywords (find, show, list) → Search mode
  - Chat keywords (what, how, why, explain) → Chat mode
- **Lazy loading**: Components load on first use for instant startup (no delay before "You:" prompt)
- **Clean UI**: Removed all emojis for cleaner, professional interface
- **Single interface**: Seamless experience without mode switching friction

**Fixed Evaluation Metrics:**
- **Removed RAGAS metrics from search evaluation**: ContextRelevance and ContextUtilization require answers
- **Added simple retrieval metrics**:
  - `num_queries`: Number of queries evaluated
  - `avg_contexts_retrieved`: Average contexts per query
  - `queries_with_results`: Success rate (0-1 ratio)
- **Fixed percentage calculations**: Corrected eval_interpreter.py to use appropriate metrics
- **No more NaN values**: Search evaluation now provides meaningful statistics

**Interactive Mode Bug Fixes:**
- **Fixed parameter passing**: Resolved TypeError when invoking commands from interactive mode
- **Proper defaults**: All command parameters now explicitly passed with correct values
- **Improved error handling**: Better exception handling and user feedback

#### Session 2025-08-10 Updates (Part 3)

**Enhanced Evaluation Reporting:**
- **Query-level details**: Each evaluation now captures request, response, and metrics per query
- **Individual results**: JSON reports include `detailed_results` with retrieved items
- **Sample queries**: Markdown reports show top 3 example queries with their results
- **Benefits**: Easier debugging, pattern identification, manual inspection
- **See**: `docs/ENHANCED_EVALUATION_GUIDE.md` for usage

**Cross-Encoder Benchmarking:**
- **Tested 5 models**: MiniLM-L-12, MiniLM-L-6, BGE-reranker, ELECTRA, TinyBERT
- **Winner**: Current ms-marco-MiniLM-L-12-v2 provides best balance
- **Performance**: 221 docs/sec with excellent ranking quality
- **Documentation**: Added `docs/CROSS_ENCODER_COMPARISON.md`
- **Benchmark tool**: `test_rerankers.py` for testing on your data

**Sample Size Guidelines:**
- **Created comprehensive guide**: `docs/EVALUATION_SAMPLE_SIZE_GUIDE.md`
- **Sweet spot**: 100 samples ($0.05, 5 minutes, ±10% confidence)
- **Progressive testing**: Start with 20, scale to 100, then 500 for production
- **Cost analysis**: Detailed API costs and runtime for each sample size

#### Session 2025-08-10 Updates (Part 2)

**Critical Vector Search Fix:**
- **Identified Issue**: GTE-large embedding model compressed all e-commerce products into narrow similarity range (0.7-0.8)
  - Made it impossible to distinguish between products
  - "Fire TV Stick" had higher similarity with earbuds (0.786) than with actual Fire TV products (0.774)
  
- **Solution**: Switched from `thenlper/gte-large` (1024d) to `sentence-transformers/all-MiniLM-L6-v2` (384d)
  - Better discrimination between product types
  - Smaller, faster model with better e-commerce performance
  - Updated collections: `products_minilm` and `reviews_minilm`
  
- **Results**:
  - Vector search relevance: 2.5% → 70% (2700% improvement)
  - RRF hybrid search: 70% → 90% relevance
  - Overall retrieval relevance: 44.1% → 80%+ 
  - Queries now return semantically relevant products

**Automatic Environment Loading:**
- **Added python-dotenv**: CLI now automatically loads `.env` file
- No need for manual `set -a` and `source .env` commands
- API keys loaded automatically when CLI starts

**Execution Command Tracking:**
- **Full command capture**: Both eval-search and eval-chat store complete execution command
- Appears in JSON reports as `"execution_command"` field
- Shown in Markdown reports for easy reproducibility
- Example: `uv run python -m app.cli eval-search --dataset ... --top-k 25`

**Realistic Test Data Generation:**
- **Consolidated to testset_generator.py**: Catalog-based query generation only
  - Uses actual products from catalog (top 500 items)
  - Realistic brand-category combinations (no more "JBL storage")
  - Queries reference real product titles and features
  - Proper comparative queries between similar products
  
- **Optimized Retrieval Parameters:**
  - Increased default top_k: 10 → 25-30 for better recall
  - Increased rrf_k: 60 → 80-100 for improved fusion
  - Added --enhanced mode support with query expansion
  - Significantly improved context relevance scores

- **Dataset Cleanup:**
  - Removed obsolete synthetic datasets with nonsensical queries
  - Kept hand-crafted evaluation sets (search_eval.jsonl, chat_eval.jsonl)
  - New realistic datasets match actual product inventory

#### Session 2025-08-10 Updates (Part 1)

**Synthetic Test Data Generation:**
- **Created app/testset_generator.py**: Comprehensive test data generator
  - 7 query types: single-hop, multi-hop, comparative, recommendation, technical, abstract, problem-solving
  - 4 distribution presets: balanced, simple, complex, mixed
  - Generates 100-1000+ samples with diverse complexity
  - Includes reference answers for evaluation
  - Entity extraction from actual product/review data
  
- **Added generate-testset CLI command**: 
  - Generates synthetic datasets with metadata
  - Visual progress indicators and statistics
  - Outputs RAGAS-compatible JSONL format
  - Increased test coverage from ~50 to 500+ samples

- **Updated CLI commands**: Changed from word commands to slash commands
  - `/help`, `/settings`, `/exit` in search mode
  - `/help`, `/context`, `/clear`, `/exit` in chat mode
  - More distinct from regular user input

- **Environment improvements**:
  - Added `TOKENIZERS_PARALLELISM=false` to .env files
  - Fixed all deprecation warnings
  - Added fastapi-sso dependency

- **Model caching optimization**:
  - Fixed HTTP 429 errors from repeated model downloads
  - Cross-encoder and sentence transformer models cached globally
  - 10-100x speedup for repeated evaluations
  - Pre-loading for cross-encoder in eval-search command

- **Enhanced evaluation reports with interpretations**:
  - Created `app/eval_interpreter.py` for metric interpretation
  - Executive summaries with system health status
  - Detailed metric-by-metric analysis with recommendations
  - Variant performance comparisons for search strategies
  - Critical issue identification and prioritization
  - Success criteria assessment
  - Strategic action plans based on results

#### Session 2025-08-09 Updates

**Major Additions:**
- **Created app/cli.py**: Full Typer CLI implementation with all core functionality
- **Created app/llm_config.py**: Centralized LLM configuration for DSPy and RAGAS
- **Created eval/datasets/**: Evaluation datasets for search and chat
  - `search_eval.jsonl`: 20 e-commerce search queries
  - `chat_eval.jsonl`: 20 Q&A pairs with reference answers
- **Created docs/RAG_EVALUATION_INSIGHTS.md**: Comprehensive guide on interpreting RAG metrics
- **Created docs/CLI.md**: Complete CLI reference documentation
- **Enhanced Evaluation Reports**: Added comprehensive call parameters to all reports
  - JSON reports include `call_parameters` section with all command arguments
  - Markdown reports include formatted "Call Parameters" section
  - Full traceability of model configurations, paths, and execution parameters
- **Interactive Modes**: Added three interactive modes for better UX
  - Interactive search mode with continuous querying
  - Enhanced chat mode with conversation history
  - Combined `interactive` command to switch between modes
- **Visual CLI Enhancements**: 
  - Color-coded outputs using `typer.secho()`
  - Relevance-based coloring (green/yellow/white)
  - Emojis for better visual distinction
  - Progress indicators and thinking animations

**Model Upgrades:**
- **Cross-Encoder**: Upgraded from ms-marco-MiniLM-L-6-v2 to ms-marco-MiniLM-L-12-v2
  - 15-20% better reranking accuracy
  - 192% improvement in context utilization (9.1% → 26.6%)
  - Maintains good speed (50-100 docs/sec)

**Critical Fixes:**
1. **Vector Search ID Mapping**: Fixed Qdrant UUID to original ID mapping
   - Extracts `original_id` from payload for correct evaluation
   - Resolves issue where vector search returned 0 scores

2. **RAGAS Score Extraction**: Fixed EvaluationResult handling
   - Properly handles `res.scores` as list containing dictionary
   - Includes NaN handling for failed evaluations

3. **Dependencies**: Added missing LiteLLM requirements
   - apscheduler, cryptography, python-multipart, email-validator

4. **Timestamp Format**: Human-readable evaluation output naming
   - Format: YYYYMMDD_HHMMSS (e.g., `search_20250809_174328.json`)

5. **LLM Configuration**: Flexible model support
   - Default: GPT-4o-mini
   - Environment-based configuration via OPENAI_API_KEY and OPENAI_API_BASE

6. **Test Organization**: Cleaned up test structure
   - Moved useful tests to `tests/` directory
   - Removed obsolete one-time fix tests
   - Added test documentation in `tests/README.md`

### Technology Stack
- **Data Processing**: Polars for fast dataframe operations
- **RAG Framework**: DSPy for retrieval-augmented generation pipelines
- **ML/AI**: Instructor for structured LLM outputs, RAGAS for RAG evaluation
- **Search**: 
  - BM25 for keyword search
  - Qdrant for vector search (all-MiniLM-L6-v2 embeddings, 384-dim)
  - Reciprocal Rank Fusion (RRF) for hybrid search
  - Cross-encoder reranking (ms-marco-MiniLM-L-12-v2)
- **Web API**: FastAPI for serving endpoints
- **External Data**: Tavily for web search integration
- **CLI**: Typer for command-line interface
- **Package Manager**: uv for fast Python package management
- **Infrastructure**: Docker Compose for Qdrant and Redis

## Development Notes

### Requirements
- Python 3.12+ required
- Docker and Docker Compose for infrastructure services
- uv package manager for Python dependencies

### Best Practices
- **Package Management**: Always use `uv add` instead of `pip install`
- **Code Execution**: Use `uv run` for all Python scripts and commands
- **Docker Services**: Must be running for vector search and caching
- **Marimo Notebooks**: Interactive Python files combining code, visualizations, and markdown
- **Evaluation Reports**: Generated with timestamps (YYYYMMDD_HHMMSS) in both JSON and Markdown formats

### Understanding RAG Metrics
- **Context Utilization 9-27% is NORMAL** for e-commerce RAG systems
- **Context Relevance 82-90% is GOOD** performance
- Low utilization means comprehensive retrieval (feature, not bug)
- See `docs/RAG_EVALUATION_INSIGHTS.md` for detailed explanation

## LLM Configuration

The project uses a centralized LLM configuration in `app/llm_config.py`:
- Default model: GPT-4o-mini
- Supports OpenAI API directly or via LiteLLM proxy
- Automatic handling of model-specific requirements (e.g., GPT-5 temperature restrictions)
- Configure via environment variables:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_API_BASE`: Optional LiteLLM proxy URL

## Important Reminders for Claude Code

### File Operations
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation files unless explicitly requested

### Git Operations
- NEVER update git config
- NEVER use interactive git commands (-i flag)
- When committing, always include Claude Code signature

### Testing Commands
When asked to test or verify changes:
```bash
# Test vector search
uv run python test_vec_fix.py

# Test cross-encoder
uv run python test_new_reranker.py

# Test evaluation reports
uv run python test_enhanced_reports.py
```

### Common Evaluation Commands
```bash
# Generate realistic test data (uses actual product catalog)
uv run python -m app.cli generate-testset \
  --num-samples 100 \
  --output-name realistic_catalog

# Full search evaluation with optimized parameters
# Note: Search evaluation now uses simple retrieval metrics (no RAGAS)
uv run python -m app.cli eval-search \
  --dataset eval/datasets/realistic_catalog_*.jsonl \
  --variants bm25,vec,rrf,rrf_ce \
  --top-k 25 --rrf-k 80 \
  --max-samples 50

# Quick chat evaluation with RAGAS metrics
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/realistic_catalog_*.jsonl \
  --top-k 8 --max-samples 20
```