# Shopping Assistant Tests

This directory contains the test suite for the Shopping Assistant project.

## Test Files

### test_vector_search.py
Tests the vector search functionality with Qdrant, including:
- UUID to original ID mapping
- Payload retrieval
- Search result formatting

### test_cross_encoder.py
Tests cross-encoder reranking models:
- Compares ms-marco-MiniLM-L-6-v2 vs L-12-v2
- Validates reranking performance
- Benchmarks speed differences

### test_reports.py
Tests evaluation report generation:
- JSON report structure with call parameters
- Markdown report formatting
- Timestamp formatting (YYYYMMDD_HHMMSS)

## Running Tests

```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test file
uv run python tests/test_vector_search.py
uv run python tests/test_cross_encoder.py
uv run python tests/test_reports.py

# Run with verbose output
uv run python -m pytest tests/ -v
```

## Test Requirements

- Docker services must be running (Qdrant, Redis)
- Environment variables must be set (OPENAI_API_KEY)
- Data must be ingested before running integration tests

## Adding New Tests

When adding new tests:
1. Follow the naming convention: `test_<feature>.py`
2. Include docstrings explaining what is being tested
3. Use clear test function names that describe the scenario
4. Clean up any test artifacts after completion