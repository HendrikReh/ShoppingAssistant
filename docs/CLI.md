## ShoppingAssistant CLI

The CLI mirrors notebook functionality: ingestion, search, and chat.

Show commands:
```bash
uv run python -m app.cli
```

### Ingest
```bash
uv run python -m app.cli ingest \
  --products-path data/top_1000_products.jsonl \
  --reviews-path  data/100_top_reviews_of_the_top_1000_products.jsonl \
  --products-batch-size 128 --reviews-batch-size 256
```
What happens:
- Reads JSONL files and normalizes records
- Embeds with Sentence-Transformers (gte-large) on CPU/MPS
- Upserts vectors and payloads into Qdrant (UUID from original_id)
- Prints progress and a final summary (with Qdrant counts when available)

### Search
```bash
uv run python -m app.cli search --query "wireless earbuds" --top-k 20 --rrf-k 60 --rerank --rerank-top-k 30
```
Variants compared in eval (see EVALUATION.md): BM25, vectors, RRF, and RRF+Cross-Encoder.

### Chat
```bash
uv run python -m app.cli chat --question "What are good budget earbuds?"
```
Uses DSPy with your configured LLM to answer using retrieved contexts from Qdrant.


