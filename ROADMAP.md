Considerations for our shopping assistant:
Use separate collections for products and reviews to support hybrid retrieval and different ranking strategies.
Payloads include parent_asin, average_rating, num_reviews, helpful_vote so we can filter/boost by quality signals.
Preprocess text: combine title + description for products; title + text for reviews. Consider adding key features if available.
Normalization: cosine distance in Qdrant with L2-normalized vectors.
Batch sizes are tuned conservatively (128/256); adjust if you have more memory/CPU.
Next steps:
done: Add BM25 keyword search and reciprocal rank fusion with vectors.
done: Add cross-encoder reranking for top-K candidates.
Periodic refresh jobs to re-embed updated content.
Guardrails for toxic/PII content in reviews before surfacing.
If you want, I can wire a small retrieval function that queries both collections and fuses results.




 Recommended Changes for Your System:

  1. Short-term: Switch to ms-marco-MiniLM-L-12-v2
    - Easy drop-in replacement
    - Noticeable accuracy improvement
    - Still fast enough for interactive use
  2. Medium-term: Implement query-type routing
    - Use TinyBERT for simple keyword searches
    - Use L-12 or base model for complex queries
    - Use QA model for chat interactions
  3. Long-term: Fine-tune on your data
    - Collect click-through data
    - Create training pairs from user interactions
    - Fine-tune base model on e-commerce relevance






    ------ TEST ---------

uv run python -m app.cli eval-search --dataset eval/datasets/search_eval.jsonl --top-k 20 --rrf-k 60 --max-samples 100 --variants bm25,vec,rrf,rrf_ce

uv run python -m app.cli eval-chat --dataset eval/datasets/chat_eval.jsonl --top-k 8 --max-samples 50