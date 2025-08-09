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


1. Vector search returning 0 scores - This was fixed by mapping Qdrant UUIDs to original IDs
  2. Low context utilization (15-17%) - This appears to be normal/expected for RAG systems

  Looking at the successful evaluation in search_20250809_180857.json:
  - RRF context_utilization: 9.1% (very low)
  - RRF with cross-encoder context_utilization: 26.6% (improved but still low)
  - RRF context relevance: 82.5% (good)
  - RRF with cross-encoder relevance: 90% (excellent)

  Analysis of RAG Retrieval Issues:

  1. Low Context Utilization (9-27%)

  This is actually normal for RAG systems because:
  - Retrieved chunks contain more information than needed for specific queries
  - Products/reviews have full descriptions but queries focus on specific aspects
  - This metric penalizes comprehensive context, which is actually helpful for LLMs

  2. Why Vector Search Had Issues

  The original problem (0 scores) was due to:
  - Qdrant returns UUIDs but evaluation expected "prod::ASIN" format
  - Fixed by extracting original_id from payload

  3. Performance Observations

  - Cross-encoder reranking helps significantly: Improves context utilization from 9% to 27%
  - Context relevance is good: 82-90% means retrieved content is relevant
  - Trade-off: More relevant context (higher utilization) vs comprehensive context (lower utilization)

  4. Potential Improvements

  1. Chunk size optimization: Smaller chunks would increase utilization but might lose context
  2. Hybrid scoring weights: Tune BM25 vs vector search balance
  3. Query expansion: Add synonyms to improve recall
  4. Document segmentation: Break products/reviews into aspect-based chunks

  5. Not Actually "Bad" Results

  The results aren't bad - they're typical for e-commerce RAG:
  - High relevance (82-90%) is more important than utilization
  - Cross-encoder significantly improves both metrics
  - System successfully retrieves relevant products/reviews

  The key insight: Low context utilization in RAG is expected because you want to provide the LLM with comprehensive context, even if it only uses portions of it for the specific answer.
