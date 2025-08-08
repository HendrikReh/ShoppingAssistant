Considerations for our shopping assistant:
Use separate collections for products and reviews to support hybrid retrieval and different ranking strategies.
Payloads include parent_asin, average_rating, num_reviews, helpful_vote so we can filter/boost by quality signals.
Preprocess text: combine title + description for products; title + text for reviews. Consider adding key features if available.
Normalization: cosine distance in Qdrant with L2-normalized vectors.
Batch sizes are tuned conservatively (128/256); adjust if you have more memory/CPU.
Next steps:
Add BM25 keyword search and reciprocal rank fusion with vectors.
Add cross-encoder reranking for top-K candidates.
Periodic refresh jobs to re-embed updated content.
Guardrails for toxic/PII content in reviews before surfacing.
If you want, I can wire a small retrieval function that queries both collections and fuses results.