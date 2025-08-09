## Evaluation Plan (RAGAS + MLflow + LiteLLM)

We evaluate retrieval (search) and chat with LLM-based metrics using RAGAS and log results to MLflow.

### Datasets
- Place JSONL under `eval/datasets/`
  - Search: `{ "query": "...", "notes": "..." }`
  - Chat: `{ "question": "...", "notes": "..." }`
- You can later add labels:
  - `relevant_ids: ["prod::...", "rev::...", ...]`
  - `reference_answer: "..."`

### Metrics
- Search (no labels): LLM-judged relevance@K proxies using RAGAS components (context_precision/recall, answer_relevancy over snippet summaries)
- Chat (single-turn): faithfulness, context_precision/context_recall, answer_relevancy
- With labels (optional, future): Recall@K, MRR@K, nDCG@K for search; answer correctness for chat

### Variants (Search)
- BM25, Vectors, RRF, RRF+Cross-Encoder (side-by-side table)

### Evaluator LLM via LiteLLM
Recommended: point OpenAI-compatible client to LiteLLM proxy so RAGAS can use it transparently.

Environment:
```
export OPENAI_API_BASE=http://localhost:4000/v1
export OPENAI_API_KEY=<your_litellm_key>
export EVAL_MODEL=gpt-5-mini
```

### CLI Commands
- Eval search:
```
uv run python -m app.cli eval-search \
  --dataset eval/datasets/search_eval.jsonl \
  --top-k 20 --rrf-k 60 --max-samples 100 --variants bm25,vec,rrf,rrf_ce
```

- Eval chat:
```
uv run python -m app.cli eval-chat \
  --dataset eval/datasets/chat_eval.jsonl \
  --top-k 8 --max-samples 50
```

### Outputs
- JSON with per-sample metrics under `eval/results/`
- Markdown report with aggregate metrics and variant comparison
- MLflow run with params/metrics/artifacts

### Notes
- Sampling is deterministic via `--seed`.
- Batch limits and timeouts protect cost.
- Keep evaluator model distinct from production model.


