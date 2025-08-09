"""ShoppingAssistant CLI using Typer.

Reimplements notebook functionality:
- Ingest embeddings into Qdrant
- Hybrid retrieval (BM25 + vectors + RRF) with optional cross-encoder rerank
- Chat with ingested data using DSPy for LLM functionality

Run examples:
  uv run python -m app.cli ingest
  uv run python -m app.cli search --query "wireless earbuds"
  uv run python -m app.cli chat
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import typer
from app.llm_config import get_llm_config


# Third-party deps loaded lazily in functions to speed up CLI startup

app = typer.Typer(help="ShoppingAssistant CLI")


# Defaults aligned with notebooks
DATA_PRODUCTS = Path("data/top_1000_products.jsonl")
DATA_REVIEWS = Path("data/100_top_reviews_of_the_top_1000_products.jsonl")
COLLECTION_PRODUCTS = "products_gte_large"
COLLECTION_REVIEWS = "reviews_gte_large"
EMBED_MODEL = "thenlper/gte-large"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
VECTOR_SIZE = 1024


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """ShoppingAssistant command-line interface.

    Run with a subcommand (ingest, search, chat). If no subcommand is given,
    show the available commands and options.
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r") as fp:
        for idx, line in enumerate(fp, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {e}") from e
    return rows


def _to_uuid_from_string(value: str) -> str:
    import uuid

    hexdigest = hashlib.md5(value.encode()).hexdigest()
    return str(uuid.UUID(hexdigest))


def _load_st_model(model_name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _device_str() -> str:
    import torch

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_collections(client, qmodels, vector_size: int, names: list[str]) -> None:
    for name in names:
        if not client.collection_exists(name):
            client.recreate_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )


def _qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(host="localhost", port=6333, prefer_grpc=False)


def _qmodels():
    from qdrant_client.http import models as qmodels

    return qmodels


def _embed_texts(model, texts: list[str], device: str, batch_size: int) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def _chunked(items: Iterable, n: int) -> Iterable[list]:
    chunk: list = []
    for item in items:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _build_product_docs(products: list[dict]) -> list[dict]:
    docs: list[dict] = []
    for row in products:
        parent_asin = str(row.get("parent_asin", ""))
        docs.append(
            {
                "id": f"prod::{parent_asin}",
                "parent_asin": parent_asin,
                "title": row.get("title") or "",
                "description": (row.get("description") or ""),
                "average_rating": row.get("average_rating"),
                "num_reviews": row.get("review_count")
                if row.get("review_count") is not None
                else row.get("rating_number"),
            }
        )
    return docs


def _build_review_docs(reviews: list[dict]) -> list[dict]:
    docs: list[dict] = []
    for i, row in enumerate(reviews):
        parent_asin = str(row.get("parent_asin", ""))
        docs.append(
            {
                "id": f"rev::{parent_asin}::{i}",
                "parent_asin": parent_asin,
                "title": row.get("title") or "",
                "text": row.get("text") or "",
                "rating": row.get("rating"),
                "helpful_vote": row.get("helpful_vote"),
            }
        )
    return docs


def _product_text(doc: dict) -> str:
    return f"Title: {doc.get('title','')}\nDescription: {doc.get('description','')}".strip()


def _review_text(doc: dict) -> str:
    return f"Title: {doc.get('title','')}\nReview: {doc.get('text','')}".strip()


@app.command()
def ingest(
    products_path: Path = typer.Option(DATA_PRODUCTS, exists=True, readable=True),
    reviews_path: Path = typer.Option(DATA_REVIEWS, exists=True, readable=True),
    products_batch_size: int = typer.Option(128, min=1),
    reviews_batch_size: int = typer.Option(256, min=1),
    model_name: str = typer.Option(EMBED_MODEL),
    collection_products: str = typer.Option(COLLECTION_PRODUCTS),
    collection_reviews: str = typer.Option(COLLECTION_REVIEWS),
) -> None:
    """Embed products and reviews with Sentence-Transformers and upsert into Qdrant."""

    device = _device_str()
    st_model = _load_st_model(model_name, device=device)
    client = _qdrant_client()
    qmodels = _qmodels()
    _ensure_collections(client, qmodels, VECTOR_SIZE, [collection_products, collection_reviews])

    typer.echo("Starting ingestion with configuration:")
    typer.echo(f"- Device: {device}")
    typer.echo(f"- Embed model: {model_name}")
    typer.echo(f"- Products path: {products_path}")
    typer.echo(f"- Reviews path:  {reviews_path}")
    typer.echo(f"- Products batch size: {products_batch_size}")
    typer.echo(f"- Reviews batch size:  {reviews_batch_size}")
    typer.echo(f"- Qdrant collections: products={collection_products} reviews={collection_reviews}")
    typer.echo("Plan:")
    typer.echo("  1) Read JSONL files")
    typer.echo("  2) Build normalized docs (products, reviews)")
    typer.echo("  3) Embed texts with Sentence-Transformers on the selected device")
    typer.echo("  4) Upsert vectors + payloads into Qdrant using deterministic UUIDs (payload keeps original_id)")
    typer.echo("  5) Summarize system state (point counts per collection)")

    products = _read_jsonl(products_path)
    reviews = _read_jsonl(reviews_path)
    product_docs = _build_product_docs(products)
    review_docs = _build_review_docs(reviews)

    def _upsert(collection: str, vectors: list[list[float]], payloads: list[dict], ids: list[str]) -> None:
        client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=_to_uuid_from_string(_id),
                    vector=vec,
                    payload={**payload, "original_id": _id},
                )
                for _id, vec, payload in zip(ids, vectors, payloads)
            ],
        )

    # Ingest products
    start = time.time()
    processed = 0
    for batch in _chunked(product_docs, products_batch_size):
        texts = [_product_text(d) for d in batch]
        vectors = _embed_texts(st_model, texts, device=device, batch_size=products_batch_size)
        ids = [d["id"] for d in batch]
        _upsert(collection_products, vectors, batch, ids)
        processed += len(batch)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        typer.echo(
            f"[products] {processed}/{len(product_docs)} ({processed*100/len(product_docs):.1f}%) "
            f"{rate:.1f}/s elapsed={_format_seconds(elapsed)}"
        )

    prod_time = time.time() - start

    # Ingest reviews
    start_r = time.time()
    processed = 0
    for batch in _chunked(review_docs, reviews_batch_size):
        texts = [_review_text(d) for d in batch]
        vectors = _embed_texts(st_model, texts, device=device, batch_size=reviews_batch_size)
        ids = [d["id"] for d in batch]
        _upsert(collection_reviews, vectors, batch, ids)
        processed += len(batch)
        elapsed = time.time() - start_r
        rate = processed / elapsed if elapsed > 0 else 0
        typer.echo(
            f"[reviews] {processed}/{len(review_docs)} ({processed*100/len(review_docs):.1f}%) "
            f"{rate:.1f}/s elapsed={_format_seconds(elapsed)}"
        )

    rev_time = time.time() - start_r
    # Summarize system state
    prod_count = None
    rev_count = None
    try:
        prod_count = client.count(collection_products, exact=True).count  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        rev_count = client.count(collection_reviews, exact=True).count  # type: ignore[attr-defined]
    except Exception:
        pass

    typer.echo("\nSummary:")
    typer.echo(f"- Ingested products: {len(product_docs)} in {_format_seconds(prod_time)}")
    typer.echo(f"- Ingested reviews:  {len(review_docs)} in {_format_seconds(rev_time)}")
    typer.echo(f"- Device: {device} • Vector dim: {VECTOR_SIZE} • Model: {model_name}")
    if prod_count is not None:
        typer.echo(f"- Qdrant points: {collection_products}={prod_count}")
    if rev_count is not None:
        typer.echo(f"- Qdrant points: {collection_reviews}={rev_count}")


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", (text or "").lower())


def _bm25_from_files(products_path: Path, reviews_path: Path):
    from rank_bm25 import BM25Okapi

    products = _read_jsonl(products_path)
    reviews = _read_jsonl(reviews_path)

    prod_ids = [f"prod::{p.get('parent_asin','')}" for p in products]
    rev_ids = [f"rev::{r.get('parent_asin','')}::{i}" for i, r in enumerate(reviews)]
    prod_docs = [_tokenize(f"{p.get('title','')}\n{p.get('description','')}") for p in products]
    rev_docs = [_tokenize(f"{r.get('title','')}\n{r.get('text','')}") for r in reviews]

    bm25_prod = BM25Okapi(prod_docs)
    bm25_rev = BM25Okapi(rev_docs)
    id_to_product = {pid: {"id": pid, **p} for pid, p in zip(prod_ids, products)}
    id_to_review = {rid: {"id": rid, **r} for rid, r in zip(rev_ids, reviews)}
    return bm25_prod, bm25_rev, id_to_product, id_to_review


def _rrf_fuse(result_lists: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            _id = item[0]
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank)
    return fused


def _vector_search(client, collection: str, vector: list[float], top_k: int = 20) -> list[tuple[str, float, dict]]:
    # Prefer legacy search for broader client compatibility
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=vector,
            with_payload=True,
            limit=top_k,
        )
        return [(str(h.id), float(h.score), h.payload) for h in hits]
    except Exception:
        res = client.query_points(
            collection_name=collection,
            query_vector=vector,  # type: ignore[arg-type]
            with_payload=True,
            limit=top_k,
        )
        points = getattr(res, "points", res)
        return [(str(p.id), float(p.score), p.payload) for p in points]


def _cross_encoder_scores(model_name: str, device: str, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    from sentence_transformers import CrossEncoder

    if not candidates:
        return []
    ce = CrossEncoder(model_name, device=device)
    pairs = [(query, text) for _, text in candidates]
    scores = ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    ranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


@app.command()
def search(
    query: str = typer.Option(..., help="User query"),
    products_path: Path = typer.Option(DATA_PRODUCTS, exists=True, readable=True),
    reviews_path: Path = typer.Option(DATA_REVIEWS, exists=True, readable=True),
    top_k: int = typer.Option(20, min=1, help="Top-K per modality"),
    rrf_k: int = typer.Option(60, help="RRF k"),
    rerank: bool = typer.Option(True, help="Use cross-encoder rerank"),
    rerank_top_k: int = typer.Option(30, help="Top-K to rerank after fusion"),
) -> None:
    """Hybrid retrieval (BM25 + vectors) with optional cross-encoder rerank."""

    device = _device_str()
    st_model = _load_st_model(EMBED_MODEL, device=device)
    client = _qdrant_client()

    # BM25
    bm25_prod, bm25_rev, id_to_product, id_to_review = _bm25_from_files(products_path, reviews_path)
    q_tokens = _tokenize(query)
    bm25_prod_scores = bm25_prod.get_scores(q_tokens)
    bm25_rev_scores = bm25_rev.get_scores(q_tokens)
    prod_ranked = sorted(
        [(pid, float(s)) for pid, s in zip(id_to_product.keys(), bm25_prod_scores)],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    rev_ranked = sorted(
        [(rid, float(s)) for rid, s in zip(id_to_review.keys(), bm25_rev_scores)],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Vectors - need to map UUIDs back to original IDs
    q_vec = st_model.encode([query], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
    prod_vec_raw = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k)
    rev_vec_raw = _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k)
    
    # Map to original IDs from payload
    prod_vec = []
    for uuid, score, payload in prod_vec_raw:
        original_id = payload.get('original_id', payload.get('id', ''))
        if original_id and original_id in id_to_product:
            prod_vec.append((original_id, score))
    
    rev_vec = []
    for uuid, score, payload in rev_vec_raw:
        original_id = payload.get('original_id', payload.get('id', ''))
        if original_id and original_id in id_to_review:
            rev_vec.append((original_id, score))

    fused = _rrf_fuse([prod_ranked, rev_ranked, prod_vec, rev_vec], k=rrf_k)
    fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:50]

    ce_scores: dict[str, float] = {}
    if rerank and fused_sorted:
        k_ce = min(rerank_top_k, len(fused_sorted))
        candidates: list[tuple[str, str]] = []
        for _id, _ in fused_sorted[:k_ce]:
            if _id.startswith("prod::") and _id in id_to_product:
                p = id_to_product[_id]
                candidates.append((_id, f"{p.get('title','')}\n{p.get('description','')}"))
            elif _id.startswith("rev::") and _id in id_to_review:
                r = id_to_review[_id]
                candidates.append((_id, f"{r.get('title','')}\n{r.get('text','')}"))
        ranked = _cross_encoder_scores(CROSS_ENCODER_MODEL, device, query, candidates)
        ce_scores = {cid: score for cid, score in ranked}
        fused_sorted = sorted(
            fused_sorted, key=lambda x: (ce_scores.get(x[0], float("-inf")), x[1]), reverse=True
        )

    # Pretty print
    for _id, score in fused_sorted[:20]:
        if _id.startswith("prod::") and _id in id_to_product:
            p = id_to_product[_id]
            typer.echo(
                f"[product] id={_id} rrf={score:.6f} ce={ce_scores.get(_id):.4f} title={(p.get('title') or '')[:100]}"
            )
        elif _id.startswith("rev::") and _id in id_to_review:
            r = id_to_review[_id]
            typer.echo(
                f"[review]  id={_id} rrf={score:.6f} ce={ce_scores.get(_id):.4f} title={(r.get('title') or '')[:100]}"
            )


@app.command()
def chat(
    question: str = typer.Option(None, help="Ask a single question; if omitted, starts an interactive chat."),
    top_k: int = typer.Option(8, help="Top-K contexts for RAG"),
) -> None:
    """Chat with ingested data using DSPy for the LLM layer."""

    import dspy

    # Use central LLM configuration
    llm_config = get_llm_config()
    try:
        lm = llm_config.get_dspy_lm(task="chat")
        dspy.configure(lm=lm)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Simple RAG module: question + context -> answer
    rag = dspy.Predict("question, context -> answer")

    def retrieve_contexts(q: str) -> list[str]:
        device = _device_str()
        st_model = _load_st_model(EMBED_MODEL, device=device)
        client = _qdrant_client()
        q_vec = (
            st_model.encode([q], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
        )
        prod_hits = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k)
        rev_hits = _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k)
        payloads = [p for _, _, p in (prod_hits + rev_hits)]
        texts: list[str] = []
        for p in payloads:
            if "description" in p:
                texts.append(f"Title: {p.get('title','')}\nDescription: {p.get('description','')}")
            elif "text" in p:
                texts.append(f"Title: {p.get('title','')}\nReview: {p.get('text','')}")
        return texts[:top_k]

    def answer_one(q: str) -> str:
        ctx = "\n\n".join(retrieve_contexts(q))
        pred = rag(question=q, context=ctx)
        return getattr(pred, "answer", "")

    if question:
        typer.echo(answer_one(question))
        raise typer.Exit(0)

    typer.echo("Chat with ShoppingAssistant. Type 'exit' to quit.")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nBye.")
            break
        if q.lower() in {"exit", "quit"}:
            typer.echo("Bye.")
            break
        if not q:
            continue
        a = answer_one(q)
        typer.echo(f"Assistant: {a}\n")


# -------------------------
# Evaluation Commands
# -------------------------

def _ensure_dirs() -> tuple[Path, Path]:
    results_dir = Path("eval/results")
    datasets_dir = Path("eval/datasets")
    results_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, datasets_dir


def _get_timestamp() -> str:
    """Generate a formatted timestamp for file naming: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _write_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(["| " + " | ".join(r) + " |" for r in rows])
    return "\n".join([head, sep, body])


def _load_jsonl(path: Path, max_samples: int | None = None, seed: int = 42) -> list[dict]:
    import random

    rng = random.Random(seed)
    rows = _read_jsonl(path)
    if max_samples is not None and len(rows) > max_samples:
        rows = rng.sample(rows, max_samples)
    return rows


def _to_context_text(payload: dict) -> str:
    if "description" in payload:
        return f"Title: {payload.get('title','')}\nDescription: {payload.get('description','')}"
    if "text" in payload:
        return f"Title: {payload.get('title','')}\nReview: {payload.get('text','')}"
    return json.dumps(payload)


@app.command("eval-search")
def eval_search(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="JSONL with {query}"),
    products_path: Path = typer.Option(DATA_PRODUCTS, exists=True),
    reviews_path: Path = typer.Option(DATA_REVIEWS, exists=True),
    top_k: int = typer.Option(20),
    rrf_k: int = typer.Option(60),
    rerank_top_k: int = typer.Option(30),
    variants: str = typer.Option("bm25,vec,rrf,rrf_ce"),
    max_samples: int = typer.Option(100),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate retrieval variants; logs metrics and report to eval/results and MLflow."""

    import mlflow
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    # Use metrics that don't require reference answers
    from ragas.metrics import ContextRelevance, ContextUtilization
    
    # Configure LLM for RAGAS evaluation
    llm_config = get_llm_config()
    try:
        llm_config.configure_ragas()
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    results_dir, _ = _ensure_dirs()
    device = _device_str()
    st_model = _load_st_model(EMBED_MODEL, device=device)
    client = _qdrant_client()
    bm25_prod, bm25_rev, id_to_product, id_to_review = _bm25_from_files(products_path, reviews_path)

    queries = _load_jsonl(dataset, max_samples=max_samples, seed=seed)
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    typer.echo("Starting eval-search with configuration:")
    typer.echo(f"- Dataset: {dataset}")
    typer.echo(f"- Samples (requested): {max_samples}; (loaded): {len(queries)}")
    typer.echo(f"- Variants: {', '.join(variant_list)}")
    typer.echo(f"- top_k={top_k} rrf_k={rrf_k} rerank_top_k={rerank_top_k}")
    typer.echo(f"- Device: {device}; Embed model: {EMBED_MODEL}")

    def run_variant(q: str, variant: str) -> list[dict]:
        """Return list of payload dicts for top_k contexts under a variant."""
        q_tokens = _tokenize(q)
        prod_ranked = []
        rev_ranked = []

        if variant in {"bm25", "rrf", "rrf_ce"}:
            bm25_prod_scores = bm25_prod.get_scores(q_tokens)
            bm25_rev_scores = bm25_rev.get_scores(q_tokens)
            prod_ranked = sorted(
                [(pid, float(s)) for pid, s in zip(id_to_product.keys(), bm25_prod_scores)],
                key=lambda x: x[1], reverse=True,
            )[:top_k]
            rev_ranked = sorted(
                [(rid, float(s)) for rid, s in zip(id_to_review.keys(), bm25_rev_scores)],
                key=lambda x: x[1], reverse=True,
            )[:top_k]

        prod_vec = []
        rev_vec = []
        if variant in {"vec", "rrf", "rrf_ce"}:
            q_vec = st_model.encode([q], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
            # Vector search returns UUIDs, need to map back to original IDs using payload
            prod_vec_raw = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k)
            rev_vec_raw = _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k)
            
            # Map UUID results to original_id from payload
            prod_vec = []
            for uuid, score, payload in prod_vec_raw:
                original_id = payload.get('original_id', payload.get('id', ''))
                if original_id:
                    prod_vec.append((original_id, score))
            
            rev_vec = []
            for uuid, score, payload in rev_vec_raw:
                original_id = payload.get('original_id', payload.get('id', ''))
                if original_id:
                    rev_vec.append((original_id, score))

        ids: list[str] = []
        if variant == "bm25":
            ids = [pid for pid, _ in prod_ranked] + [rid for rid, _ in rev_ranked]
        elif variant == "vec":
            ids = [pid for pid, _ in prod_vec] + [rid for rid, _ in rev_vec]
        else:
            fused = _rrf_fuse([prod_ranked, rev_ranked, prod_vec, rev_vec], k=rrf_k)
            fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
            if variant == "rrf_ce" and fused_sorted:
                k_ce = min(rerank_top_k, len(fused_sorted))
                candidates: list[tuple[str, str]] = []
                for _id, _ in fused_sorted[:k_ce]:
                    if _id.startswith("prod::") and _id in id_to_product:
                        p = id_to_product[_id]
                        candidates.append((_id, f"{p.get('title','')}\n{p.get('description','')}"))
                    elif _id.startswith("rev::") and _id in id_to_review:
                        r = id_to_review[_id]
                        candidates.append((_id, f"{r.get('title','')}\n{r.get('text','')}"))
                ranked = _cross_encoder_scores(CROSS_ENCODER_MODEL, device, q, candidates)
                ce_scores = {cid: score for cid, score in ranked}
                fused_sorted = sorted(fused_sorted, key=lambda x: (ce_scores.get(x[0], float("-inf")), x[1]), reverse=True)
            ids = [i for i, _ in fused_sorted]

        payloads: list[dict] = []
        for _id in ids[:top_k]:
            if _id.startswith("prod::") and _id in id_to_product:
                payloads.append(id_to_product[_id])
            elif _id.startswith("rev::") and _id in id_to_review:
                payloads.append(id_to_review[_id])
        return payloads

    timestamp = _get_timestamp()
    out_json = results_dir / f"search_{timestamp}.json"
    out_md = results_dir / f"search_{timestamp}.md"

    with mlflow.start_run(run_name=f"eval-search-{timestamp}"):
        mlflow.log_param("variants", ",".join(variant_list))
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("rrf_k", rrf_k)
        mlflow.log_param("rerank_top_k", rerank_top_k)
        mlflow.log_param("max_samples", max_samples)

        variant_to_rows: dict[str, list[dict]] = {v: [] for v in variant_list}
        typer.echo("Collecting contexts per variant...")
        variant_counts: dict[str, int] = {v: 0 for v in variant_list}
        start_collect = time.time()
        for idx, row in enumerate(queries, start=1):
            q = row.get("query") or row.get("question") or ""
            if not q:
                continue
            for variant in variant_list:
                payloads = run_variant(q, variant)
                contexts = [_to_context_text(p) for p in payloads[:top_k]]
                variant_to_rows[variant].append({"question": q, "contexts": contexts, "answer": ""})
                variant_counts[variant] += 1
            if idx % 10 == 0:
                elapsed = time.time() - start_collect
                typer.echo(
                    f"  processed {idx}/{len(queries)} queries in {elapsed:.1f}s; per-variant: "
                    + ", ".join([f"{v}={variant_counts[v]}" for v in variant_list])
                )

        # Evaluate with RAGAS metric that works without references
        aggregates: dict[str, dict[str, float]] = {}
        for variant, rows in variant_to_rows.items():
            if not rows:
                aggregates[variant] = {}
                continue
            ds = Dataset.from_list(rows)
            try:
                res = ragas_evaluate(ds, metrics=[ContextRelevance(), ContextUtilization()])
                scores: dict[str, float] = {}
                # Try to aggregate robustly across ragas versions
                try:
                    df = res.to_pandas()  # type: ignore[attr-defined]
                    for col in df.columns:
                        if col in {"question", "answer", "contexts", "ground_truth"}:
                            continue
                        try:
                            scores[col] = float(df[col].astype(float).mean())
                        except Exception:
                            pass
                except Exception:
                    pass
                # Fallback: dict-like
                if not scores:
                    try:
                        for k, v in dict(res).items():  # type: ignore[arg-type]
                            try:
                                scores[k] = float(v)
                            except Exception:
                                continue
                    except Exception:
                        scores = {}
            except Exception as e:
                typer.echo(f"Error evaluating {variant}: {e}", err=True)
                raise
            aggregates[variant] = scores
            for k, v in scores.items():
                mlflow.log_metric(f"{variant}_{k}", v)
            typer.echo(
                f"Evaluated {variant}: "
                + ", ".join([f"{m}={aggregates[variant].get(m, float('nan')):.4f}" for m in sorted(aggregates[variant].keys())])
                if aggregates[variant]
                else f"Evaluated {variant}: no scores"
            )

        # Capture all call parameters
        call_params = {
            "command": "eval-search",
            "timestamp": timestamp,
            "dataset": str(dataset),
            "products_path": str(products_path),
            "reviews_path": str(reviews_path),
            "top_k": top_k,
            "rrf_k": rrf_k,
            "rerank_top_k": rerank_top_k,
            "variants": variant_list,
            "max_samples": max_samples,
            "seed": seed,
            "samples_loaded": len(queries),
            "device": device,
            "embed_model": EMBED_MODEL,
            "cross_encoder_model": CROSS_ENCODER_MODEL,
        }
        
        report = {
            "call_parameters": call_params,
            "config": {
                "variants": variant_list,
                "top_k": top_k,
                "rrf_k": rrf_k,
                "rerank_top_k": rerank_top_k,
                "max_samples": max_samples,
                "dataset": str(dataset),
                "device": device,
                "embed_model": EMBED_MODEL,
            },
            "aggregates": aggregates,
        }
        _save_json(out_json, report)
        mlflow.log_artifact(str(out_json))

        # Markdown side-by-side table
        headers = ["metric"] + variant_list
        metrics_set = set()
        for v in variant_list:
            metrics_set.update(aggregates.get(v, {}).keys())
        rows_md: list[list[str]] = []
        for metric in sorted(metrics_set):
            row = [metric] + [f"{aggregates.get(v, {}).get(metric, float('nan')):.4f}" for v in variant_list]
            rows_md.append(row)
        md_config = (
            f"### Call Parameters\n"
            f"- **Command**: `eval-search`\n"
            f"- **Timestamp**: {timestamp}\n"
            f"- **Dataset**: `{dataset}`\n"
            f"- **Products**: `{products_path}`\n"
            f"- **Reviews**: `{reviews_path}`\n"
            f"- **Samples**: requested={max_samples}, loaded={len(queries)}\n"
            f"- **Variants**: {', '.join(variant_list)}\n"
            f"- **Search params**: top_k={top_k}, rrf_k={rrf_k}, rerank_top_k={rerank_top_k}\n"
            f"- **Seed**: {seed}\n"
            f"- **Device**: {device}\n"
            f"- **Embed model**: `{EMBED_MODEL}`\n"
            f"- **Cross-encoder**: `{CROSS_ENCODER_MODEL}`\n"
        )
        md = "# Search Evaluation Report\n\n" + md_config + "\n## Metrics\n" + _write_markdown_table(headers, rows_md) + "\n"
        out_md.write_text(md)
        mlflow.log_artifact(str(out_md))

    typer.echo(f"Wrote: {out_json}\nWrote: {out_md}")


@app.command("eval-chat")
def eval_chat(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="JSONL with {question}"),
    top_k: int = typer.Option(8),
    max_samples: int = typer.Option(50),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate single-turn chat using RAGAS metrics; logs to eval/results and MLflow."""

    import mlflow
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    import dspy

    # Configure LLMs for both DSPy chat and RAGAS evaluation
    llm_config = get_llm_config()
    try:
        llm_config.configure_ragas()  # For evaluation
        lm = llm_config.get_dspy_lm(task="chat")  # For generating answers
        dspy.configure(lm=lm)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    results_dir, _ = _ensure_dirs()
    device = _device_str()
    st_model = _load_st_model(EMBED_MODEL, device=device)
    client = _qdrant_client()

    rag = dspy.Predict("question, context -> answer")

    def retrieve(q: str) -> list[str]:
        q_vec = (
            st_model.encode([q], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
        )
        prod_hits = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k)
        rev_hits = _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k)
        payloads = [p for _, _, p in (prod_hits + rev_hits)]
        texts: list[str] = []
        for p in payloads:
            texts.append(_to_context_text(p))
        return texts[:top_k]

    rows_in = _load_jsonl(dataset, max_samples=max_samples, seed=seed)
    rows_eval: list[dict] = []
    for row in rows_in:
        q = row.get("question") or row.get("query") or ""
        if not q:
            continue
        ctxs = retrieve(q)
        pred = rag(question=q, context="\n\n".join(ctxs))
        ans = getattr(pred, "answer", "")
        rows_eval.append({
            "question": q,
            "contexts": ctxs,
            "answer": ans,
            "ground_truth": row.get("reference_answer") or "",
        })

    timestamp = _get_timestamp()
    out_json = results_dir / f"chat_{timestamp}.json"
    out_md = results_dir / f"chat_{timestamp}.md"

    with mlflow.start_run(run_name=f"eval-chat-{timestamp}"):
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("max_samples", max_samples)
        ds = Dataset.from_list(rows_eval)
        res = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        # Extract scores from EvaluationResult object
        scores: dict[str, float] = {}
        try:
            # Try to get scores from the result object
            if hasattr(res, 'scores'):
                # res.scores could be a list or dict
                if isinstance(res.scores, dict):
                    scores = {k: float(v) for k, v in res.scores.items()}
                elif isinstance(res.scores, list) and len(res.scores) > 0:
                    # If it's a list, it usually contains a single dict with all metrics
                    if isinstance(res.scores[0], dict):
                        scores = {k: float(v) if not pd.isna(v) else 0.0 for k, v in res.scores[0].items()}
                    else:
                        # Fallback: list of values
                        for i, metric in enumerate([faithfulness, answer_relevancy, context_precision, context_recall]):
                            metric_name = metric.name if hasattr(metric, 'name') else str(type(metric).__name__)
                            if i < len(res.scores):
                                scores[metric_name] = float(res.scores[i]) if not pd.isna(res.scores[i]) else 0.0
            
            # If no scores yet, try pandas dataframe
            if not scores and hasattr(res, 'to_pandas'):
                # Aggregate from dataframe
                df = res.to_pandas()
                for col in df.columns:
                    if col not in {"question", "answer", "contexts", "ground_truth", "reference"}:
                        try:
                            scores[col] = float(df[col].astype(float).mean())
                        except Exception:
                            pass
            
            # If still no scores, try direct dictionary/attribute access
            if not scores:
                for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    metric_name = metric.name if hasattr(metric, 'name') else str(type(metric).__name__)
                    # Try as dict key
                    if hasattr(res, '__getitem__'):
                        try:
                            scores[metric_name] = float(res[metric_name])
                        except (KeyError, TypeError):
                            pass
                    # Try as attribute
                    if metric_name not in scores and hasattr(res, metric_name):
                        try:
                            scores[metric_name] = float(getattr(res, metric_name))
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            typer.echo(f"Warning: Could not extract scores from evaluation result: {e}", err=True)
            # Debug: show what we actually got
            typer.echo(f"Result type: {type(res)}", err=True)
            if hasattr(res, 'scores'):
                typer.echo(f"res.scores type: {type(res.scores)}", err=True)
            scores = {"error": 0.0}
        
        for k, v in scores.items():
            mlflow.log_metric(k, v)

        # Capture all call parameters
        call_params = {
            "command": "eval-chat",
            "timestamp": timestamp,
            "dataset": str(dataset),
            "top_k": top_k,
            "max_samples": max_samples,
            "seed": seed,
            "samples_loaded": len(rows_in),
            "samples_evaluated": len(rows_eval),
            "device": device,
            "embed_model": EMBED_MODEL,
            "llm_model": llm_config.chat_model,
            "eval_model": llm_config.eval_model,
        }
        
        report = {
            "call_parameters": call_params,
            "config": {"top_k": top_k, "max_samples": max_samples},
            "aggregates": scores,
        }
        _save_json(out_json, report)
        mlflow.log_artifact(str(out_json))

        md_params = (
            f"### Call Parameters\n"
            f"- **Command**: `eval-chat`\n"
            f"- **Timestamp**: {timestamp}\n"
            f"- **Dataset**: `{dataset}`\n"
            f"- **Samples**: requested={max_samples}, loaded={len(rows_in)}, evaluated={len(rows_eval)}\n"
            f"- **Top-K contexts**: {top_k}\n"
            f"- **Seed**: {seed}\n"
            f"- **Device**: {device}\n"
            f"- **Embed model**: `{EMBED_MODEL}`\n"
            f"- **Chat LLM**: `{llm_config.chat_model}`\n"
            f"- **Eval LLM**: `{llm_config.eval_model}`\n\n"
        )
        
        md = "# Chat Evaluation Report\n\n" + md_params + "## Metrics\n" + _write_markdown_table(["metric", "score"], [[k, f"{v:.4f}"] for k, v in scores.items()]) + "\n"
        out_md.write_text(md)
        mlflow.log_artifact(str(out_md))

    typer.echo(f"Wrote: {out_json}\nWrote: {out_md}")

if __name__ == "__main__":
    # Allow both `python app/cli.py` and `python -m app.cli`
    try:
        import click as _click  # noqa: F401
    except Exception:  # pragma: no cover
        pass
    app()


