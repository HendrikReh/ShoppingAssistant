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
from pathlib import Path
from typing import Iterable, List, Tuple

import typer


# Third-party deps loaded lazily in functions to speed up CLI startup

app = typer.Typer(help="ShoppingAssistant CLI")


# Defaults aligned with notebooks
DATA_PRODUCTS = Path("data/top_1000_products.jsonl")
DATA_REVIEWS = Path("data/100_top_reviews_of_the_top_1000_products.jsonl")
COLLECTION_PRODUCTS = "products_gte_large"
COLLECTION_REVIEWS = "reviews_gte_large"
EMBED_MODEL = "thenlper/gte-large"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
        for line in fp:
            rows.append(json.loads(line.strip()))
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

    # Vectors
    q_vec = st_model.encode([query], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
    prod_vec = [(pid, s) for pid, s, _ in _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k)]
    rev_vec = [(rid, s) for rid, s, _ in _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k)]

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

    # Configure LM (assumes OPENAI_API_KEY is set) per memory
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

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


if __name__ == "__main__":
    # Allow both `python app/cli.py` and `python -m app.cli`
    try:
        import click as _click  # noqa: F401
    except Exception:  # pragma: no cover
        pass
    app()


