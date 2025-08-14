"""Ingestion command implementation, factored out from the monolithic CLI.

This module exposes a function that can be registered by the Typer app.
It retains the same parameters and behavior to preserve the CLI surface.
"""

from __future__ import annotations

from pathlib import Path
import time
import typer

from app.cli import (
    _device_str,
    _load_st_model,
    _qdrant_client,
    _qmodels,
    _read_jsonl,
    _build_product_docs,
    _build_review_docs,
    _embed_texts,
    _chunked,
    _format_seconds,
    _to_uuid_from_string,
)


def ingest_command(
    products_path: Path,
    reviews_path: Path,
    products_batch_size: int,
    reviews_batch_size: int,
    model_name: str,
    collection_products: str,
    collection_reviews: str,
    device: str,
) -> None:
    """Run ingestion with the same logic as the original CLI."""

    resolved_device = _device_str() if device == "auto" else device
    st_model = _load_st_model(model_name, device=resolved_device)
    client = _qdrant_client()
    qmodels = _qmodels()
    try:
        embed_dim = int(getattr(st_model, "get_sentence_embedding_dimension")())
    except Exception:
        from app.cli import VECTOR_SIZE

        embed_dim = VECTOR_SIZE

    # Ensure collections
    if not client.collection_exists(collection_products):
        client.recreate_collection(
            collection_name=collection_products,
            vectors_config=qmodels.VectorParams(size=embed_dim, distance=qmodels.Distance.COSINE),
        )
    if not client.collection_exists(collection_reviews):
        client.recreate_collection(
            collection_name=collection_reviews,
            vectors_config=qmodels.VectorParams(size=embed_dim, distance=qmodels.Distance.COSINE),
        )

    # Load files
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
        texts = [f"Title: {d.get('title','')}\nDescription: {d.get('description','')}".strip() for d in batch]
        vectors = _embed_texts(st_model, texts, device=resolved_device, batch_size=products_batch_size)
        ids = [d["id"] for d in batch]
        _upsert(collection_products, vectors, batch, ids)
        processed += len(batch)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        progress_pct = processed * 100 / len(product_docs)
        color = typer.colors.GREEN if progress_pct == 100 else typer.colors.BLUE
        typer.secho(
            f"[products] {processed}/{len(product_docs)} ({progress_pct:.1f}%) {rate:.1f}/s elapsed={_format_seconds(elapsed)}",
            fg=color,
        )

    prod_time = time.time() - start

    # Ingest reviews
    start_r = time.time()
    processed = 0
    for batch in _chunked(review_docs, reviews_batch_size):
        texts = [f"Title: {d.get('title','')}\nReview: {d.get('text','')}".strip() for d in batch]
        vectors = _embed_texts(st_model, texts, device=resolved_device, batch_size=reviews_batch_size)
        ids = [d["id"] for d in batch]
        _upsert(collection_reviews, vectors, batch, ids)
        processed += len(batch)
        elapsed = time.time() - start_r
        rate = processed / elapsed if elapsed > 0 else 0
        progress_pct = processed * 100 / len(review_docs)
        color = typer.colors.GREEN if progress_pct == 100 else typer.colors.BLUE
        typer.secho(
            f"[reviews] {processed}/{len(review_docs)} ({progress_pct:.1f}%) {rate:.1f}/s elapsed={_format_seconds(elapsed)}",
            fg=color,
        )

    rev_time = time.time() - start_r

    typer.secho("\nSummary:", fg=typer.colors.GREEN, bold=True)
    typer.secho(f" Ingested products: {len(product_docs)} in {_format_seconds(prod_time)}", fg=typer.colors.WHITE)
    typer.secho(f" Ingested reviews: {len(review_docs)} in {_format_seconds(rev_time)}", fg=typer.colors.WHITE)
    typer.secho(f" Device: {resolved_device} • Vector dim: {embed_dim} • Model: {model_name}", fg=typer.colors.WHITE)


