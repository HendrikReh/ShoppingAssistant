import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import os

    import polars as pl
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    DATA_PRODUCTS = "data/top_1000_products.jsonl"
    DATA_REVIEWS = "data/100_top_reviews_of_the_top_1000_products.jsonl"
    COLLECTION_PRODUCTS = "products_gte_large"
    COLLECTION_REVIEWS = "reviews_gte_large"
    MODEL_NAME = "thenlper/gte-large"
    VECTOR_SIZE = 1024

    header_panel = mo.vstack([
        mo.md("# Ingest and Embed: Products and Reviews → Qdrant (gte-large)"),
        mo.callout(
            mo.md(
                """
                ### This notebook performs four steps:
                - **Controls**: Set batch sizes and press "Start Ingestion".
                - **Embedding**: Encode product and review texts with `thenlper/gte-large` on the selected device.
                - **Upsert**: Write vectors and payloads to Qdrant with deterministic UUIDs (`original_id` kept in payload).
                - **Summary**: Display throughput, total counts, device, and collection details at the end.

                Prerequisites: Qdrant running locally (`docker-compose up`) and data files present in `data/`.
                """
            ),
            kind="info",
        ),
    ], align="stretch", gap=0.5)
    header_panel
    return (
        COLLECTION_PRODUCTS,
        COLLECTION_REVIEWS,
        DATA_PRODUCTS,
        DATA_REVIEWS,
        MODEL_NAME,
        VECTOR_SIZE,
        json,
        mo,
        os,
        pl,
        qmodels,
        QdrantClient,
    )


@app.cell
def _(MODEL_NAME):
    import torch

    device = (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )

    def load_model(model_name: str, target_device: str):
        # Local import avoids cross-cell global name collisions
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=target_device)

    model = load_model(MODEL_NAME, device)
    info = {"device": device, "torch_version": torch.__version__}
    return device, info, load_model, model


@app.cell
def _(
    QdrantClient,
    qmodels,
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
    VECTOR_SIZE,
):
    client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)

    # Show current server status and collections before any changes
    try:
        existing = client.get_collections().collections
        existing_list = [c.name for c in existing]
    except Exception as e:
        existing_list = []

    for name in [COLLECTION_PRODUCTS, COLLECTION_REVIEWS]:
        if not client.collection_exists(name):
            client.recreate_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    # Fetch stats after ensuring collections
    try:
        after = client.get_collections().collections
        after_list = [c.name for c in after]
    except Exception:
        after_list = []

    from marimo import md as _md
    _md(
        f"""
    ## Qdrant Status
    - Existing collections (before): {existing_list}
    - Collections (after ensure): {after_list}
    """
    )

    return (client,)


@app.cell
def _(
    DATA_PRODUCTS,
    DATA_REVIEWS,
    json,
    pl,
):
    def read_jsonl(path: str) -> list[dict]:
        records: list[dict] = []
        with open(path, "r") as fp:
            for line in fp:
                records.append(json.loads(line.strip()))
        return records

    products_df = pl.DataFrame(read_jsonl(DATA_PRODUCTS))
    reviews_df = pl.DataFrame(read_jsonl(DATA_REVIEWS))

    # Normalize product fields used downstream
    products_df = products_df.with_columns(
        [
            pl.when(pl.col("review_count").is_not_null())
            .then(pl.col("review_count"))
            .otherwise(pl.col("rating_number"))
            .alias("num_reviews"),
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").alias("title"),
        ]
    )

    reviews_df = reviews_df.with_columns(
        [
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").alias("title"),
            pl.col("text").cast(pl.Utf8, strict=False).fill_null("").alias("text"),
        ]
    )

    return products_df, reviews_df


@app.cell
def _(mo, products_df, reviews_df):
    mo.md(
        f"""
    ## Dataset Snapshot
    - Products: {products_df.height:,}
    - Reviews: {reviews_df.height:,}
    """
    )
    return


@app.cell
def _():
    from typing import TypedDict
    import uuid

    class ProductDoc(TypedDict):
        id: str
        parent_asin: str
        title: str
        description: str
        average_rating: float | None
        num_reviews: int | None

    class ReviewDoc(TypedDict):
        id: str
        parent_asin: str
        title: str
        text: str
        rating: float | None
        helpful_vote: int | None

    return ProductDoc, ReviewDoc, uuid


@app.cell
def _(
    ProductDoc,
    ReviewDoc,
    mo,
    pl,
):
    def build_product_docs(df: pl.DataFrame) -> list[ProductDoc]:
        docs: list[ProductDoc] = []
        for row in df.iter_rows(named=True):
            parent_asin = str(row.get("parent_asin", ""))
            doc: ProductDoc = {
                "id": f"prod::{parent_asin}",
                "parent_asin": parent_asin,
                "title": row.get("title") or "",
                "description": (row.get("description") or ""),
                "average_rating": row.get("average_rating"),
                "num_reviews": row.get("num_reviews"),
            }
            docs.append(doc)
        return docs

    def build_review_docs(df: pl.DataFrame) -> list[ReviewDoc]:
        docs: list[ReviewDoc] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            parent_asin = str(row.get("parent_asin", ""))
            doc: ReviewDoc = {
                "id": f"rev::{parent_asin}::{i}",
                "parent_asin": parent_asin,
                "title": row.get("title") or "",
                "text": row.get("text") or "",
                "rating": row.get("rating"),
                "helpful_vote": row.get("helpful_vote"),
            }
            docs.append(doc)
        return docs

    mo.md("Defined normalization helpers for products and reviews.")
    return build_product_docs, build_review_docs


@app.cell
def _(
    device,
    model,
    pl,
):
    def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
        # SentenceTransformer handles batching internally if specified
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            device=device,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def to_text_for_product(title: str, description: str) -> str:
        return f"Title: {title}\nDescription: {description}".strip()

    def to_text_for_review(title: str, text: str) -> str:
        return f"Title: {title}\nReview: {text}".strip()

    return device, embed_texts, to_text_for_product, to_text_for_review


@app.cell
def _(
    QdrantClient,
    qmodels,
    client,
    device,
    embed_texts,
    build_product_docs,
    build_review_docs,
    products_df,
    reviews_df,
    to_text_for_product,
    to_text_for_review,
):
    def upsert_points(collection: str, vectors: list[list[float]], payloads: list[dict], ids: list[str]) -> None:
        # Convert string IDs to UUIDs deterministically based on the string content
        import hashlib
        import uuid

        uuid_ids = []
        for str_id in ids:
            # Create deterministic UUID from string ID
            hash_obj = hashlib.md5(str_id.encode())
            uuid_ids.append(str(uuid.UUID(hash_obj.hexdigest())))

        client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(id=uid, vector=vec, payload={**payload, "original_id": orig_id})
                for uid, vec, payload, orig_id in zip(uuid_ids, vectors, payloads, ids)
            ],
        )

    def chunked(iterable, n: int):
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == n:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def format_time(seconds: float) -> str:
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    # Build docs (lightweight)
    product_docs = build_product_docs(products_df)
    review_docs = build_review_docs(reviews_df)

    total_products = len(product_docs)
    total_reviews = len(review_docs)

    return (
        upsert_points,
        chunked,
        format_time,
        product_docs,
        review_docs,
        total_products,
        total_reviews,
        to_text_for_product,
        to_text_for_review,
        embed_texts,
        device,
    )

@app.cell
def _(mo):
    # UI controls must be created in a separate cell from where their values are read
    start_button = mo.ui.run_button(label="Start Ingestion", tooltip="Embed and upsert all products & reviews")
    params = mo.ui.dictionary({
        "Product batch size": mo.ui.number(32, 1024, value=128, step=32),
        "Review batch size": mo.ui.number(64, 2048, value=256, step=64),
    })
    return start_button, params

@app.cell
def _(
    mo,
    start_button,
    params,
    upsert_points,
    chunked,
    format_time,
    product_docs,
    review_docs,
    total_products,
    total_reviews,
    to_text_for_product,
    to_text_for_review,
    embed_texts,
    device,
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
    VECTOR_SIZE,
    MODEL_NAME,
):
    import time as _time

    controls_panel = mo.vstack([
        mo.md("### Controls"),
        mo.hstack([start_button, params], gap=1),
    ], align="stretch", gap=0.5)

    summary_panel = mo.callout("Ready to run ingestion. Adjust batch sizes and press Start.", kind="info")

    if start_button.value:
        # Parse params
        product_batch_size = int(params.value["Product batch size"]) if isinstance(params.value, dict) else 128
        review_batch_size = int(params.value["Review batch size"]) if isinstance(params.value, dict) else 256

        # Ingest products
        print(f"[products] device={device} total={total_products}")
        product_start_time = _time.time()
        processed_products = 0

        for batch_idx, batch in enumerate(chunked(product_docs, product_batch_size)):
            texts = [to_text_for_product(d["title"], d["description"]) for d in batch]
            vectors = embed_texts(texts)
            ids = [d["id"] for d in batch]
            upsert_points(COLLECTION_PRODUCTS, vectors, batch, ids)

            processed_products += len(batch)
            elapsed = _time.time() - product_start_time
            rate = processed_products / elapsed if elapsed > 0 else 0
            eta = (total_products - processed_products) / rate if rate > 0 else 0
            progress_pct = processed_products * 100 / total_products
            print(
                f"[products] batch={batch_idx + 1} count={processed_products}/{total_products} ({progress_pct:.1f}%) "
                f"speed={rate:.1f}/s elapsed={format_time(elapsed)} eta~{format_time(eta)}"
            )

        product_time = _time.time() - product_start_time

        # Ingest reviews
        print(f"[reviews] total={total_reviews} starting…")
        review_start_time = _time.time()
        processed_reviews = 0

        for batch_idx, batch in enumerate(chunked(review_docs, review_batch_size)):
            texts = [to_text_for_review(d["title"], d["text"]) for d in batch]
            vectors = embed_texts(texts)
            ids = [d["id"] for d in batch]
            upsert_points(COLLECTION_REVIEWS, vectors, batch, ids)

            processed_reviews += len(batch)
            elapsed = _time.time() - review_start_time
            rate = processed_reviews / elapsed if elapsed > 0 else 0
            eta = (total_reviews - processed_reviews) / rate if rate > 0 else 0
            progress_pct = processed_reviews * 100 / total_reviews
            print(
                f"[reviews] batch={batch_idx + 1} count={processed_reviews}/{total_reviews} ({progress_pct:.1f}%) "
                f"speed={rate:.1f}/s elapsed={format_time(elapsed)} eta~{format_time(eta)}"
            )

        review_time = _time.time() - review_start_time
        total_time = product_time + review_time

        summary_md = f"""
        ### ✅ Embedding Complete

        - **Products:** {total_products:,} embedded in {format_time(product_time)} ({total_products/product_time:.1f} items/sec)
        - **Reviews:** {total_reviews:,} embedded in {format_time(review_time)} ({total_reviews/review_time:.1f} items/sec)
        - **Total Time:** {format_time(total_time)}
        - **Device:** {device}

        **Storage**  
        - Products: `{COLLECTION_PRODUCTS}` • Reviews: `{COLLECTION_REVIEWS}`  
        - Vector dim: {VECTOR_SIZE} • Model: {MODEL_NAME}
        """
        summary_panel = mo.md(summary_md)

    mo.ui.tabs({
        "Controls": controls_panel,
        "Summary": summary_panel,
    })

    return None


@app.cell
def _(mo):
    mo.md(
        """
    ## Notes on embedding choices
    - Model: gte-large (1024-dim) chosen for strong retrieval performance on MTEB and efficient size.
    - Normalization: embeddings L2-normalized for cosine similarity in Qdrant.
    - Text construction:
      - Products: title + description (if available). Consider adding key features when present.
      - Reviews: title + body text; captures sentiment and specifics for RAG grounding.
    - Collections: separate `products` and `reviews` to support hybrid retrieval and different scoring.
    - Metadata payloads include `parent_asin`, ratings, counts, etc., enabling filtered or boosted search.
    - Batch sizes: 128 for products, 256 for reviews; tune based on memory.
    - Next: add reranking (e.g., cross-encoder), BM25 fusion, and freshness/helpfulness boosts.
    """
    )
    return


if __name__ == "__main__":
    app.run()


