import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    from typing import Dict, List, Tuple

    import polars as pl
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from rank_bm25 import BM25Okapi

    DATA_PRODUCTS = "data/top_1000_products.jsonl"
    DATA_REVIEWS = "data/100_top_reviews_of_the_top_1000_products.jsonl"
    COLLECTION_PRODUCTS = "products_gte_large"
    COLLECTION_REVIEWS = "reviews_gte_large"
    MODEL_NAME = "thenlper/gte-large"
    VECTOR_SIZE = 1024

    mo.md("# Hybrid Retrieval with BM25 + Vectors (RRF)")
    return (
        BM25Okapi,
        COLLECTION_PRODUCTS,
        COLLECTION_REVIEWS,
        DATA_PRODUCTS,
        DATA_REVIEWS,
        MODEL_NAME,
        VECTOR_SIZE,
        mo,
        pl,
        qmodels,
        QdrantClient,
        re,
    )


@app.cell
def _(MODEL_NAME):
    import torch
    from importlib import import_module

    device = (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )

    def load_model(model_name: str, target_device: str):
        st = import_module("sentence_transformers")
        SentenceTransformer = getattr(st, "SentenceTransformer")
        return SentenceTransformer(model_name, device=target_device)

    model = load_model(MODEL_NAME, device)
    return device, load_model, model


@app.cell
def _(QdrantClient, qmodels, COLLECTION_PRODUCTS, COLLECTION_REVIEWS, VECTOR_SIZE):
    client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
    # Ensure collections exist
    for name in [COLLECTION_PRODUCTS, COLLECTION_REVIEWS]:
        if not client.collection_exists(name):
            client.recreate_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            )
    return (client,)


@app.cell
def _(DATA_PRODUCTS, DATA_REVIEWS, pl):
    import json

    def read_jsonl(path: str) -> list[dict]:
        rows: list[dict] = []
        with open(path, "r") as fp:
            for line in fp:
                rows.append(json.loads(line.strip()))
        return rows

    products_df = pl.DataFrame(read_jsonl(DATA_PRODUCTS))
    # Normalize review counts and textual fields; handle list-type descriptions
    products_df = products_df.with_columns(
        [
            pl.when(pl.col("review_count").is_not_null())
            .then(pl.col("review_count"))
            .otherwise(pl.col("rating_number"))
            .alias("num_reviews"),
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").alias("title"),
            pl.col("description")
            .map_elements(
                lambda v: " ".join(v) if isinstance(v, list) else (v if isinstance(v, str) else ""),
                return_dtype=pl.Utf8,
            )
            .fill_null("")
            .alias("description"),
        ]
    )

    reviews_df = pl.DataFrame(read_jsonl(DATA_REVIEWS)).with_columns(
        [
            pl.col("title").cast(pl.Utf8, strict=False).fill_null("").alias("title"),
            pl.col("text").cast(pl.Utf8, strict=False).fill_null("").alias("text"),
        ]
    )

    return products_df, reviews_df


@app.cell
def _(BM25Okapi, pl, products_df, reviews_df, re):
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower())

    # Build corpora and doc maps
    prod_docs = []
    prod_ids = []
    for row in products_df.iter_rows(named=True):
        prod_ids.append(f"prod::{row.get('parent_asin','')}")
        prod_docs.append(tokenize(f"{row.get('title','')}\n{row.get('description','')}") )

    rev_docs = []
    rev_ids = []
    for i, row in enumerate(reviews_df.iter_rows(named=True)):
        rev_ids.append(f"rev::{row.get('parent_asin','')}::{i}")
        rev_docs.append(tokenize(f"{row.get('title','')}\n{row.get('text','')}") )

    bm25_prod = BM25Okapi(prod_docs)
    bm25_rev = BM25Okapi(rev_docs)

    # Lightweight lookup maps for display
    id_to_product = {pid: {"id": pid, **row} for pid, row in zip(prod_ids, products_df.to_dicts())}
    id_to_review = {rid: {"id": rid, **row} for rid, row in zip(rev_ids, reviews_df.to_dicts())}

    return bm25_prod, bm25_rev, id_to_product, id_to_review, tokenize


@app.cell
def _(device, model):
    def embed_query(text: str) -> list[float]:
        vec = model.encode([text], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0]
        return vec.tolist()

    return (embed_query,)


@app.cell
def _(
    client,
    qmodels,
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
):
    def vector_search(collection: str, vector: list[float], top_k: int = 20):
        hits = client.search(
            collection_name=collection,
            query_vector=vector,
            with_payload=True,
            limit=top_k,
        )
        # Convert to uniform list
        return [(str(hit.id), float(hit.score), hit.payload) for hit in hits]

    return (vector_search,)


@app.cell
def _():
    def rrf_fuse(result_lists: list[list[tuple]], k: int = 60) -> dict:
        # result_lists: lists of (id, score, payload?) or (id, score)
        # Use ranks only; ignore raw scores for stability
        fused: dict = {}
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                _id = item[0]
                fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank)
        return fused

    return (rrf_fuse,)


@app.cell
def _(
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
    bm25_prod,
    bm25_rev,
    embed_query,
    id_to_product,
    id_to_review,
    mo,
    tokenize,
    vector_search,
    rrf_fuse,
    pl,
):
    query = mo.ui.text(placeholder="Search products and reviews...", label="Query")
    top_k = mo.ui.slider(5, 50, 20, label="Top-K per modality")
    k_rrf = mo.ui.slider(10, 100, 60, label="RRF k")

    # Display controls cell-only
    mo.md(f"""
    ## Query
    {query}
    {top_k}
    {k_rrf}
    """)
    return k_rrf, query, top_k


@app.cell
def _(
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
    bm25_prod,
    bm25_rev,
    embed_query,
    id_to_product,
    id_to_review,
    mo,
    query,
    tokenize,
    top_k,
    vector_search,
    k_rrf,
    rrf_fuse,
    pl,
):
    view = None
    if not query.value:
        view = mo.md("Enter a query above.")
    else:
        # BM25
        q_tokens = tokenize(query.value)
        bm25_prod_scores = bm25_prod.get_scores(q_tokens)
        bm25_rev_scores = bm25_rev.get_scores(q_tokens)

        prod_ranked = sorted(
            [(pid, float(score)) for pid, score in zip(id_to_product.keys(), bm25_prod_scores)],
            key=lambda x: x[1],
            reverse=True,
        )[: top_k.value]

        rev_ranked = sorted(
            [(rid, float(score)) for rid, score in zip(id_to_review.keys(), bm25_rev_scores)],
            key=lambda x: x[1],
            reverse=True,
        )[: top_k.value]

        # Vector
        q_vec = embed_query(query.value)
        prod_vec = [(pid, s) for pid, s, _ in vector_search(COLLECTION_PRODUCTS, q_vec, top_k=top_k.value)]
        rev_vec = [(rid, s) for rid, s, _ in vector_search(COLLECTION_REVIEWS, q_vec, top_k=top_k.value)]

        # RRF across four lists
        fused_scores = rrf_fuse([prod_ranked, rev_ranked, prod_vec, rev_vec], k=k_rrf.value)
        fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:50]

        # Build display rows
        rows = []
        for _id, score in fused_sorted:
            if _id.startswith("prod::") and _id in id_to_product:
                p = id_to_product[_id]
                rows.append({
                    "type": "product",
                    "id": _id,
                    "title": p.get("title", "")[:120],
                    "rating": p.get("average_rating"),
                    "reviews": p.get("num_reviews"),
                    "rrf": round(score, 6),
                })
            elif _id.startswith("rev::") and _id in id_to_review:
                r = id_to_review[_id]
                rows.append({
                    "type": "review",
                    "id": _id,
                    "title": r.get("title", "")[:120],
                    "rating": r.get("rating"),
                    "helpful": r.get("helpful_vote"),
                    "rrf": round(score, 6),
                })

        result_df = pl.DataFrame(rows)
        if result_df.height == 0:
            view = mo.md("No results found.")
        else:
            table = mo.ui.table(result_df.to_pandas(), pagination=False)
            view = mo.md(
                f"""
            ### Fused Results (RRF)
            {table}
            """
            )

    view
    return


if __name__ == "__main__":
    app.run()


