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
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress LiteLLM verbose logging
import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM.Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
from typing import Iterable, List, Tuple

import typer
from app.llm_config import get_llm_config


# Third-party deps loaded lazily in functions to speed up CLI startup

app = typer.Typer(help="ShoppingAssistant CLI")


# Defaults aligned with notebooks
DATA_PRODUCTS = Path("data/top_1000_products.jsonl")
DATA_REVIEWS = Path("data/100_top_reviews_of_the_top_1000_products.jsonl")
COLLECTION_PRODUCTS = "products_minilm"
COLLECTION_REVIEWS = "reviews_minilm"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
VECTOR_SIZE = 384 # MiniLM has 384 dimensions


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


# Global cache for sentence transformer models
_st_model_cache = {}

def _load_st_model(model_name: str, device: str | None = None):
  """Load or get cached sentence transformer model."""
  cache_key = f"{model_name}_{device}"
  if cache_key not in _st_model_cache:
    try:
      from sentence_transformers import SentenceTransformer
      import os
      
      # Try to load from cache first
      cache_folder = os.path.expanduser(f"~/.cache/torch/sentence_transformers/{model_name.replace('/', '_')}")
      if os.path.exists(cache_folder):
        typer.secho(f"    Loading from cache: {cache_folder}", fg=typer.colors.BRIGHT_BLACK)
        _st_model_cache[cache_key] = SentenceTransformer(cache_folder, device=device)
      else:
        # Try to download from Hugging Face
        _st_model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    except Exception as e:
      if "Failed to resolve" in str(e) or "Connection" in str(e):
        typer.secho(f"\n⚠️  Cannot connect to Hugging Face to download model: {model_name}", fg=typer.colors.YELLOW)
        typer.secho("    This could be due to:", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    • No internet connection", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    • Hugging Face is down", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    • Firewall/proxy blocking access", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("\n    Solutions:", fg=typer.colors.GREEN)
        typer.secho("    1. Check your internet connection", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    2. Try again later if Hugging Face is down", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    3. Pre-download the model:", fg=typer.colors.BRIGHT_BLACK)
        typer.secho(f"       python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\"", fg=typer.colors.BRIGHT_BLACK)
        typer.secho("    4. Use a different embedding model in pyproject.toml", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(1)
      else:
        typer.secho(f"\n❌ Error loading model {model_name}: {e}", fg=typer.colors.RED)
        raise
  return _st_model_cache[cache_key]


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

  typer.secho("Starting ingestion with configuration:", fg=typer.colors.CYAN, bold=True)
  typer.secho(f" Device: {device}", fg=typer.colors.WHITE)
  typer.secho(f" Embed model: {model_name}", fg=typer.colors.WHITE)
  typer.secho(f" Products path: {products_path}", fg=typer.colors.WHITE)
  typer.secho(f" Reviews path: {reviews_path}", fg=typer.colors.WHITE)
  typer.secho(f" Products batch size: {products_batch_size}", fg=typer.colors.WHITE)
  typer.secho(f" Reviews batch size: {reviews_batch_size}", fg=typer.colors.WHITE)
  typer.secho(f" Qdrant collections: products={collection_products} reviews={collection_reviews}", fg=typer.colors.WHITE)
  typer.secho("\nPlan:", fg=typer.colors.YELLOW, bold=True)
  typer.echo(" 1) Read JSONL files")
  typer.echo(" 2) Build normalized docs (products, reviews)")
  typer.echo(" 3) Embed texts with Sentence-Transformers on the selected device")
  typer.echo(" 4) Upsert vectors + payloads into Qdrant using deterministic UUIDs")
  typer.echo(" 5) Summarize system state (point counts per collection)")

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
    progress_pct = processed*100/len(product_docs)
    color = typer.colors.GREEN if progress_pct == 100 else typer.colors.BLUE
    typer.secho(
      f"[products] {processed}/{len(product_docs)} ({progress_pct:.1f}%) "
      f"{rate:.1f}/s elapsed={_format_seconds(elapsed)}",
      fg=color
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
    progress_pct = processed*100/len(review_docs)
    color = typer.colors.GREEN if progress_pct == 100 else typer.colors.BLUE
    typer.secho(
      f"[reviews] {processed}/{len(review_docs)} ({progress_pct:.1f}%) "
      f"{rate:.1f}/s elapsed={_format_seconds(elapsed)}",
      fg=color
    )

  rev_time = time.time() - start_r
  # Summarize system state
  prod_count = None
  rev_count = None
  try:
    prod_count = client.count(collection_products, exact=True).count # type: ignore[attr-defined]
  except Exception:
    pass
  try:
    rev_count = client.count(collection_reviews, exact=True).count # type: ignore[attr-defined]
  except Exception:
    pass

  typer.secho("\nSummary:", fg=typer.colors.GREEN, bold=True)
  typer.secho(f" Ingested products: {len(product_docs)} in {_format_seconds(prod_time)}", fg=typer.colors.WHITE)
  typer.secho(f" Ingested reviews: {len(review_docs)} in {_format_seconds(rev_time)}", fg=typer.colors.WHITE)
  typer.secho(f" Device: {device} • Vector dim: {VECTOR_SIZE} • Model: {model_name}", fg=typer.colors.WHITE)
  if prod_count is not None:
    typer.secho(f" Qdrant points: {collection_products}={prod_count}", fg=typer.colors.CYAN)
  if rev_count is not None:
    typer.secho(f" Qdrant points: {collection_reviews}={rev_count}", fg=typer.colors.CYAN)


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


def _rrf_fuse(result_lists: list[list[tuple[str, float]]], k: int = 60, product_boost: float = 1.5) -> dict[str, float]:
  """RRF fusion with optional product boost.
  
  Args:
    result_lists: List of ranked results from different methods
    k: RRF k parameter
    product_boost: Multiplicative boost for product results (default 1.5)
  """
  fused: dict[str, float] = {}
  for results in result_lists:
    for rank, item in enumerate(results, start=1):
      _id = item[0]
      score = 1.0 / (k + rank)
      # Boost products over reviews
      if _id.startswith("prod::") and product_boost > 1.0:
        score *= product_boost
      fused[_id] = fused.get(_id, 0.0) + score
  return fused


def _vector_search(client, collection: str, vector: list[float], top_k: int = 20) -> list[tuple[str, float, dict]]:
  # Use query_points to avoid deprecation warning
  try:
    # Try the newer query_points API first
    res = client.query_points(
      collection_name=collection,
      query=vector, # Use query instead of query_vector
      with_payload=True,
      limit=top_k,
    )
    points = getattr(res, "points", res)
    return [(str(p.id), float(p.score), p.payload) for p in points]
  except Exception:
    # Fallback to search if query_points doesn't work
    try:
      hits = client.search(
        collection_name=collection,
        query_vector=vector,
        with_payload=True,
        limit=top_k,
      )
      return [(str(h.id), float(h.score), h.payload) for h in hits]
    except Exception as e:
      # Last resort - return empty list
      return []


# Global cache for cross-encoder model
_cross_encoder_cache = {}

def _get_cross_encoder(model_name: str, device: str):
  """Get or create cached cross-encoder model."""
  cache_key = f"{model_name}_{device}"
  if cache_key not in _cross_encoder_cache:
    typer.secho(f"  • Loading cross-encoder model (first time only)...", fg=typer.colors.BRIGHT_BLACK)
    try:
      from sentence_transformers import CrossEncoder
      import os
      
      # Try to load from cache first
      cache_folder = os.path.expanduser(f"~/.cache/torch/sentence_transformers/{model_name.replace('/', '_')}")
      if os.path.exists(cache_folder):
        typer.secho(f"    Loading from cache: {cache_folder}", fg=typer.colors.BRIGHT_BLACK)
        _cross_encoder_cache[cache_key] = CrossEncoder(cache_folder, device=device)
      else:
        _cross_encoder_cache[cache_key] = CrossEncoder(model_name, device=device)
    except Exception as e:
      if "Failed to resolve" in str(e) or "Connection" in str(e):
        typer.secho(f"\n⚠️  Cannot download cross-encoder model: {model_name}", fg=typer.colors.YELLOW)
        typer.secho("    Continuing without reranking (results may be less relevant)", fg=typer.colors.BRIGHT_BLACK)
        return None
      else:
        raise
  return _cross_encoder_cache[cache_key]

def _cross_encoder_scores(model_name: str, device: str, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
  if not candidates:
    return []
  ce = _get_cross_encoder(model_name, device)
  if ce is None:
    # Fallback: return candidates with their original order as scores
    return [(cid, float(len(candidates) - i)) for i, (cid, _) in enumerate(candidates)]
  pairs = [(query, text) for _, text in candidates]
  scores = ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
  ranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
  ranked.sort(key=lambda x: x[1], reverse=True)
  return ranked


@app.command()
def search(
  query: str = typer.Option(None, help="User query (if omitted, starts interactive mode)"),
  products_path: Path = typer.Option(DATA_PRODUCTS, exists=True, readable=True),
  reviews_path: Path = typer.Option(DATA_REVIEWS, exists=True, readable=True),
  top_k: int = typer.Option(20, min=1, help="Top-K per modality"),
  rrf_k: int = typer.Option(60, help="RRF k"),
  rerank: bool = typer.Option(True, help="Use cross-encoder rerank"),
  rerank_top_k: int = typer.Option(30, help="Top-K to rerank after fusion"),
  enable_web: bool = typer.Option(False, "--web/--no-web", help="Enable web search enhancement"),
  web_only: bool = typer.Option(False, help="Use only web search (no local)"),
) -> None:
  """Hybrid retrieval (BM25 + vectors) with optional cross-encoder rerank."""

  # Show initialization progress
  typer.secho("Initializing search components...", fg=typer.colors.YELLOW, italic=True)
  
  device = _device_str()
  typer.secho(f"  • Device: {device}", fg=typer.colors.BRIGHT_BLACK)
  
  typer.secho("  • Loading embedding model...", fg=typer.colors.BRIGHT_BLACK)
  st_model = _load_st_model(EMBED_MODEL, device=device)
  
  typer.secho("  • Connecting to Qdrant...", fg=typer.colors.BRIGHT_BLACK)
  client = _qdrant_client()

  # Load data once for the session
  typer.secho("  • Loading product and review data...", fg=typer.colors.BRIGHT_BLACK)
  bm25_prod, bm25_rev, id_to_product, id_to_review = _bm25_from_files(products_path, reviews_path)
  
  typer.secho("Ready!\n", fg=typer.colors.GREEN)
  
  # Initialize web search if enabled
  web_agent = None
  orchestrator = None
  if enable_web or web_only:
    import os
    from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig, WebSearchCache
    from app.hybrid_retrieval_orchestrator import HybridRetrievalOrchestrator
    import redis
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
      typer.secho("Error: TAVILY_API_KEY not found in environment", fg=typer.colors.RED)
      raise typer.Exit(1)
    
    # Set up web search with caching
    try:
      redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
      redis_client.ping()
      cache = WebSearchCache(redis_client)
      typer.secho("Web search cache enabled (Redis)", fg=typer.colors.GREEN, dim=True)
    except:
      cache = None
      typer.secho("Web search cache disabled (Redis not available)", fg=typer.colors.YELLOW, dim=True)
    
    config = WebSearchConfig(api_key=tavily_key, enable_web_search=True)
    web_agent = TavilyWebSearchAgent(config, cache)
    
    # Create orchestrator for hybrid retrieval
    orchestrator = HybridRetrievalOrchestrator(
      st_model=st_model,
      qdrant_client=client,
      bm25_products=bm25_prod,
      bm25_reviews=bm25_rev,
      web_search_agent=web_agent,
      enable_web_search=True
    )
    
    if web_only:
      typer.secho("Web-only search mode enabled", fg=typer.colors.CYAN)
  
  def perform_search(search_query: str) -> None:
    """Execute a single search."""
    
    # Use orchestrator if web search is enabled
    if orchestrator and (enable_web or web_only):
      from app.hybrid_retrieval_orchestrator import HybridResult
      
      # Use orchestrator for hybrid retrieval
      results = orchestrator.retrieve(
        query=search_query,
        top_k=top_k,
        force_web=web_only
      )
      
      # Display orchestrated results
      typer.secho(f"\nSearch Results for: '{search_query}'", fg=typer.colors.CYAN, bold=True)
      
      # Count sources
      web_count = sum(1 for r in results if r.is_web)
      local_count = len(results) - web_count
      
      source_info = []
      if local_count > 0:
        source_info.append(f"{local_count} local")
      if web_count > 0:
        source_info.append(f"{web_count} web")
      
      typer.secho(f"Found {len(results)} results ({', '.join(source_info)})\n", fg=typer.colors.WHITE)
      
      for idx, result in enumerate(results[:20], 1):
        # Determine source (RAG vs WEB) and content type
        source_str = "[WEB]" if result.is_web else "[RAG]"
        
        # Determine content type
        if result.is_product:
          content_type = "PRODUCT"
        elif result.is_review:
          content_type = "REVIEW"
        else:
          content_type = ""  # Web results don't need additional type
        
        # Color based on relevance
        if result.score > 0.8:
          color = typer.colors.GREEN
        elif result.score > 0.5:
          color = typer.colors.YELLOW
        else:
          color = typer.colors.WHITE
        
        # Format title with source and type
        if content_type:
          display_str = f"{idx:2}. {source_str} {content_type}: {result.title[:80]}"
        else:
          display_str = f"{idx:2}. {source_str} {result.title[:80]}"
        
        typer.secho(display_str, fg=color, bold=True)
        
        # Show metadata
        if result.url:
          typer.secho(f"    URL: {result.url}", fg=typer.colors.BRIGHT_BLACK)
        if result.metadata.get("domain"):
          typer.secho(f"    Source: {result.metadata['domain']}", fg=typer.colors.BRIGHT_BLACK)
        
        typer.secho(f"    Score: {result.score:.3f}", fg=typer.colors.BRIGHT_BLACK)
        
        # Show snippet of content
        content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
        typer.secho(f"    {content_preview}", fg=typer.colors.WHITE, dim=True)
        typer.echo()
      
      return
    
    # Original local-only search logic
    q_tokens = _tokenize(search_query)
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
    typer.secho("  • Encoding query...", fg=typer.colors.BRIGHT_BLACK)
    q_vec = st_model.encode([search_query], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
    
    typer.secho("  • Searching vector database...", fg=typer.colors.BRIGHT_BLACK)
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
      ranked = _cross_encoder_scores(CROSS_ENCODER_MODEL, device, search_query, candidates)
      ce_scores = {cid: score for cid, score in ranked}
      fused_sorted = sorted(
        fused_sorted, key=lambda x: (ce_scores.get(x[0], float("-inf")), x[1]), reverse=True
      )

    # Pretty print results
    typer.secho(f"\nSearch Results for: '{search_query}'", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"Found {len(fused_sorted)} results (showing top 20)\n", fg=typer.colors.WHITE)
    
    for idx, (_id, score) in enumerate(fused_sorted[:20], 1):
      if _id.startswith("prod::") and _id in id_to_product:
        p = id_to_product[_id]
        ce_score = ce_scores.get(_id, 0.0)
        title = (p.get('title') or '')[:100]
        rating = p.get('average_rating', 0.0)
        
        # Color based on score
        if ce_score > 0.8:
          color = typer.colors.GREEN
        elif ce_score > 0.5:
          color = typer.colors.YELLOW
        else:
          color = typer.colors.WHITE
          
        typer.secho(
          f"{idx:2d}. [RAG] PRODUCT: {title}",
          fg=color, bold=True
        )
        typer.echo(f"  ID: {_id} | RRF: {score:.4f} | CE: {ce_score:.4f} | Rating: {rating}")
        
      elif _id.startswith("rev::") and _id in id_to_review:
        r = id_to_review[_id]
        ce_score = ce_scores.get(_id, 0.0)
        title = (r.get('title') or '')[:100]
        rating = r.get('rating', 0.0)
        
        # Color based on score
        if ce_score > 0.8:
          color = typer.colors.GREEN
        elif ce_score > 0.5:
          color = typer.colors.YELLOW
        else:
          color = typer.colors.WHITE
          
        typer.secho(
          f"{idx:2d}. [RAG] REVIEW: {title}",
          fg=color
        )
        typer.echo(f"  ID: {_id} | RRF: {score:.4f} | CE: {ce_score:.4f} | Rating: {rating}")
  
  # Check if interactive mode or single query
  if query:
    # Single query mode
    perform_search(query)
  else:
    # Interactive mode
    typer.secho("\nInteractive Search Mode", fg=typer.colors.CYAN, bold=True)
    typer.secho("Type your queries below. Commands:", fg=typer.colors.WHITE)
    typer.secho(" /help - Show search tips", fg=typer.colors.WHITE)
    typer.secho(" /settings - Show current settings", fg=typer.colors.WHITE)
    typer.secho(" /exit - Exit interactive mode", fg=typer.colors.WHITE)
    typer.secho(" Press Ctrl+C to interrupt\n", fg=typer.colors.WHITE)
    
    while True:
      try:
        typer.secho("Search> ", fg=typer.colors.YELLOW, bold=True, nl=False)
        sys.stdout.flush()  # Ensure prompt is displayed
        
        # Clear input buffer
        try:
          import termios
          termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except:
          pass
          
        user_query = input().strip()
      except (EOFError, KeyboardInterrupt):
        typer.secho("\n\nGoodbye!", fg=typer.colors.CYAN)
        break
        
      if not user_query:
        continue
        
      if user_query.lower() in {"/exit", "/quit"}:
        typer.secho("Goodbye!", fg=typer.colors.CYAN)
        break
      elif user_query.lower() == "/help":
        typer.secho("\n Search Tips:", fg=typer.colors.CYAN, bold=True)
        typer.secho(" • Use specific product names for best results", fg=typer.colors.WHITE)
        typer.secho(" • Include key features: 'wireless earbuds noise cancelling'", fg=typer.colors.WHITE)
        typer.secho(" • Price queries work: 'budget laptop under $500'", fg=typer.colors.WHITE)
        typer.secho(" • Brand searches: 'Apple headphones'", fg=typer.colors.WHITE)
        typer.secho(" • Comparative: 'best gaming mouse 2023'\n", fg=typer.colors.WHITE)
      elif user_query.lower() == "/settings":
        typer.secho("\n Current Settings:", fg=typer.colors.CYAN, bold=True)
        typer.secho(f" Top-K per modality: {top_k}", fg=typer.colors.WHITE)
        typer.secho(f" RRF fusion parameter: {rrf_k}", fg=typer.colors.WHITE)
        typer.secho(f" Cross-encoder reranking: {'Enabled' if rerank else 'Disabled'}", fg=typer.colors.WHITE)
        typer.secho(f" Rerank top-K: {rerank_top_k}", fg=typer.colors.WHITE)
        typer.secho(f" Device: {device}\n", fg=typer.colors.WHITE)
      else:
        perform_search(user_query)


def _hybrid_search_inline(
  query: str,
  st_model,
  client,
  bm25_prod,
  bm25_rev,
  id_to_product,
  id_to_review,
  ce_model,
  top_k: int = 20,
  rrf_k: int = 60,
  rerank_top_k: int = 30,
  products_only: bool = False
) -> list:
  """Perform hybrid search with all components already loaded."""
  device = _device_str()
  
  # BM25 search
  q_tokens = _tokenize(query.lower())
  prod_ids = list(id_to_product.keys())
  
  prod_ranked = []
  for i, score in enumerate(bm25_prod.get_scores(q_tokens)):
    if score > 0 and i < len(prod_ids):
      prod_ranked.append((prod_ids[i], score))
  
  # Only include reviews if not products_only
  rev_ranked = []
  rev_vec = []
  if not products_only:
    rev_ids = list(id_to_review.keys())
    for i, score in enumerate(bm25_rev.get_scores(q_tokens)):
      if score > 0 and i < len(rev_ids):
        rev_ranked.append((rev_ids[i], score))
  
  # Vector search
  q_vec = st_model.encode([query], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
  prod_hits = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=top_k*2)
  
  # Only search reviews if not products_only
  if not products_only:
    rev_hits = _vector_search(client, COLLECTION_REVIEWS, q_vec, top_k=top_k*2)
  else:
    rev_hits = []
  
  # Convert to consistent format for fusion
  prod_vec = []
  for doc_id, score, payload in prod_hits:
    original_id = payload.get('original_id', payload.get('id', ''))
    if original_id and original_id in id_to_product:
      prod_vec.append((original_id, score))
  
  if not products_only:
    for doc_id, score, payload in rev_hits:
      original_id = payload.get('original_id', payload.get('id', ''))
      if original_id and original_id in id_to_review:
        rev_vec.append((original_id, score))
  
  # RRF fusion - only include review lists if not products_only
  if products_only:
    fused = _rrf_fuse([prod_ranked, prod_vec], k=rrf_k)
  else:
    fused = _rrf_fuse([prod_ranked, rev_ranked, prod_vec, rev_vec], k=rrf_k)
  fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:50]
  
  # Cross-encoder reranking
  ce_scores = {}
  if fused_sorted:
    k_ce = min(rerank_top_k, len(fused_sorted))
    candidates = []
    for _id, _ in fused_sorted[:k_ce]:
      if _id.startswith("prod::") and _id in id_to_product:
        p = id_to_product[_id]
        candidates.append((_id, f"{p.get('title','')}\n{p.get('description','')}"))
      elif _id.startswith("rev::") and _id in id_to_review:
        r = id_to_review[_id]
        candidates.append((_id, f"{r.get('title','')}\n{r.get('text','')}"))
    
    if candidates:
      # Use the passed ce_model directly instead of reloading
      pairs = [(query, text) for _, text in candidates]
      scores = ce_model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
      ranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
      ranked.sort(key=lambda x: x[1], reverse=True)
      ce_scores = {cid: score for cid, score in ranked}
  
  # Combine and sort results
  results = []
  for doc_id, rrf_score in fused_sorted:
    ce_score = ce_scores.get(doc_id, 0.0)
    if doc_id.startswith("prod::") and doc_id in id_to_product:
      payload = id_to_product[doc_id]
    elif doc_id.startswith("rev::") and doc_id in id_to_review:
      payload = id_to_review[doc_id]
    else:
      continue
    results.append((doc_id, rrf_score, ce_score, payload))
  
  # Sort by cross-encoder score if available, otherwise RRF
  results.sort(key=lambda x: (x[2], x[1]), reverse=True)
  return results[:top_k]


@app.command()
def interactive() -> None:
  """Unified interactive mode - just type naturally to search or chat."""
  
  typer.secho("\nShopping Assistant", fg=typer.colors.CYAN, bold=True)
  typer.secho("Type naturally - I'll search for products or answer questions as needed.", fg=typer.colors.WHITE)
  typer.secho("\nCommands:", fg=typer.colors.YELLOW)
  typer.secho(" /help - Show tips and examples", fg=typer.colors.WHITE)
  typer.secho(" /exit or Ctrl+C - Exit", fg=typer.colors.WHITE)
  
  # Small delay and flush to ensure initial messages are fully displayed
  import time
  time.sleep(0.1)
  sys.stdout.flush()
  sys.stderr.flush()
  
  # Lazy-load components on first use for faster startup
  import dspy
  from sentence_transformers import SentenceTransformer
  
  # Component cache
  components = {
    'initialized': False,
    'st_model': None,
    'client': None,
    'bm25_prod': None,
    'bm25_rev': None,
    'id_to_product': None,
    'id_to_review': None,
    'ce_model': None,
    'rag': None,
    'device': None
  }
  
  def initialize_components():
    """Initialize all components on first query."""
    if components['initialized']:
      return
      
    typer.secho("Loading models (first query only)...", fg=typer.colors.YELLOW, italic=True)
    
    # Setup for chat
    llm_config = get_llm_config()
    try:
      lm = llm_config.get_dspy_lm(task="chat")
      dspy.configure(lm=lm)
    except ValueError as e:
      typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
      raise typer.Exit(1)
    
    # Load models and data
    components['device'] = _device_str()
    components['st_model'] = _load_st_model(EMBED_MODEL, device=components['device'])
    components['client'] = _qdrant_client()
    bm25_data = _bm25_from_files(DATA_PRODUCTS, DATA_REVIEWS)
    components['bm25_prod'] = bm25_data[0]
    components['bm25_rev'] = bm25_data[1]
    components['id_to_product'] = bm25_data[2]
    components['id_to_review'] = bm25_data[3]
    components['ce_model'] = _get_cross_encoder(CROSS_ENCODER_MODEL, components['device'])
    components['rag'] = dspy.Predict("question, context -> answer")
    components['initialized'] = True
    
    typer.echo("\033[F\033[K", nl=False) # Move up and clear line
  
  def is_search_query(text: str) -> bool:
    """Heuristic to determine if input is a search query vs chat question."""
    # Keywords that suggest search intent
    search_keywords = ['find', 'show', 'list', 'search', 'looking for', 'need', 'want to buy']
    # Keywords that suggest chat/question intent
    chat_keywords = ['what', 'how', 'why', 'when', 'which', 'tell me', 'explain', 'compare', 
            'difference', 'recommend', 'should i', 'is it', 'are there', '?']
    
    text_lower = text.lower()
    
    # Check for question mark
    if '?' in text:
      return False
      
    # Count keyword matches
    search_score = sum(1 for kw in search_keywords if kw in text_lower)
    chat_score = sum(1 for kw in chat_keywords if kw in text_lower)
    
    # Simple product name queries are usually searches
    if len(text.split()) <= 3 and chat_score == 0:
      return True
      
    return search_score > chat_score
  
  def handle_query(user_input: str):
    """Process user input and route to search or chat."""
    # Initialize components on first use
    initialize_components()
    
    # Clean up common search prefixes that users might type
    query = user_input.strip()
    search_prefixes = ['search for ', 'search ', 'find ', 'show me ', 'look for ', 'find me ']
    query_lower = query.lower()
    for prefix in search_prefixes:
      if query_lower.startswith(prefix):
        query = query[len(prefix):].strip()
        break
    
    # Remove "products only" or similar suffixes
    products_only = False
    if ', products only' in query.lower() or ' products only' in query.lower():
      products_only = True
      query = query.replace(', products only', '').replace(' products only', '')
    
    # Check for typos and suggest corrections using fast LLM approach
    try:
      from app.fast_query_correction import suggest_correction
      correction = suggest_correction(query)
      if correction and correction != query:
        typer.secho(f" (Auto-corrected to: '{correction}')", fg=typer.colors.BRIGHT_BLACK, italic=True)
        query = correction
    except Exception:
      # If correction fails, continue with original query
      pass
    
    # Auto-detect mode based on query content
    if is_search_query(query):
      mode = 'search'
    else:
      mode = 'chat'
    
    if mode == 'search':
      typer.secho(f"\nSearching for: '{query}'", fg=typer.colors.BLUE)
      # Perform hybrid search
      results = _hybrid_search_inline(
        query, components['st_model'], components['client'], 
        components['bm25_prod'], components['bm25_rev'], 
        components['id_to_product'], components['id_to_review'], 
        components['ce_model'],
        top_k=20, rrf_k=60, rerank_top_k=30,
        products_only=products_only
      )
      
      if not results:
        typer.secho("No results found.", fg=typer.colors.YELLOW)
      else:
        typer.secho(f"Found {len(results)} results (showing top 10):\n", fg=typer.colors.GREEN)
        for i, (doc_id, rrf_score, ce_score, payload) in enumerate(results[:10], 1):
          _type = "PRODUCT" if doc_id.startswith("prod::") else "REVIEW"
          title = payload.get("title", "No title")[:80]
          rating = payload.get("rating", payload.get("average_rating", "N/A"))
          
          typer.secho(f" {i:2}. [{_type:7}] {title}", fg=typer.colors.WHITE, bold=True)
          typer.secho(f"   Rating: {rating} | Relevance: {ce_score:.2f}", fg=typer.colors.BRIGHT_BLACK)
    
    else: # chat mode
      typer.secho(f"\nThinking about: '{query}'", fg=typer.colors.BLUE)
      # Get contexts for RAG
      q_vec = components['st_model'].encode([query], batch_size=1, normalize_embeddings=True, 
                         device=components['device'], convert_to_numpy=True)[0].tolist()
      prod_hits = _vector_search(components['client'], COLLECTION_PRODUCTS, q_vec, top_k=8)
      rev_hits = _vector_search(components['client'], COLLECTION_REVIEWS, q_vec, top_k=8)
      payloads = [p for _, _, p in (prod_hits + rev_hits)]
      
      contexts = []
      for p in payloads[:8]:
        if "description" in p:
          contexts.append(f"Title: {p.get('title','')}\nDescription: {p.get('description','')}")
        elif "text" in p:
          contexts.append(f"Title: {p.get('title','')}\nReview: {p.get('text','')}")
      
      ctx = "\n\n".join(contexts)
      pred = components['rag'](question=query, context=ctx)
      answer = getattr(pred, "answer", "I couldn't find relevant information to answer that.")
      
      typer.secho("\nAssistant:", fg=typer.colors.GREEN, bold=True)
      typer.echo(f"  {answer}")
  
  # Main loop
  while True:
    try:
      typer.secho("\nYou: ", fg=typer.colors.CYAN, bold=True, nl=False)
      sys.stdout.flush()  # Ensure prompt is displayed
      
      # Platform-specific input buffer clearing
      try:
        import termios
        # For Unix-like systems (Linux, macOS)
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
      except (ImportError, AttributeError):
        # For Windows or if termios not available
        try:
          import msvcrt
          while msvcrt.kbhit():
            msvcrt.getch()
        except ImportError:
          pass
      
      # NOW accept the actual user input
      user_input = input().strip()
      
      if not user_input:
        continue
        
      # Handle commands
      if user_input.lower() in ['/exit', 'exit', 'quit', 'bye']:
        typer.secho("\nGoodbye!", fg=typer.colors.CYAN)
        break
      elif user_input.lower() == '/help':
        typer.secho("\nTips:", fg=typer.colors.YELLOW, bold=True)
        typer.secho(" - Type product names to search (e.g., 'wireless mouse', 'laptop')", fg=typer.colors.WHITE)
        typer.secho(" - Ask questions for recommendations (e.g., 'what's the best tablet?')", fg=typer.colors.WHITE)
        typer.secho(" - The system automatically understands what you need", fg=typer.colors.WHITE)
        typer.secho("\nExamples:", fg=typer.colors.YELLOW, bold=True)
        typer.secho(" Search: gaming headset, bluetooth speaker, USB hub", fg=typer.colors.BRIGHT_BLACK)
        typer.secho(" Chat: compare iPhone vs Android, best laptop for students", fg=typer.colors.BRIGHT_BLACK)
        continue
      
      # Process the query
      handle_query(user_input)
      
    except (EOFError, KeyboardInterrupt):
      typer.secho("\n\nGoodbye!", fg=typer.colors.CYAN)
      break
    except Exception as e:
      typer.secho(f"\nError: {e}", fg=typer.colors.RED)
      typer.secho("Please try again or type /help for assistance.", fg=typer.colors.YELLOW)


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
    typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
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
    typer.secho(f"\nQuestion: {question}", fg=typer.colors.CYAN)
    typer.secho("\nAnswer:", fg=typer.colors.GREEN)
    typer.echo(answer_one(question))
    raise typer.Exit(0)

  typer.secho("\nInteractive Chat Mode", fg=typer.colors.CYAN, bold=True)
  typer.secho("I can help you find products, compare items, and answer questions.", fg=typer.colors.WHITE)
  typer.secho("\nCommands:", fg=typer.colors.WHITE)
  typer.secho(" /help - Show example questions", fg=typer.colors.WHITE)
  typer.secho(" /context - Show how many contexts are being retrieved", fg=typer.colors.WHITE)
  typer.secho(" /clear - Clear the screen", fg=typer.colors.WHITE)
  typer.secho(" /exit - Exit chat mode", fg=typer.colors.WHITE)
  typer.secho(" Press Ctrl+C to interrupt\n", fg=typer.colors.WHITE)
  
  # Keep track of conversation history for context
  history = []
  
  while True:
    try:
      typer.secho("You: ", fg=typer.colors.YELLOW, bold=True, nl=False)
      sys.stdout.flush()  # Ensure prompt is displayed
      
      # Clear input buffer
      try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
      except:
        pass
        
      q = input().strip()
    except (EOFError, KeyboardInterrupt):
      typer.secho("\n\n Goodbye!", fg=typer.colors.CYAN)
      break
      
    if not q:
      continue
      
    if q.lower() in {"/exit", "/quit"}:
      typer.secho(" Goodbye!", fg=typer.colors.CYAN)
      break
    elif q.lower() == "/help":
      typer.secho("\n Example Questions:", fg=typer.colors.CYAN, bold=True)
      typer.secho(" • What are the best wireless earbuds under $200?", fg=typer.colors.WHITE)
      typer.secho(" • Compare Sony WH-1000XM4 and Bose QuietComfort", fg=typer.colors.WHITE)
      typer.secho(" • Which laptop is good for programming?", fg=typer.colors.WHITE)
      typer.secho(" • What do customers say about the Apple AirPods Pro?", fg=typer.colors.WHITE)
      typer.secho(" • Find me a gaming mouse with RGB lighting", fg=typer.colors.WHITE)
      typer.secho(" • What are the pros and cons of mechanical keyboards?\n", fg=typer.colors.WHITE)
      continue
    elif q.lower() == "/context":
      typer.secho(f"\n Context Settings:", fg=typer.colors.CYAN, bold=True)
      typer.secho(f" Retrieving top {top_k} contexts per query", fg=typer.colors.WHITE)
      typer.secho(f" Sources: Products + Customer Reviews\n", fg=typer.colors.WHITE)
      continue
    elif q.lower() == "/clear":
      import os
      os.system('clear' if os.name == 'posix' else 'cls')
      typer.secho("Interactive Chat Mode", fg=typer.colors.CYAN, bold=True)
      continue
    
    # Show thinking indicator
    typer.secho("\nThinking...", fg=typer.colors.BLUE, italic=True)
    
    # Get answer
    a = answer_one(q)
    
    # Clear thinking indicator and show answer
    typer.echo("\033[F\033[K", nl=False) # Move up and clear line
    typer.secho("Assistant: ", fg=typer.colors.GREEN, bold=True)
    
    # Format answer with better line wrapping
    import textwrap
    wrapped = textwrap.fill(a, width=80, initial_indent="  ", subsequent_indent="  ")
    typer.echo(f"{wrapped}\n")
    
    # Store in history
    history.append({"question": q, "answer": a})
    
    # Show follow-up suggestion if applicable
    if len(history) == 1:
      typer.secho(" Tip: Ask follow-up questions for more details!\n", fg=typer.colors.CYAN, dim=True)


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
  enhanced: bool = typer.Option(False, help="Use enhanced retrieval with query expansion"),
) -> None:
  """Evaluate retrieval variants; logs metrics and report to eval/results and MLflow."""

  import mlflow
  
  # Capture the full execution command
  execution_command = " ".join(sys.argv)
  
  # Note: RAGAS metrics are not used for search evaluation
  # They require generated answers which we don't have in retrieval-only evaluation
  # For proper retrieval evaluation, use custom metrics or human evaluation

  results_dir, _ = _ensure_dirs()
  device = _device_str()
  st_model = _load_st_model(EMBED_MODEL, device=device)
  client = _qdrant_client()
  bm25_prod, bm25_rev, id_to_product, id_to_review = _bm25_from_files(products_path, reviews_path)
  
  # Extract ID lists for BM25 scoring (needed for enhanced mode)
  prod_ids = list(id_to_product.keys())
  rev_ids = list(id_to_review.keys())

  queries = _load_jsonl(dataset, max_samples=max_samples, seed=seed)
  variant_list = [v.strip() for v in variants.split(",") if v.strip()]

  typer.echo("Starting eval-search with configuration:")
  typer.echo(f"- Dataset: {dataset}")
  typer.echo(f"- Samples (requested): {max_samples}; (loaded): {len(queries)}")
  typer.echo(f"- Variants: {', '.join(variant_list)}")
  typer.echo(f"- top_k={top_k} rrf_k={rrf_k} rerank_top_k={rerank_top_k}")
  typer.echo(f"- Device: {device}; Embed model: {EMBED_MODEL}")
  
  # Pre-load cross-encoder model if needed to avoid repeated downloads
  if "rrf_ce" in variant_list:
    typer.echo(f"Pre-loading cross-encoder model: {CROSS_ENCODER_MODEL}")
    _get_cross_encoder(CROSS_ENCODER_MODEL, device)
    typer.echo("✓ Cross-encoder model loaded and cached")

  # Import enhanced retrieval if needed
  if enhanced:
    from app.search_improvements import ImprovedRetriever, QueryPreprocessor
    improved_retriever = ImprovedRetriever(top_k=top_k, rrf_k=rrf_k)
    query_preprocessor = QueryPreprocessor()
  
  def run_variant(q: str, variant: str) -> list[dict]:
    """Return list of payload dicts for top_k contexts under a variant."""
    # Use enhanced retrieval if enabled
    if enhanced and variant in ["rrf", "rrf_ce"]:
      # Process query for better results
      enhanced_query = query_preprocessor.process(q)
      
      # Define BM25 and vector functions for improved retriever
      def bm25_func(keywords):
        # Use keywords for BM25
        query_text = " ".join(keywords) if keywords else q.lower()
        q_tokens = _tokenize(query_text)
        
        prod_scores = []
        for i, score in enumerate(bm25_prod.get_scores(q_tokens)):
          if score > 0:
            prod_scores.append((prod_ids[i], score))
        
        rev_scores = []
        for i, score in enumerate(bm25_rev.get_scores(q_tokens)):
          if score > 0:
            rev_scores.append((rev_ids[i], score))
        
        # Combine and sort
        all_scores = prod_scores + rev_scores
        return sorted(all_scores, key=lambda x: x[1], reverse=True)[:top_k*2]
      
      def vector_func(query_text):
        # Use expanded query for vectors
        query_vec = st_model.encode([query_text], convert_to_tensor=True, device=device)[0]
        
        # Search products
        prod_results = client.search(
          collection_name="products_minilm",
          query_vector=query_vec.cpu().numpy().tolist(),
          limit=top_k,
          with_payload=True
        )
        
        # Search reviews
        rev_results = client.search(
          collection_name="reviews_minilm",
          query_vector=query_vec.cpu().numpy().tolist(),
          limit=top_k,
          with_payload=True
        )
        
        # Combine results
        all_scores = []
        for hit in prod_results:
          original_id = hit.payload.get('original_id', hit.payload.get('id', ''))
          if original_id:
            all_scores.append((original_id, hit.score))
        
        for hit in rev_results:
          original_id = hit.payload.get('original_id', hit.payload.get('id', ''))
          if original_id:
            all_scores.append((original_id, hit.score))
        
        return all_scores
      
      # Get improved results
      improved_results = improved_retriever.retrieve_with_fallback(
        q,
        bm25_func,
        vector_func,
        {**id_to_product, **id_to_review}
      )
      
      # Apply cross-encoder if needed
      if variant == "rrf_ce" and improved_results:
        k_ce = min(rerank_top_k, len(improved_results))
        candidates: list[tuple[str, str]] = []
        for doc_id, _ in improved_results[:k_ce]:
          if doc_id.startswith("prod::") and doc_id in id_to_product:
            p = id_to_product[doc_id]
            candidates.append((doc_id, f"{p.get('title','')}\n{p.get('description','')}"))
          elif doc_id.startswith("rev::") and doc_id in id_to_review:
            r = id_to_review[doc_id]
            candidates.append((doc_id, f"{r.get('title','')}\n{r.get('text','')}"))
        
        if candidates:
          ranked = _cross_encoder_scores(CROSS_ENCODER_MODEL, device, q, candidates)
          ce_scores = {cid: score for cid, score in ranked}
          improved_results = sorted(improved_results, key=lambda x: (ce_scores.get(x[0], float("-inf")), x[1]), reverse=True)
      
      # Convert to payloads
      payloads: list[dict] = []
      for doc_id, _ in improved_results[:top_k]:
        if doc_id.startswith("prod::") and doc_id in id_to_product:
          payloads.append(id_to_product[doc_id])
        elif doc_id.startswith("rev::") and doc_id in id_to_review:
          payloads.append(id_to_review[doc_id])
      
      return payloads
    
    # Original variant logic for non-enhanced or non-RRF variants
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

  # Adjust parameters if using enhanced mode
  if enhanced:
    from app.search_improvements import IMPROVED_SEARCH_CONFIG
    # Override with improved parameters
    if top_k == 20: # Use default improved value only if not explicitly set
      top_k = IMPROVED_SEARCH_CONFIG["top_k"]
    if rrf_k == 60: # Use default improved value only if not explicitly set
      rrf_k = IMPROVED_SEARCH_CONFIG["rrf_k"]
    if rerank_top_k == 30: # Use default improved value only if not explicitly set
      rerank_top_k = IMPROVED_SEARCH_CONFIG["rerank_top_k"]
    
    typer.echo(f" Enhanced mode enabled with optimized parameters:")
    typer.echo(f"  top_k={top_k}, rrf_k={rrf_k}, rerank_top_k={rerank_top_k}")
    typer.echo(f"  Query expansion: enabled")
    typer.echo(f"  Fallback strategies: enabled")
  
  with mlflow.start_run(run_name=f"eval-search-{timestamp}"):
    mlflow.log_param("variants", ",".join(variant_list))
    mlflow.log_param("top_k", top_k)
    mlflow.log_param("rrf_k", rrf_k)
    mlflow.log_param("rerank_top_k", rerank_top_k)
    mlflow.log_param("max_samples", max_samples)
    mlflow.log_param("enhanced", enhanced)

    variant_to_rows: dict[str, list[dict]] = {v: [] for v in variant_list}
    # NEW: Store detailed results for each query
    detailed_results: dict[str, list[dict]] = {v: [] for v in variant_list}
    
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
        # Generate a simple answer from the first few contexts for RAGAS metrics
        # This allows ContextRelevance and ContextUtilization to work
        if contexts:
          # Take first 3 contexts and create a brief summary as answer
          answer_contexts = contexts[:3]
          answer = "Based on the search results: " + " ".join([c[:100] for c in answer_contexts])
        else:
          answer = "No relevant information found."
        variant_to_rows[variant].append({"question": q, "contexts": contexts, "answer": answer})
        
        # NEW: Store detailed info for this query
        detailed_info = {
          "query": q,
          "retrieved_items": [
            {
              "rank": i + 1,
              "type": "product" if p.get("title") else "review",
              "title": p.get("title", "N/A"),
              "category": p.get("category_name", p.get("main_category", "N/A")),
              "rating": p.get("rating", p.get("average_rating", "N/A")),
              "snippet": _to_context_text(p)[:200] + "..." if len(_to_context_text(p)) > 200 else _to_context_text(p)
            }
            for i, p in enumerate(payloads[:top_k])
          ],
          "num_retrieved": len(payloads)
        }
        detailed_results[variant].append(detailed_info)
        
        variant_counts[variant] += 1
      if idx % 10 == 0:
        elapsed = time.time() - start_collect
        typer.echo(
          f" processed {idx}/{len(queries)} queries in {elapsed:.1f}s; per-variant: "
          + ", ".join([f"{v}={variant_counts[v]}" for v in variant_list])
        )

    # Skip RAGAS metrics for search evaluation - they require actual answers
    # Search evaluation only measures retrieval quality, not answer generation
    aggregates: dict[str, dict[str, float]] = {}
    # Store per-query metrics (empty for now, could add custom retrieval metrics later)
    per_query_metrics: dict[str, list[dict]] = {v: [] for v in variant_list}
    
    # For search evaluation, we'll compute simple retrieval metrics instead
    for variant, rows in variant_to_rows.items():
      if not rows:
        aggregates[variant] = {}
        continue
      
      # Compute simple retrieval quality metrics
      variant_scores = {
        "num_queries": len(rows),
        "avg_contexts_retrieved": sum(len(r["contexts"]) for r in rows) / len(rows) if rows else 0,
        "queries_with_results": sum(1 for r in rows if r["contexts"]) / len(rows) if rows else 0,
      }
      
      aggregates[variant] = variant_scores
      for k, v in variant_scores.items():
        mlflow.log_metric(f"{variant}_{k}", v)
      
      # Add empty metrics for each query to maintain structure
      for _ in rows:
        per_query_metrics[variant].append({})
      
      metric_str = ", ".join([f"{m}={aggregates[variant].get(m, 0):.2f}" for m in sorted(aggregates[variant].keys())])
      typer.secho(
        f" ✓ Evaluated {variant}: {metric_str}",
        fg=typer.colors.GREEN
      )

    # Capture all call parameters
    call_params = {
      "execution_command": execution_command,
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
    
    # NEW: Combine detailed results with metrics
    detailed_results_with_metrics = {}
    for variant in variant_list:
      variant_details = []
      for i, detail in enumerate(detailed_results[variant]):
        combined = detail.copy()
        if i < len(per_query_metrics[variant]):
          combined["metrics"] = per_query_metrics[variant][i]
        else:
          combined["metrics"] = {}
        variant_details.append(combined)
      detailed_results_with_metrics[variant] = variant_details
    
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
      "detailed_results": detailed_results_with_metrics # NEW: Add detailed results
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
      f"- **Execution Command**: `{execution_command}`\n"
      f"- **Command**: `eval-search`\n"
      f"- **Timestamp**: {timestamp}\n"
      f"- **Dataset**: `{dataset}`\n"
      f"- **Products**: `{products_path}`\n"
      f"- **Reviews**: `{reviews_path}`\n"
      f"- **Samples**: requested={max_samples}, loaded={len(queries)}\n"
      f"- **Variants**: {', '.join(variant_list)}\n"
      f"- **Search params**: top_k={top_k}, rrf_k={rrf_k}, rerank_top_k={rerank_top_k}\n"
      f"- **Enhanced mode**: {' Enabled (query expansion + fallback)' if enhanced else ' Disabled'}\n"
      f"- **Seed**: {seed}\n"
      f"- **Device**: {device}\n"
      f"- **Embed model**: `{EMBED_MODEL}`\n"
      f"- **Cross-encoder**: `{CROSS_ENCODER_MODEL}`\n"
    )
    # Generate interpretation
    from app.eval_interpreter import generate_search_interpretation
    
    interpretation_config = {
      "dataset": str(dataset),
      "max_samples": max_samples,
      "top_k": top_k,
      "rrf_k": rrf_k,
      "rerank_top_k": rerank_top_k
    }
    
    interpretation = generate_search_interpretation(aggregates, interpretation_config)
    
    # NEW: Add sample queries section to markdown
    sample_queries_md = "\n## Sample Query Results\n\n"
    
    # Show first 3 queries for each variant as examples
    num_examples = min(3, len(queries))
    for variant in variant_list:
      sample_queries_md += f"\n### {variant.upper()} Variant\n\n"
      
      for i in range(num_examples):
        if i >= len(detailed_results_with_metrics[variant]):
          break
          
        result = detailed_results_with_metrics[variant][i]
        query_metrics = result.get("metrics", {})
        
        sample_queries_md += f"**Query {i+1}:** {result['query']}\n\n"
        
        # Show metrics for this query
        if query_metrics:
          metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in query_metrics.items() if v is not None])
          sample_queries_md += f"*Metrics:* {metrics_str}\n\n"
        
        # Show top 3 retrieved items
        sample_queries_md += "*Top Retrieved Items:*\n"
        for item in result['retrieved_items'][:3]:
          sample_queries_md += f"- **#{item['rank']}** [{item['type'].upper()}] {item['title']} (Rating: {item['rating']})\n"
          sample_queries_md += f" - Category: {item['category']}\n"
          sample_queries_md += f" - Snippet: {item['snippet']}\n"
        
        sample_queries_md += "\n"
    
    md = (
      "# Search Evaluation Report\n\n" + 
      md_config + 
      "\n## Metrics\n" + 
      _write_markdown_table(headers, rows_md) + 
      sample_queries_md + # NEW: Add sample queries
      "\n\n---\n\n" +
      interpretation
    )
    out_md.write_text(md)
    mlflow.log_artifact(str(out_md))

  typer.secho(f"\n Evaluation Complete!", fg=typer.colors.GREEN, bold=True)
  typer.secho(f" 💾 JSON: {out_json}", fg=typer.colors.WHITE)
  typer.secho(f" 📄 Report: {out_md}", fg=typer.colors.WHITE)


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
  from app.ragas_config import configure_ragas_metrics
  import dspy

  # Capture the full execution command
  execution_command = " ".join(sys.argv)

  # Configure LLMs for both DSPy chat and RAGAS evaluation with GPT-5 fix
  from app.ragas_gpt5_fix import setup_gpt5_compatibility
  setup_gpt5_compatibility()
  
  llm_config = get_llm_config()
  try:
    llm_config.configure_ragas() # For evaluation
    lm = llm_config.get_dspy_lm(task="chat") # For generating answers
    dspy.configure(lm=lm)
  except ValueError as e:
    typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
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

  typer.secho("\n Starting Chat Evaluation", fg=typer.colors.CYAN, bold=True)
  typer.secho(f" Dataset: {dataset}", fg=typer.colors.WHITE)
  typer.secho(f" Top-K contexts: {top_k}", fg=typer.colors.WHITE)
  typer.secho(f" Max samples: {max_samples}", fg=typer.colors.WHITE)
  
  rows_in = _load_jsonl(dataset, max_samples=max_samples, seed=seed)
  typer.secho(f"\n⏳ Generating answers for {len(rows_in)} questions...", fg=typer.colors.BLUE)
  
  rows_eval: list[dict] = []
  for idx, row in enumerate(rows_in, 1):
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
    
    if idx % 5 == 0 or idx == len(rows_in):
      typer.secho(f" Processed {idx}/{len(rows_in)} questions", fg=typer.colors.BLUE)

  timestamp = _get_timestamp()
  out_json = results_dir / f"chat_{timestamp}.json"
  out_md = results_dir / f"chat_{timestamp}.md"

  typer.secho(f"\n⏳ Running RAGAS evaluation...", fg=typer.colors.BLUE)
  
  with mlflow.start_run(run_name=f"eval-chat-{timestamp}"):
    mlflow.log_param("top_k", top_k)
    mlflow.log_param("max_samples", max_samples)
    ds = Dataset.from_list(rows_eval)
    # Create metrics with GPT-5 compatible configuration
    metrics = configure_ragas_metrics([faithfulness, answer_relevancy, context_precision, context_recall])
    res = ragas_evaluate(ds, metrics=metrics)
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
      typer.secho(f"⚠️ Warning: Could not extract scores from evaluation result: {e}", fg=typer.colors.YELLOW, err=True)
      scores = {"error": 0.0}
    
    for k, v in scores.items():
      mlflow.log_metric(k, v)
    
    # Display scores
    if scores and "error" not in scores:
      typer.secho("\n✓ Evaluation Metrics:", fg=typer.colors.GREEN, bold=True)
      for metric, value in scores.items():
        color = typer.colors.GREEN if value > 0.8 else typer.colors.YELLOW if value > 0.6 else typer.colors.RED
        typer.secho(f" {metric}: {value:.4f}", fg=color)

    # Capture all call parameters
    call_params = {
      "execution_command": execution_command,
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
      f"- **Execution Command**: `{execution_command}`\n"
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
    
    # Generate interpretation
    from app.eval_interpreter import generate_chat_interpretation
    
    interpretation_config = {
      "dataset": str(dataset),
      "max_samples": max_samples,
      "top_k": top_k,
      "model": llm_config.chat_model
    }
    
    interpretation = generate_chat_interpretation(scores, interpretation_config)
    
    md = (
      "# Chat Evaluation Report\n\n" + 
      md_params + 
      "## Metrics\n" + 
      _write_markdown_table(["metric", "score"], [[k, f"{v:.4f}"] for k, v in scores.items()]) + 
      "\n\n---\n\n" +
      interpretation
    )
    out_md.write_text(md)
    mlflow.log_artifact(str(out_md))

  typer.secho(f"\n Evaluation Complete!", fg=typer.colors.GREEN, bold=True)
  typer.secho(f" 💾 JSON: {out_json}", fg=typer.colors.WHITE)
  typer.secho(f" 📄 Report: {out_md}", fg=typer.colors.WHITE)


@app.command("generate-testset")
def generate_testset(
  num_samples: int = typer.Option(500, help="Number of test samples to generate"),
  output_name: str = typer.Option("realistic", help="Output filename prefix"),
  include_reference: bool = typer.Option(True, help="Include reference answers"),
  distribution_preset: str = typer.Option("balanced", help="Query distribution: balanced, simple, complex, mixed"),
  seed: int = typer.Option(42, help="Random seed for reproducibility"),
  products_path: Path = typer.Option(DATA_PRODUCTS, exists=True),
  reviews_path: Path = typer.Option(DATA_REVIEWS, exists=True),
) -> None:
  """Generate realistic test dataset based on actual product catalog.
  
  Creates test data with various query complexities:
  - Single-hop factual queries (simple lookups)
  - Multi-hop reasoning (cross-document analysis)
  - Comparative queries (product comparisons)
  - Recommendations (personalized suggestions)
  - Technical queries (specifications)
  - Problem-solving (troubleshooting)
  
  Queries are generated using real products, brands, and categories from the catalog.
  """
  from app.testset_generator import RealisticQueryGenerator
  
  typer.secho("\n🧪 Realistic Test Data Generation", fg=typer.colors.CYAN, bold=True)
  typer.secho(f" Generator: Catalog-based (using actual products)", fg=typer.colors.WHITE)
  typer.secho(f" Target samples: {num_samples}", fg=typer.colors.WHITE)
  typer.secho(f" Distribution: {distribution_preset}", fg=typer.colors.WHITE)
  typer.secho(f" Include references: {include_reference}", fg=typer.colors.WHITE)
  typer.secho(f" Random seed: {seed}\n", fg=typer.colors.WHITE)
  
  # Load documents
  typer.secho(" Loading source documents...", fg=typer.colors.BLUE)
  products = _read_jsonl(products_path)
  reviews = _read_jsonl(reviews_path)
  typer.secho(f" ✓ Loaded {len(products)} products", fg=typer.colors.GREEN)
  typer.secho(f" ✓ Loaded {len(reviews)} reviews\n", fg=typer.colors.GREEN)
  
  # Initialize generator
  generator = RealisticQueryGenerator(products, reviews, seed)
  
  # Define distribution presets
  distributions = {
    "balanced": {
      "single_hop_factual": 0.25,
      "multi_hop_reasoning": 0.20,
      "abstract_interpretive": 0.10,
      "comparative": 0.15,
      "recommendation": 0.15,
      "technical": 0.10,
      "problem_solving": 0.05
    },
    "simple": {
      "single_hop_factual": 0.50,
      "technical": 0.25,
      "comparative": 0.15,
      "recommendation": 0.10,
      "multi_hop_reasoning": 0.0,
      "abstract_interpretive": 0.0,
      "problem_solving": 0.0
    },
    "complex": {
      "single_hop_factual": 0.10,
      "multi_hop_reasoning": 0.35,
      "abstract_interpretive": 0.20,
      "comparative": 0.10,
      "recommendation": 0.10,
      "technical": 0.05,
      "problem_solving": 0.10
    },
    "mixed": {
      "single_hop_factual": 0.30,
      "multi_hop_reasoning": 0.25,
      "comparative": 0.20,
      "recommendation": 0.15,
      "technical": 0.05,
      "abstract_interpretive": 0.03,
      "problem_solving": 0.02
    }
  }
  
  distribution = distributions.get(distribution_preset, distributions["balanced"])
  
  # Show distribution
  typer.secho(" Query Type Distribution:", fg=typer.colors.YELLOW)
  for query_type, weight in distribution.items():
    count = int(num_samples * weight)
    bar_length = int(weight * 40)
    bar = "█" * bar_length + "░" * (40 - bar_length)
    typer.secho(
      f" {query_type:25} {bar} {count:4} ({weight*100:5.1f}%)",
      fg=typer.colors.WHITE
    )
  
  # Generate dataset
  typer.secho(f"\n Generating {num_samples} test samples...", fg=typer.colors.BLUE)
  
  # Generator uses distribution string directly
  dataset = generator.generate_dataset(num_samples, distribution_preset)
  
  # Analyze generated dataset
  complexity_stats = {"simple": 0, "moderate": 0, "complex": 0}
  query_type_stats = {}
  
  for sample in dataset:
    # Handle both v1 and v2 format
    metadata = sample.get("metadata", {})
    complexity = metadata.get("complexity", "moderate")
    complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1
    
    query_type = metadata.get("query_type", "general")
    query_type_stats[query_type] = query_type_stats.get(query_type, 0) + 1
  
  # Save dataset
  results_dir, datasets_dir = _ensure_dirs()
  timestamp = _get_timestamp()
  output_path = datasets_dir / f"{output_name}_{num_samples}_{timestamp}.jsonl"
  
  with open(output_path, 'w') as f:
    # Write metadata as first line
    metadata = {
      "_metadata": {
        "total_samples": len(dataset),
        "generation_seed": seed,
        "distribution_preset": distribution_preset,
        "distribution": distribution,
        "complexity_distribution": complexity_stats,
        "query_type_distribution": query_type_stats,
        "timestamp": timestamp,
        "include_reference": include_reference
      }
    }
    f.write(json.dumps(metadata) + "\n")
    
    # Write samples
    for i, sample in enumerate(dataset):
      output_sample = {
        "question": sample.get("query", ""),
        "query": sample.get("query", ""),
        "query_id": sample.get("query_id", f"q_{i:04d}"),
        "metadata": sample.get("metadata", {})
      }
      
      if "reference_answer" in sample:
        output_sample["reference_answer"] = sample["reference_answer"]
        output_sample["ground_truth"] = sample["reference_answer"]
      elif "reference" in sample:
        output_sample["reference_answer"] = sample["reference"]
        output_sample["ground_truth"] = sample["reference"]
      
      if "expected_context_type" in sample:
        output_sample["expected_context_type"] = sample["expected_context_type"]
      
      f.write(json.dumps(output_sample) + "\n")
  
  # Display results
  typer.secho(f"\n Successfully generated {len(dataset)} test samples!", fg=typer.colors.GREEN, bold=True)
  typer.secho(f" 💾 Saved to: {output_path}", fg=typer.colors.WHITE)
  
  typer.secho(f"\n Dataset Statistics:", fg=typer.colors.CYAN, bold=True)
  typer.secho(" Complexity Distribution:", fg=typer.colors.YELLOW)
  for level in ["simple", "moderate", "complex"]:
    count = complexity_stats[level]
    pct = count / len(dataset) * 100
    typer.secho(f"  {level:10} {count:4} ({pct:5.1f}%)", fg=typer.colors.WHITE)
  
  typer.secho("\n Query Types Generated:", fg=typer.colors.YELLOW)
  for qtype, count in sorted(query_type_stats.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(dataset) * 100
    typer.secho(f"  {qtype:25} {count:4} ({pct:5.1f}%)", fg=typer.colors.WHITE)
  
  # Save metadata separately
  metadata_path = datasets_dir / f"{output_name}_{num_samples}_{timestamp}_metadata.json"
  with open(metadata_path, 'w') as f:
    json.dump(metadata["_metadata"], f, indent=2)
  typer.secho(f"\n  Metadata: {metadata_path}", fg=typer.colors.WHITE)
  
  # Show sample queries
  typer.secho("\n📝 Sample Generated Queries:", fg=typer.colors.CYAN, bold=True)
  for i, sample in enumerate(dataset[:5], 1):
    query_type = sample["metadata"]["query_type"]
    complexity = sample["metadata"]["complexity"]
    typer.secho(
      f" {i}. [{complexity:8}] {sample['query'][:80]}...",
      fg=typer.colors.WHITE
    )
    typer.secho(
      f"   Type: {query_type}",
      fg=typer.colors.BRIGHT_BLACK
    )
  
  typer.secho(f"\n Next steps:", fg=typer.colors.YELLOW)
  typer.secho(f" 1. Run evaluation: python -m app.cli eval-search --dataset {output_path}", fg=typer.colors.WHITE)
  typer.secho(f" 2. Run chat eval: python -m app.cli eval-chat --dataset {output_path}", fg=typer.colors.WHITE)


@app.command()
def check_price(
  product_name: str = typer.Argument(..., help="Product name to check price for"),
  show_history: bool = typer.Option(False, help="Show price history if available"),
) -> None:
  """Check current price and availability for a product using web search."""
  import os
  from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig, WebSearchCache
  import redis
  
  tavily_key = os.getenv("TAVILY_API_KEY")
  if not tavily_key:
    typer.secho("Error: TAVILY_API_KEY not found in environment", fg=typer.colors.RED)
    raise typer.Exit(1)
  
  # Set up web search with caching
  try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    cache = WebSearchCache(redis_client, default_ttl=3600)  # 1 hour for prices
  except:
    cache = None
  
  config = WebSearchConfig(api_key=tavily_key)
  agent = TavilyWebSearchAgent(config, cache)
  
  typer.secho(f"\nChecking prices for: {product_name}", fg=typer.colors.CYAN, bold=True)
  
  # Get product information
  info = agent.search_product_info(product_name, info_type="price")
  
  if info.get("prices"):
    typer.secho("\nCurrent Prices:", fg=typer.colors.GREEN, bold=True)
    for price_info in info["prices"][:5]:
      typer.secho(f"  {price_info['price']} - {price_info['source']}", fg=typer.colors.WHITE)
      typer.secho(f"    {price_info['url']}", fg=typer.colors.BRIGHT_BLACK)
      typer.secho(f"    Checked: {price_info['date']}", fg=typer.colors.BRIGHT_BLACK)
  else:
    typer.secho("No price information found", fg=typer.colors.YELLOW)
  
  # Also check availability
  info_avail = agent.search_product_info(product_name, info_type="availability")
  
  if info_avail.get("availability"):
    typer.secho("\nAvailability:", fg=typer.colors.GREEN, bold=True)
    for avail in info_avail["availability"][:5]:
      status_color = typer.colors.GREEN if avail['status'] == 'in_stock' else typer.colors.RED
      typer.secho(f"  {avail['retailer']}: {avail['status']}", fg=status_color)


@app.command()
def find_alternatives(
  product_name: str = typer.Argument(..., help="Product to find alternatives for"),
  max_results: int = typer.Option(5, help="Maximum alternatives to show"),
) -> None:
  """Find alternative products using web search."""
  import os
  from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig
  
  tavily_key = os.getenv("TAVILY_API_KEY")
  if not tavily_key:
    typer.secho("Error: TAVILY_API_KEY not found in environment", fg=typer.colors.RED)
    raise typer.Exit(1)
  
  config = WebSearchConfig(api_key=tavily_key)
  agent = TavilyWebSearchAgent(config)
  
  typer.secho(f"\nFinding alternatives to: {product_name}", fg=typer.colors.CYAN, bold=True)
  
  alternatives = agent.search_alternatives(product_name, max_alternatives=max_results)
  
  if alternatives:
    typer.secho(f"\nFound {len(alternatives)} alternatives:\n", fg=typer.colors.GREEN)
    for idx, alt in enumerate(alternatives, 1):
      typer.secho(f"{idx}. {alt['name']}", fg=typer.colors.WHITE, bold=True)
      typer.secho(f"   Why: {alt['reason']}", fg=typer.colors.YELLOW)
      typer.secho(f"   Source: {alt['source']}", fg=typer.colors.BRIGHT_BLACK)
      typer.secho(f"   {alt['url']}", fg=typer.colors.BRIGHT_BLACK)
      typer.echo()
  else:
    typer.secho("No alternatives found", fg=typer.colors.YELLOW)


if __name__ == "__main__":
  app()


