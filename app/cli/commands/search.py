"""Search command implementation."""

import typer
from pathlib import Path
from typing import Optional, List
import time

from app.search import BM25Search, VectorSearch, rrf_fuse, CrossEncoderReranker
from app.search.bm25 import create_bm25_index
from app.models import load_sentence_transformer
from app.data import to_context_text
from app.utils import format_seconds, tokenize

# Default paths
DEFAULT_PRODUCTS = Path("data/top_1000_products.jsonl")
DEFAULT_REVIEWS = Path("data/100_top_reviews_of_the_top_1000_products.jsonl")


def search(
    query: Optional[str] = typer.Option(None, help="Search query"),
    top_k: int = typer.Option(20, help="Number of results"),
    variant: str = typer.Option("rrf_ce", help="Search variant: bm25, vec, rrf, rrf_ce"),
    products_path: Path = typer.Option(DEFAULT_PRODUCTS, help="Products JSONL path"),
    reviews_path: Path = typer.Option(DEFAULT_REVIEWS, help="Reviews JSONL path"),
    rrf_k: int = typer.Option(60, help="RRF k parameter"),
    rerank: bool = typer.Option(True, help="Use cross-encoder reranking"),
    rerank_top_k: int = typer.Option(30, help="Rerank top-k candidates"),
    web: bool = typer.Option(False, help="Include web search results"),
    web_only: bool = typer.Option(False, help="Use only web search")
):
    """Search for products and reviews."""
    
    # Interactive mode if no query provided
    if not query:
        typer.secho("Entering interactive search mode...", fg=typer.colors.CYAN)
        _interactive_search(
            top_k=top_k,
            variant=variant,
            products_path=products_path,
            reviews_path=reviews_path,
            rrf_k=rrf_k,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
            web=web,
            web_only=web_only
        )
        return
    
    # Single query search
    results = perform_search(
        query=query,
        top_k=top_k,
        variant=variant,
        products_path=products_path,
        reviews_path=reviews_path,
        rrf_k=rrf_k,
        rerank=rerank,
        rerank_top_k=rerank_top_k,
        web=web,
        web_only=web_only
    )
    
    # Display results
    display_search_results(results, query)


def perform_search(
    query: str,
    top_k: int = 20,
    variant: str = "rrf_ce",
    products_path: Path = DEFAULT_PRODUCTS,
    reviews_path: Path = DEFAULT_REVIEWS,
    rrf_k: int = 60,
    rerank: bool = True,
    rerank_top_k: int = 30,
    web: bool = False,
    web_only: bool = False
) -> List[tuple]:
    """Perform search with specified parameters.
    
    Returns:
        List of (doc_id, score, payload) tuples
    """
    start_time = time.time()
    
    if web_only:
        # Web-only search
        results = _perform_web_search(query, top_k)
    elif web:
        # Hybrid local + web search
        local_results = _perform_local_search(
            query, top_k, variant, products_path, reviews_path,
            rrf_k, rerank, rerank_top_k
        )
        web_results = _perform_web_search(query, top_k // 2)
        results = _merge_results(local_results, web_results, top_k)
    else:
        # Local-only search
        results = _perform_local_search(
            query, top_k, variant, products_path, reviews_path,
            rrf_k, rerank, rerank_top_k
        )
    
    elapsed = time.time() - start_time
    typer.secho(f"Search completed in {format_seconds(elapsed)}", fg=typer.colors.GREEN)
    
    return results


def _perform_local_search(
    query: str,
    top_k: int,
    variant: str,
    products_path: Path,
    reviews_path: Path,
    rrf_k: int,
    rerank: bool,
    rerank_top_k: int
) -> List[tuple]:
    """Perform local search using BM25/vector/hybrid."""
    
    # Initialize search components
    typer.echo("Loading search indices...")
    
    # Load BM25
    bm25_prod, bm25_rev, id_to_product, id_to_review = create_bm25_index(
        products_path, reviews_path
    )
    
    results = []
    
    if variant in ["vec", "rrf", "rrf_ce"]:
        # Load embedding model
        typer.echo("Loading embedding model...")
        st_manager = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        
        if st_manager:
            # Generate query embedding
            query_vec = st_manager.encode_single(query)
            
            # Vector search
            vector_search = VectorSearch()
            vec_products = vector_search.search_with_id_mapping(
                "products_minilm", query_vec, top_k
            )
            vec_reviews = vector_search.search_with_id_mapping(
                "reviews_minilm", query_vec, top_k
            )
    
    # Get BM25 results
    query_tokens = tokenize(query)
    bm25_products = bm25_prod.search(query_tokens, top_k)
    bm25_reviews = bm25_rev.search(query_tokens, top_k)
    
    if variant == "bm25":
        # BM25 only
        results = _combine_and_format_results(
            bm25_products, bm25_reviews,
            id_to_product, id_to_review, top_k
        )
    
    elif variant == "vec":
        # Vector only
        results = _combine_and_format_results(
            vec_products, vec_reviews,
            id_to_product, id_to_review, top_k
        )
    
    elif variant in ["rrf", "rrf_ce"]:
        # RRF fusion
        fused = rrf_fuse(
            [bm25_products, vec_products, bm25_reviews, vec_reviews],
            k=rrf_k,
            product_boost=1.5
        )
        
        # Format results
        sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:rerank_top_k]
        
        for doc_id, score in sorted_results:
            if doc_id in id_to_product:
                results.append((doc_id, score, id_to_product[doc_id]))
            elif doc_id in id_to_review:
                results.append((doc_id, score, id_to_review[doc_id]))
        
        # Cross-encoder reranking if requested
        if variant == "rrf_ce" and rerank and results:
            typer.echo("Reranking with cross-encoder...")
            reranker = CrossEncoderReranker()
            
            if reranker.model:
                reranked = reranker.rerank_with_payloads(
                    query, results, to_context_text, top_k
                )
                results = reranked
    
    return results[:top_k]


def _perform_web_search(query: str, top_k: int) -> List[tuple]:
    """Perform web search using Tavily."""
    try:
        from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig
        import os
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            typer.secho("TAVILY_API_KEY not set, skipping web search", fg=typer.colors.YELLOW)
            return []
        
        config = WebSearchConfig(api_key=api_key, max_results=top_k)
        agent = TavilyWebSearchAgent(config)
        
        web_results = agent.search(query)
        
        # Format as tuples
        formatted = []
        for r in web_results:
            formatted.append((
                f"web::{r.url}",
                r.score,
                {
                    "title": r.title,
                    "content": r.content,
                    "url": r.url,
                    "source": "web"
                }
            ))
        
        return formatted
        
    except Exception as e:
        typer.secho(f"Web search failed: {e}", fg=typer.colors.RED)
        return []


def _combine_and_format_results(
    products: List[tuple],
    reviews: List[tuple],
    id_to_product: dict,
    id_to_review: dict,
    top_k: int
) -> List[tuple]:
    """Combine and format search results."""
    results = []
    
    # Add products
    for doc_id, score in products:
        if doc_id in id_to_product:
            results.append((doc_id, score, id_to_product[doc_id]))
    
    # Add reviews
    for doc_id, score in reviews:
        if doc_id in id_to_review:
            results.append((doc_id, score, id_to_review[doc_id]))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def _merge_results(local: List[tuple], web: List[tuple], top_k: int) -> List[tuple]:
    """Merge local and web results."""
    # Simple interleaving strategy
    merged = []
    local_idx = 0
    web_idx = 0
    
    while len(merged) < top_k and (local_idx < len(local) or web_idx < len(web)):
        # Add from local
        if local_idx < len(local):
            merged.append(local[local_idx])
            local_idx += 1
        
        # Add from web
        if web_idx < len(web) and len(merged) < top_k:
            merged.append(web[web_idx])
            web_idx += 1
    
    return merged


def display_search_results(results: List[tuple], query: str):
    """Display search results."""
    if not results:
        typer.secho("No results found.", fg=typer.colors.YELLOW)
        return
    
    typer.secho(f"\nSearch Results for: {query}", fg=typer.colors.CYAN, bold=True)
    typer.secho("-" * 60, fg=typer.colors.BLUE)
    
    for i, (doc_id, score, payload) in enumerate(results, 1):
        # Determine source
        if doc_id.startswith("web::"):
            source = "[WEB]"
            color = typer.colors.MAGENTA
        elif doc_id.startswith("rev::"):
            source = "[RAG] REVIEW"
            color = typer.colors.YELLOW
        else:
            source = "[RAG] PRODUCT"
            color = typer.colors.GREEN
        
        # Determine score color
        if score > 0.8:
            score_color = typer.colors.GREEN
        elif score > 0.5:
            score_color = typer.colors.YELLOW
        else:
            score_color = typer.colors.WHITE
        
        # Display result
        typer.secho(f"\n{i}. {source}", fg=color, bold=True)
        
        if "title" in payload:
            typer.echo(f"   Title: {payload['title']}")
        
        if "category" in payload:
            typer.echo(f"   Category: {payload['category']}")
        
        if "rating" in payload and payload["rating"] > 0:
            typer.echo(f"   Rating: {payload['rating']:.1f}/5")
        
        if "url" in payload:
            typer.echo(f"   URL: {payload['url']}")
        
        typer.secho(f"   Score: {score:.4f}", fg=score_color)
        
        # Show snippet
        snippet = _get_snippet(payload)
        if snippet:
            typer.echo(f"   {snippet[:150]}...")


def _get_snippet(payload: dict) -> str:
    """Extract snippet from payload."""
    for field in ["description", "content", "review", "text"]:
        if field in payload and payload[field]:
            return str(payload[field])
    return ""


def _interactive_search(**kwargs):
    """Interactive search mode."""
    typer.secho("\nInteractive Search Mode", fg=typer.colors.CYAN, bold=True)
    typer.echo("Commands: /help, /settings, /exit")
    
    while True:
        try:
            query = typer.prompt("\nYou")
            
            if query.lower() in ["/exit", "/quit", "exit", "quit"]:
                break
            
            if query.lower() == "/help":
                typer.echo("Tips:")
                typer.echo("  - Enter product names to search")
                typer.echo("  - Use 'products only' to exclude reviews")
                typer.echo("  - Try different search variants with /settings")
                continue
            
            if query.lower() == "/settings":
                typer.echo(f"Current settings:")
                typer.echo(f"  - Variant: {kwargs.get('variant', 'rrf_ce')}")
                typer.echo(f"  - Top-k: {kwargs.get('top_k', 20)}")
                typer.echo(f"  - Web search: {kwargs.get('web', False)}")
                continue
            
            # Perform search
            results = perform_search(query, **kwargs)
            display_search_results(results, query)
            
        except KeyboardInterrupt:
            break
    
    typer.secho("\nExiting search mode...", fg=typer.colors.CYAN)