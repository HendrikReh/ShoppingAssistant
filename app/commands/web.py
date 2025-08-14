"""Web-related CLI commands using Tavily or the higher-level web search agent.

These functions are pure and do not depend on Typer decorators; the
top-level CLI should call them.
"""

from __future__ import annotations

import os
import typer


def check_price_cmd(item: str) -> None:
    """Implementation for price check via Tavily.

    Prints friendly warnings when API key is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        typer.secho(
            "Warning: TAVILY_API_KEY is not set. Please export TAVILY_API_KEY to enable price checks.",
            fg=typer.colors.YELLOW,
        )
        return

    query = f"current price of {item} best retailers"
    used_agent = False
    entries: list[dict] = []
    result = None

    try:
        # Prefer the higher-level agent if available (handles filtering/caching)
        from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig

        config = WebSearchConfig(api_key=api_key, max_results=5, search_depth="advanced")
        agent = TavilyWebSearchAgent(config)
        results = agent.search(query, search_type="price", use_cache=False)
        used_agent = True
        # Normalize to simple dicts for printing
        for r in results:
            entries.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content or "",
                    "score": getattr(r, "score", 0.0),
                }
            )
    except Exception:
        # Fallback to minimal Tavily client
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            result = client.search(query=query, max_results=5)
            if isinstance(result, dict):
                entries = result.get("results") or []
            else:
                entries = []
        except Exception as exc:
            typer.secho(f"Search failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    typer.secho(f"\nPrice check for: '{item}'", fg=typer.colors.CYAN, bold=True)
    if used_agent and not entries:
        typer.secho("No sources found.", fg=typer.colors.YELLOW)
        return
    if not used_agent:
        # Try to display summary if provided by raw API
        try:
            summary = result.get("answer") or result.get("summary")  # type: ignore[union-attr]
            if summary:
                typer.secho(f" {summary}", fg=typer.colors.WHITE)
        except Exception:
            pass

    if entries:
        typer.secho("\nTop sources:", fg=typer.colors.GREEN)
        for idx, r in enumerate(entries[:5], 1):
            title = (r.get("title") or "").strip()[:100]
            url = r.get("url") or ""
            typer.secho(f" {idx}. {title}", fg=typer.colors.WHITE)
            if url:
                typer.secho(f"     {url}", fg=typer.colors.BRIGHT_BLACK)
    else:
        typer.secho("No sources found.", fg=typer.colors.YELLOW)


def find_alternatives_cmd(item: str, max_items: int = 5) -> None:
    """Implementation for finding alternatives via Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        typer.secho(
            "Warning: TAVILY_API_KEY is not set. Please export TAVILY_API_KEY to enable alternatives search.",
            fg=typer.colors.YELLOW,
        )
        return

    query = f"best alternatives to {item} similar products compare"
    used_agent = False
    entries: list[dict] = []

    try:
        from app.web_search_agent import TavilyWebSearchAgent, WebSearchConfig

        config = WebSearchConfig(api_key=api_key, max_results=max_items, search_depth="advanced")
        agent = TavilyWebSearchAgent(config)
        results = agent.search(query, search_type="general", use_cache=False)
        used_agent = True
        for r in results:
            entries.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content or "",
                    "score": getattr(r, "score", 0.0),
                }
            )
    except Exception:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            result = client.search(query=query, max_results=max_items)
            if isinstance(result, dict):
                entries = result.get("results") or []
            else:
                entries = []
        except Exception as exc:
            typer.secho(f"Search failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    typer.secho(f"\nAlternatives for: '{item}'", fg=typer.colors.CYAN, bold=True)
    if not entries:
        typer.secho("No alternatives found.", fg=typer.colors.YELLOW)
        return

    for idx, r in enumerate(entries[:max_items], 1):
        title = (r.get("title") or "").strip()[:100]
        url = r.get("url") or ""
        snippet = (r.get("content") or "").strip()[:140]
        typer.secho(f" {idx}. {title}", fg=typer.colors.WHITE, bold=True)
        if snippet:
            typer.secho(f"    {snippet}", fg=typer.colors.BRIGHT_BLACK)
        if url:
            typer.secho(f"    {url}", fg=typer.colors.BRIGHT_BLACK)


