"""Chat and interactive commands implementation.

Keeps behavior compatible with existing CLI flags and flows.
"""

from __future__ import annotations

import sys
import typer

from app.llm_config import get_llm_config
from app.cli import (
    _device_str,
    _load_st_model,
    _qdrant_client,
    _vector_search,
    COLLECTION_PRODUCTS,
    COLLECTION_REVIEWS,
)


def chat_once(question: str, top_k: int = 8) -> str:
    import dspy

    llm_config = get_llm_config()
    lm = llm_config.get_dspy_lm(task="chat")
    dspy.configure(lm=lm)
    rag = dspy.Predict("question, context -> answer")

    device = _device_str()
    st_model = _load_st_model("sentence-transformers/all-MiniLM-L6-v2", device=device)
    client = _qdrant_client()

    q_vec = (
        st_model.encode([question], batch_size=1, normalize_embeddings=True, device=device, convert_to_numpy=True)[0].tolist()
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

    ctx = "\n\n".join(texts[:top_k])
    pred = rag(question=question, context=ctx)
    return getattr(pred, "answer", "")


def interactive_loop() -> None:
    typer.secho("\nInteractive Chat Mode", fg=typer.colors.CYAN, bold=True)
    typer.secho("I can help you find products, compare items, and answer questions.", fg=typer.colors.WHITE)
    typer.secho("\nCommands:", fg=typer.colors.WHITE)
    typer.secho(" /help - Show example questions", fg=typer.colors.WHITE)
    typer.secho(" /context - Show how many contexts are being retrieved", fg=typer.colors.WHITE)
    typer.secho(" /clear - Clear the screen", fg=typer.colors.WHITE)
    typer.secho(" /exit - Exit chat mode", fg=typer.colors.WHITE)
    typer.secho(" Press Ctrl+C to interrupt\n", fg=typer.colors.WHITE)

    while True:
        try:
            q = typer.prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            typer.secho("\n\n Goodbye!", fg=typer.colors.CYAN)
            raise typer.Exit(130)

        if not q:
            continue

        if q.lower() in {"/exit", "/quit", "exit", "quit"}:
            typer.secho(" Goodbye!", fg=typer.colors.CYAN)
            raise typer.Exit(130)
        elif q.lower() == "/help":
            typer.secho("\n Example Questions:", fg=typer.colors.CYAN, bold=True)
            typer.secho(" • What are the best wireless earbuds under $200?", fg=typer.colors.WHITE)
            typer.secho(" • Compare Sony WH-1000XM4 and Bose QuietComfort", fg=typer.colors.WHITE)
            typer.secho(" • Which laptop is good for programming?", fg=typer.colors.WHITE)
            continue
        elif q.lower() == "/clear":
            import os
            os.system('clear' if os.name == 'posix' else 'cls')
            typer.secho("Interactive Chat Mode", fg=typer.colors.CYAN, bold=True)
            continue

        typer.secho("\nThinking...", fg=typer.colors.BLUE, italic=True)
        a = chat_once(q)
        typer.echo("\033[F\033[K", nl=False)
        typer.secho("Assistant: ", fg=typer.colors.GREEN, bold=True)
        import textwrap
        wrapped = textwrap.fill(a, width=80, initial_indent="  ", subsequent_indent="  ")
        typer.echo(f"{wrapped}\n")


