"""Refactored Shopping Assistant CLI.

This is a simplified version showing how the CLI would look after refactoring.
The actual migration would involve updating all commands to use the new modules.
"""

from pathlib import Path
import typer
from typing import Optional

# Import from new modular structure
from app.cli.commands import (
    ingest,
    search,
    chat,
    interactive,
    eval_search,
    eval_chat,
    check_price,
    find_alternatives,
    generate_testset
)

# Create the main app
app = typer.Typer(
    help="Shopping Assistant CLI - Refactored Version",
    no_args_is_help=True
)

# Register commands
app.command()(ingest)
app.command()(search)
app.command()(chat)
app.command()(interactive)
app.command("eval-search")(eval_search)
app.command("eval-chat")(eval_chat)
app.command("check-price")(check_price)
app.command("find-alternatives")(find_alternatives)
app.command("generate-testset")(generate_testset)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version")
):
    """Shopping Assistant - Advanced e-commerce search and recommendation system."""
    if version:
        typer.echo("Shopping Assistant v0.2.0 (Refactored)")
        raise typer.Exit()


if __name__ == "__main__":
    app()