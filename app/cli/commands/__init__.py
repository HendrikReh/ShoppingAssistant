"""CLI commands module."""

from .ingest import ingest
from .search import search
from .chat import chat
from .interactive import interactive
from .eval import eval_search, eval_chat
from .web import check_price, find_alternatives
from .testset import generate_testset

__all__ = [
    "ingest",
    "search",
    "chat",
    "interactive",
    "eval_search",
    "eval_chat",
    "check_price",
    "find_alternatives",
    "generate_testset",
]