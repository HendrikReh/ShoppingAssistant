"""Data module for Shopping Assistant.

This module provides data loading, processing, and caching functionality.
"""

from .loader import read_jsonl, load_jsonl, save_json
from .processor import (
    build_product_docs,
    build_review_docs,
    product_text,
    review_text,
    to_context_text
)
from .embeddings import EmbeddingGenerator, embed_texts
from .cache import RedisCache

__all__ = [
    "read_jsonl",
    "load_jsonl",
    "save_json",
    "build_product_docs",
    "build_review_docs",
    "product_text",
    "review_text",
    "to_context_text",
    "EmbeddingGenerator",
    "embed_texts",
    "RedisCache",
]