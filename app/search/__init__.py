"""Search module for Shopping Assistant.

This module provides various search implementations including:
- BM25 keyword search
- Vector semantic search
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
"""

from .bm25 import BM25Search, create_bm25_index
from .vector import VectorSearch
from .fusion import rrf_fuse
from .reranker import CrossEncoderReranker

__all__ = [
    "BM25Search",
    "create_bm25_index",
    "VectorSearch",
    "rrf_fuse",
    "CrossEncoderReranker",
]