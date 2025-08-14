"""BM25 keyword search implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 keyword search for products and reviews."""
    
    def __init__(self, corpus: List[List[str]], doc_ids: List[str]):
        """Initialize BM25 search.
        
        Args:
            corpus: List of tokenized documents
            doc_ids: List of document IDs corresponding to corpus
        """
        self.bm25 = BM25Okapi(corpus)
        self.doc_ids = doc_ids
        self.corpus = corpus
    
    def search(self, query_tokens: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """Search for documents using BM25.
        
        Args:
            query_tokens: Tokenized query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (doc_id, score) pairs
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results
    
    @classmethod
    def from_documents(cls, documents: List[Dict], text_key: str = "text") -> "BM25Search":
        """Create BM25 index from documents.
        
        Args:
            documents: List of documents with 'id' and text fields
            text_key: Key for text field in documents
            
        Returns:
            BM25Search instance
        """
        corpus = []
        doc_ids = []
        
        for doc in documents:
            if "id" in doc and text_key in doc:
                tokens = tokenize(doc[text_key])
                corpus.append(tokens)
                doc_ids.append(doc["id"])
        
        return cls(corpus, doc_ids)


def create_bm25_index(
    products_file: Path,
    reviews_file: Path
) -> Tuple[BM25Search, BM25Search, Dict[str, Dict], Dict[str, Dict]]:
    """Create BM25 indices for products and reviews.
    
    Args:
        products_file: Path to products JSONL file
        reviews_file: Path to reviews JSONL file
        
    Returns:
        Tuple of (product_bm25, review_bm25, id_to_product, id_to_review)
    """
    from ..data.loader import read_jsonl
    from ..data.processor import build_product_docs, build_review_docs, product_text, review_text
    
    # Load data
    products_raw = read_jsonl(products_file)
    reviews_raw = read_jsonl(reviews_file)
    
    # Build documents
    product_docs = build_product_docs(products_raw)
    review_docs = build_review_docs(reviews_raw)
    
    # Create ID mappings
    id_to_product = {doc["id"]: doc for doc in product_docs}
    id_to_review = {doc["id"]: doc for doc in review_docs}
    
    # Tokenize and create corpus
    product_corpus = []
    product_ids = []
    for doc in product_docs:
        text = product_text(doc)
        tokens = tokenize(text)
        product_corpus.append(tokens)
        product_ids.append(doc["id"])
    
    review_corpus = []
    review_ids = []
    for doc in review_docs:
        text = review_text(doc)
        tokens = tokenize(text)
        review_corpus.append(tokens)
        review_ids.append(doc["id"])
    
    # Create BM25 indices
    product_bm25 = BM25Search(product_corpus, product_ids)
    review_bm25 = BM25Search(review_corpus, review_ids)
    
    return product_bm25, review_bm25, id_to_product, id_to_review


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    import re
    
    # Convert to lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens