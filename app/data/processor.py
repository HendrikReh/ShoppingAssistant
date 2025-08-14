"""Data processing utilities for products and reviews."""

import hashlib
import uuid
from typing import List, Dict, Any


def build_product_docs(products: List[Dict]) -> List[Dict]:
    """Build standardized product documents.
    
    Args:
        products: Raw product data
        
    Returns:
        List of processed product documents
    """
    docs = []
    for p in products:
        doc = {
            "id": p.get("id", ""),
            "title": p.get("title", ""),
            "category": p.get("main_category", ""),
            "rating": float(p.get("rating", 0.0)),
            "num_reviews": int(p.get("ratings", 0)),
            "description": ""
        }
        
        # Handle description field (might be list or string)
        if "description" in p:
            desc = p["description"]
            if isinstance(desc, list):
                doc["description"] = " ".join(desc)
            else:
                doc["description"] = str(desc)
        
        # Add price if available
        if "price" in p:
            doc["price"] = p["price"]
        
        docs.append(doc)
    
    return docs


def build_review_docs(reviews: List[Dict]) -> List[Dict]:
    """Build standardized review documents.
    
    Args:
        reviews: Raw review data
        
    Returns:
        List of processed review documents
    """
    docs = []
    for r in reviews:
        # Combine title and text for review content
        review_text = ""
        if "title" in r:
            review_text = r["title"]
        if "text" in r:
            if review_text:
                review_text += " " + r["text"]
            else:
                review_text = r["text"]
        
        doc = {
            "id": f"rev::{r.get('parent_asin', '')}::{len(docs)}",
            "product_id": r.get("parent_asin", ""),
            "review": review_text or " ",
            "rating": float(r.get("rating", 0.0)),
            "helpful_votes": int(r.get("helpful_vote", 0)),
            "verified": r.get("verified_purchase", False)
        }
        
        docs.append(doc)
    
    return docs


def product_text(doc: Dict) -> str:
    """Generate searchable text for a product document.
    
    Args:
        doc: Product document
        
    Returns:
        Searchable text representation
    """
    parts = []
    
    if doc.get("title"):
        parts.append(doc["title"])
    
    if doc.get("category"):
        parts.append(doc["category"])
    
    if doc.get("description"):
        parts.append(doc["description"])
    
    return " ".join(parts)


def review_text(doc: Dict) -> str:
    """Generate searchable text for a review document.
    
    Args:
        doc: Review document
        
    Returns:
        Searchable text representation
    """
    return doc.get("review", "")


def to_context_text(payload: Dict) -> str:
    """Convert payload to context text for LLM.
    
    Args:
        payload: Document payload
        
    Returns:
        Formatted context text
    """
    # Check if it's a product
    if "title" in payload and "category" in payload:
        parts = [f"Product: {payload.get('title', '')}"]
        
        if payload.get('category'):
            parts.append(f"Category: {payload['category']}")
        
        if payload.get('rating'):
            parts.append(f"Rating: {payload['rating']}")
        
        if payload.get('num_reviews'):
            parts.append(f"Reviews: {payload['num_reviews']}")
        
        if payload.get('description'):
            parts.append(f"Description: {payload['description']}")
        
        return " | ".join(parts)
    
    # Check if it's a review
    elif "review" in payload:
        parts = []
        
        if payload.get('product_id'):
            parts.append(f"Product ID: {payload['product_id']}")
        
        if payload.get('rating'):
            parts.append(f"Rating: {payload['rating']}")
        
        parts.append(f"Review: {payload.get('review', '')}")
        
        return " | ".join(parts)
    
    # Fallback to simple text extraction
    elif "text" in payload:
        return payload["text"]
    
    else:
        # Try to extract any text-like field
        for field in ['content', 'description', 'summary']:
            if field in payload:
                return str(payload[field])
        
        # Last resort: stringify the entire payload
        return str(payload)


def to_uuid_from_string(value: str) -> str:
    """Generate deterministic UUID from string.
    
    Args:
        value: Input string
        
    Returns:
        UUID string
    """
    hexdigest = hashlib.md5(value.encode()).hexdigest()
    return str(uuid.UUID(hexdigest))


def extract_metadata(doc: Dict) -> Dict[str, Any]:
    """Extract metadata from document for indexing.
    
    Args:
        doc: Document dictionary
        
    Returns:
        Metadata dictionary
    """
    metadata = {}
    
    # Common fields to include as metadata
    for field in ['id', 'title', 'category', 'rating', 'num_reviews', 
                  'product_id', 'helpful_votes', 'verified']:
        if field in doc:
            metadata[field] = doc[field]
    
    # Add original_id for UUID mapping
    if 'id' in doc:
        metadata['original_id'] = doc['id']
    
    return metadata