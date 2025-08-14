"""Text processing utilities."""

import re
from typing import List, Optional


def tokenize(text: str) -> List[str]:
    """Tokenize text for search indexing.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Convert to lowercase and extract words
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    truncate_at = max_length - len(suffix)
    if truncate_at <= 0:
        return text[:max_length]
    
    return text[:truncate_at] + suffix


def normalize_query(query: str) -> str:
    """Normalize search query.
    
    Args:
        query: Query to normalize
        
    Returns:
        Normalized query
    """
    # Remove common prefixes
    prefixes = [
        "search for",
        "find",
        "show me",
        "look for",
        "get",
        "list"
    ]
    
    query_lower = query.lower().strip()
    for prefix in prefixes:
        if query_lower.startswith(prefix):
            query = query[len(prefix):].strip()
            break
    
    return query


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: Text to extract from
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    # Tokenize
    tokens = tokenize(text)
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "need", "dare", "ought", "used", "better", "rather"
    }
    
    keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Count frequencies
    from collections import Counter
    keyword_counts = Counter(keywords)
    
    # Return top keywords
    top_keywords = [kw for kw, _ in keyword_counts.most_common(max_keywords)]
    return top_keywords


def highlight_text(text: str, terms: List[str], before: str = "**", after: str = "**") -> str:
    """Highlight terms in text.
    
    Args:
        text: Text to highlight in
        terms: Terms to highlight
        before: String to insert before term
        after: String to insert after term
        
    Returns:
        Text with highlights
    """
    for term in terms:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f"{before}{term}{after}", text)
    
    return text


def similarity_ratio(text1: str, text2: str) -> float:
    """Calculate simple similarity ratio between texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity ratio (0 to 1)
    """
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union) if union else 0.0