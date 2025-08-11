"""Fast LLM-based query correction with caching."""

import re
from typing import Optional, Tuple, Dict
from functools import lru_cache
import dspy
from app.llm_config import get_llm_config


class SimpleQueryCorrector(dspy.Signature):
    """Fix typos in product search query. Return only the corrected text."""
    
    query = dspy.InputField()
    corrected = dspy.OutputField(desc="The query with typos fixed")


# Cache for corrections to avoid repeated LLM calls
_correction_cache: Dict[str, str] = {}
_corrector = None


def get_corrector():
    """Get or create the corrector instance."""
    global _corrector
    if _corrector is None:
        llm_config = get_llm_config()
        lm = llm_config.get_dspy_lm(task="query_correction")
        dspy.configure(lm=lm)
        _corrector = dspy.Predict(SimpleQueryCorrector)
    return _corrector


@lru_cache(maxsize=128)
def correct_query_cached(query: str) -> str:
    """
    Correct query with caching for performance.
    
    Args:
        query: Search query that may have typos
        
    Returns:
        Corrected query
    """
    query_lower = query.lower().strip()
    
    # Check cache first
    if query_lower in _correction_cache:
        return _correction_cache[query_lower]
    
    try:
        corrector = get_corrector()
        
        # Simple prompt for speed
        result = corrector(query=query)
        corrected = result.corrected.strip()
        
        # Clean up the response
        # Remove quotes
        corrected = corrected.strip("'\"")
        
        # If response contains arrow, take the part after it
        if "→" in corrected:
            corrected = corrected.split("→")[-1].strip()
        
        # Cache the result
        _correction_cache[query_lower] = corrected
        
        return corrected
        
    except Exception as e:
        print(f"Query correction error: {e}")
        return query


def needs_correction(query: str) -> bool:
    """
    Quick heuristic to check if query might need correction.
    This avoids LLM calls for obviously correct queries.
    
    Args:
        query: The search query
        
    Returns:
        True if query might have typos
    """
    # Common indicators of typos
    typo_patterns = [
        r'(.)\1{3,}',  # Same letter repeated 3+ times
        r'\b[bcdfghjklmnpqrstvwxyz]{4,}\b',  # 4+ consonants in a row
        r'teh\b',  # Common typo for "the"
        r'recieve',  # Common spelling error
        r'occured',  # Missing r
        r'seperate',  # Common spelling error
    ]
    
    query_lower = query.lower()
    
    # Check for obvious typo patterns
    for pattern in typo_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Check for common product term typos
    common_typos = {
        'stciks': 'sticks',
        'earbds': 'earbuds',
        'keybord': 'keyboard',
        'mous': 'mouse',
        'speker': 'speaker',
        'hedphones': 'headphones',
        'blutooth': 'bluetooth',
        'wireles': 'wireless',
        'hardrive': 'hard drive',
        'eksternal': 'external',
        'mechancal': 'mechanical',
        'noice': 'noise',
        'chargr': 'charger',
        'samsum': 'samsung',
        'mackbook': 'macbook',
    }
    
    for typo in common_typos:
        if typo in query_lower:
            return True
    
    return False


def correct_query_fast(query: str) -> Tuple[str, bool]:
    """
    Fast query correction with pre-filtering.
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (corrected_query, was_corrected)
    """
    # First check with heuristics
    if not needs_correction(query):
        return query, False
    
    # If likely has typos, use LLM
    corrected = correct_query_cached(query)
    was_corrected = corrected.lower() != query.lower()
    
    return corrected, was_corrected


def suggest_correction(query: str) -> Optional[str]:
    """
    Suggest a correction if needed.
    
    Args:
        query: Search query
        
    Returns:
        Corrected query or None if no correction needed
    """
    corrected, was_corrected = correct_query_fast(query)
    
    if was_corrected:
        return corrected
    return None


# Pre-populate cache with common corrections
_correction_cache.update({
    'tv stciks': 'tv sticks',
    'fire tv stciks': 'fire tv sticks',
    'wireles earbds': 'wireless earbuds',
    'mechancal keybord': 'mechanical keyboard',
    'blutooth speker': 'bluetooth speaker',
    'eksternal hardrive': 'external hard drive',
    'gaming mous': 'gaming mouse',
    'noice canceling': 'noise canceling',
    'iphone chargr': 'iphone charger',
    'samsum': 'samsung',
    'mackbook': 'macbook',
})