"""LLM-based query correction for typos and improvements."""

import dspy
from typing import Optional, Tuple
from app.llm_config import get_llm_config


class QueryCorrector(dspy.Signature):
    """Correct typos and improve product search queries."""
    
    original_query = dspy.InputField(
        desc="The user's original search query that may contain typos or be unclear"
    )
    corrected_query = dspy.OutputField(
        desc="The corrected query with typos fixed and clarity improved. Return ONLY the corrected query text, nothing else."
    )


class QueryCorrectionModule(dspy.Module):
    """Module for correcting search queries using LLM."""
    
    def __init__(self):
        super().__init__()
        self.correct = dspy.ChainOfThought(QueryCorrector)
    
    def forward(self, query: str) -> str:
        """
        Correct a query using LLM.
        
        Args:
            query: Original query that may have typos
            
        Returns:
            Corrected query string
        """
        result = self.correct(original_query=query)
        return result.corrected_query


# Global instance to avoid reloading
_corrector_instance = None


def get_query_corrector() -> QueryCorrectionModule:
    """Get or create the global query corrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        # Configure DSPy with LLM
        llm_config = get_llm_config()
        lm = llm_config.get_dspy_lm(task="query_correction")
        dspy.configure(lm=lm)
        _corrector_instance = QueryCorrectionModule()
    return _corrector_instance


def correct_query(query: str, check_threshold: float = 0.9) -> Tuple[str, bool]:
    """
    Correct typos in a query using LLM.
    
    Args:
        query: The search query potentially containing typos
        check_threshold: Not used with LLM approach, kept for compatibility
        
    Returns:
        Tuple of (corrected_query, was_corrected)
    """
    try:
        corrector = get_query_corrector()
        corrected = corrector(query).strip()
        
        # Check if correction was made
        was_corrected = corrected.lower() != query.lower()
        
        return corrected, was_corrected
        
    except Exception as e:
        # If LLM fails, return original query
        print(f"Query correction failed: {e}")
        return query, False


def suggest_correction(query: str) -> Optional[str]:
    """
    Suggest a correction if the query appears to have typos.
    
    Returns None if no correction needed, or the corrected string.
    """
    corrected, was_corrected = correct_query(query)
    
    if was_corrected:
        return corrected
    return None


# Examples for few-shot learning
CORRECTION_EXAMPLES = [
    dspy.Example(
        original_query="TV Stciks",
        corrected_query="TV Sticks"
    ),
    dspy.Example(
        original_query="wireles earbds",
        corrected_query="wireless earbuds"
    ),
    dspy.Example(
        original_query="mechancal keybord",
        corrected_query="mechanical keyboard"
    ),
    dspy.Example(
        original_query="eksternal hard driv 2tb",
        corrected_query="external hard drive 2tb"
    ),
    dspy.Example(
        original_query="blutooth speker",
        corrected_query="bluetooth speaker"
    ),
    dspy.Example(
        original_query="gaming mous rgb",
        corrected_query="gaming mouse rgb"
    ),
    dspy.Example(
        original_query="iphone chargr cable",
        corrected_query="iphone charger cable"
    ),
    dspy.Example(
        original_query="laptop stnd adjustable",
        corrected_query="laptop stand adjustable"
    )
]


class ImprovedQueryCorrector(dspy.Module):
    """Enhanced query corrector with few-shot examples."""
    
    def __init__(self):
        super().__init__()
        self.correct = dspy.ChainOfThought(QueryCorrector)
    
    def forward(self, query: str) -> str:
        """Correct query with few-shot examples for better accuracy."""
        
        # Build a prompt with examples
        prompt_examples = []
        for ex in CORRECTION_EXAMPLES[:3]:  # Use top 3 examples
            prompt_examples.append(f"'{ex.original_query}' → '{ex.corrected_query}'")
        
        # Create an enhanced prompt
        enhanced_prompt = f"""Fix any typos in this search query. Only return the corrected query.

Examples of corrections:
{chr(10).join(prompt_examples)}

Query to correct: {query}"""
        
        try:
            result = self.correct(original_query=enhanced_prompt)
            # Extract just the corrected query from response
            corrected = result.corrected_query.strip()
            
            # Clean up response if it contains extra text
            if "→" in corrected:
                corrected = corrected.split("→")[-1].strip()
            if ":" in corrected and len(corrected.split(":")) == 2:
                corrected = corrected.split(":")[-1].strip()
            
            # Remove quotes if present
            corrected = corrected.strip("'\"")
            
            return corrected
        except:
            # Fallback to simple correction
            result = self.correct(original_query=query)
            return result.corrected_query