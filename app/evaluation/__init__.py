"""Evaluation module for Shopping Assistant.

This module provides evaluation functionality for search and chat systems.
"""

from .search_eval import SearchEvaluator, evaluate_search_variants
from .chat_eval import ChatEvaluator, evaluate_chat_with_ragas
from .reporter import EvaluationReporter, generate_evaluation_report

__all__ = [
    "SearchEvaluator",
    "evaluate_search_variants",
    "ChatEvaluator",
    "evaluate_chat_with_ragas",
    "EvaluationReporter",
    "generate_evaluation_report",
]