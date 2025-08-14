"""Chat/Q&A evaluation functionality using RAGAS."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Container for chat evaluation results."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    generation_time: float
    retrieval_time: float
    ragas_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class ChatEvaluator:
    """Evaluate chat/Q&A system performance."""
    
    def __init__(self, chat_fn, retrieval_fn):
        """Initialize chat evaluator.
        
        Args:
            chat_fn: Function that generates answers
            retrieval_fn: Function that retrieves context
        """
        self.chat_fn = chat_fn
        self.retrieval_fn = retrieval_fn
        self.results = []
    
    def evaluate_question(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        top_k: int = 8,
        **kwargs
    ) -> ChatResult:
        """Evaluate a single question.
        
        Args:
            question: Question to answer
            ground_truth: Expected answer (for evaluation)
            top_k: Number of contexts to retrieve
            **kwargs: Additional parameters
            
        Returns:
            ChatResult object
        """
        # Retrieve context
        retrieval_start = time.time()
        try:
            search_results = self.retrieval_fn(
                query=question,
                top_k=top_k,
                **kwargs
            )
            contexts = [self._extract_context(r) for r in search_results]
            retrieval_time = time.time() - retrieval_start
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            contexts = []
            retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        try:
            answer = self.chat_fn(
                question=question,
                contexts=contexts
            )
            generation_time = time.time() - generation_start
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "I couldn't generate an answer due to an error."
            generation_time = time.time() - generation_start
        
        result = ChatResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            metadata=kwargs
        )
        
        self.results.append(result)
        return result
    
    def evaluate_dataset(
        self,
        questions: List[Dict[str, str]],
        top_k: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate multiple questions.
        
        Args:
            questions: List of question dicts with 'question' and optional 'ground_truth'
            top_k: Number of contexts
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics
        """
        results = []
        for item in questions:
            result = self.evaluate_question(
                question=item["question"],
                ground_truth=item.get("ground_truth"),
                top_k=top_k,
                **kwargs
            )
            results.append(result)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        metrics["detailed_results"] = [asdict(r) for r in results]
        
        return metrics
    
    def calculate_metrics(self, results: List[ChatResult]) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            results: List of chat results
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {}
        
        num_questions = len(results)
        avg_generation_time = sum(r.generation_time for r in results) / num_questions
        avg_retrieval_time = sum(r.retrieval_time for r in results) / num_questions
        avg_contexts = sum(len(r.contexts) for r in results) / num_questions
        
        metrics = {
            "num_questions": num_questions,
            "avg_generation_time": avg_generation_time,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_total_time": avg_generation_time + avg_retrieval_time,
            "avg_contexts_used": avg_contexts
        }
        
        # Add RAGAS scores if available
        if any(r.ragas_scores for r in results):
            ragas_metrics = self._aggregate_ragas_scores(results)
            metrics.update(ragas_metrics)
        
        return metrics
    
    def _extract_context(self, search_result) -> str:
        """Extract context text from search result.
        
        Args:
            search_result: Search result tuple or dict
            
        Returns:
            Context string
        """
        if isinstance(search_result, tuple) and len(search_result) >= 3:
            # (doc_id, score, payload)
            payload = search_result[2]
            return self._payload_to_text(payload)
        elif isinstance(search_result, dict):
            return self._payload_to_text(search_result)
        else:
            return str(search_result)
    
    def _payload_to_text(self, payload: Dict) -> str:
        """Convert payload to text."""
        from ..data.processor import to_context_text
        return to_context_text(payload)
    
    def _aggregate_ragas_scores(self, results: List[ChatResult]) -> Dict[str, float]:
        """Aggregate RAGAS scores across results."""
        score_sums = {}
        score_counts = {}
        
        for result in results:
            if result.ragas_scores:
                for metric, score in result.ragas_scores.items():
                    if metric not in score_sums:
                        score_sums[metric] = 0
                        score_counts[metric] = 0
                    score_sums[metric] += score
                    score_counts[metric] += 1
        
        # Calculate averages
        avg_scores = {}
        for metric in score_sums:
            avg_scores[f"avg_{metric}"] = score_sums[metric] / score_counts[metric]
        
        return avg_scores


def evaluate_chat_with_ragas(
    questions: List[Dict[str, str]],
    chat_fn,
    retrieval_fn,
    top_k: int = 8,
    use_ragas: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate chat system with optional RAGAS metrics.
    
    Args:
        questions: List of questions with ground truth
        chat_fn: Chat function
        retrieval_fn: Retrieval function
        top_k: Number of contexts
        use_ragas: Whether to use RAGAS evaluation
        **kwargs: Additional parameters
        
    Returns:
        Evaluation results with metrics
    """
    evaluator = ChatEvaluator(chat_fn, retrieval_fn)
    
    # Generate answers
    results = []
    for item in questions:
        result = evaluator.evaluate_question(
            question=item["question"],
            ground_truth=item.get("ground_truth"),
            top_k=top_k,
            **kwargs
        )
        results.append(result)
    
    # Apply RAGAS evaluation if requested
    if use_ragas and any(r.ground_truth for r in results):
        try:
            ragas_scores = apply_ragas_evaluation(results)
            for result, scores in zip(results, ragas_scores):
                result.ragas_scores = scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
    
    # Calculate final metrics
    metrics = evaluator.calculate_metrics(results)
    metrics["detailed_results"] = [asdict(r) for r in results]
    
    return metrics


def apply_ragas_evaluation(results: List[ChatResult]) -> List[Dict[str, float]]:
    """Apply RAGAS evaluation to chat results.
    
    Args:
        results: List of chat results
        
    Returns:
        List of RAGAS scores for each result
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ..ragas_config import configure_ragas_metrics
        
        # Prepare data for RAGAS
        data = {
            "question": [r.question for r in results],
            "answer": [r.answer for r in results],
            "contexts": [r.contexts for r in results],
            "ground_truth": [r.ground_truth or "" for r in results]
        }
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Configure metrics
        metrics = configure_ragas_metrics()
        
        # Run evaluation
        eval_result = ragas_evaluate(dataset, metrics=metrics)
        
        # Extract scores for each sample
        if hasattr(eval_result, 'scores') and eval_result.scores:
            if isinstance(eval_result.scores, list):
                return eval_result.scores
            else:
                # Single score dict, replicate for all results
                return [eval_result.scores] * len(results)
        
        return [{}] * len(results)
        
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return [{}] * len(results)