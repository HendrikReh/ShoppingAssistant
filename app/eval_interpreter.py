"""Evaluation result interpreter for generating detailed insights from metrics.

This module provides functions to interpret RAGAS metrics and generate
actionable recommendations based on evaluation results.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PerformanceLevel(Enum):
    """Performance level categories for metrics."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class MetricInterpretation:
    """Interpretation for a single metric."""
    metric_name: str
    value: float
    level: PerformanceLevel
    interpretation: str
    recommendations: List[str]


def classify_performance(value: float, thresholds: Dict[str, float]) -> PerformanceLevel:
    """Classify performance based on thresholds.
    
    Args:
        value: Metric value to classify
        thresholds: Dict with keys 'excellent', 'good', 'acceptable', 'poor'
    
    Returns:
        PerformanceLevel enum
    """
    if value >= thresholds.get('excellent', 0.9):
        return PerformanceLevel.EXCELLENT
    elif value >= thresholds.get('good', 0.75):
        return PerformanceLevel.GOOD
    elif value >= thresholds.get('acceptable', 0.6):
        return PerformanceLevel.ACCEPTABLE
    elif value >= thresholds.get('poor', 0.4):
        return PerformanceLevel.NEEDS_IMPROVEMENT
    else:
        return PerformanceLevel.POOR


def interpret_context_relevance(value: float) -> MetricInterpretation:
    """Interpret context relevance metric."""
    thresholds = {'excellent': 0.85, 'good': 0.75, 'acceptable': 0.65, 'poor': 0.5}
    level = classify_performance(value, thresholds)
    
    interpretations = {
        PerformanceLevel.EXCELLENT: f"Outstanding context relevance ({value:.1%}). The retrieval system is finding highly relevant information for queries, indicating excellent embedding quality and search algorithms.",
        PerformanceLevel.GOOD: f"Strong context relevance ({value:.1%}). Retrieved contexts are mostly relevant, with room for minor improvements in edge cases.",
        PerformanceLevel.ACCEPTABLE: f"Acceptable context relevance ({value:.1%}). The system finds relevant information for most queries but may struggle with complex or ambiguous requests.",
        PerformanceLevel.NEEDS_IMPROVEMENT: f"Below-target context relevance ({value:.1%}). Many retrieved contexts are only partially relevant, indicating retrieval quality issues.",
        PerformanceLevel.POOR: f"Poor context relevance ({value:.1%}). The retrieval system is struggling to find relevant information, requiring immediate attention."
    }
    
    recommendations_map = {
        PerformanceLevel.EXCELLENT: [
            "Maintain current retrieval configuration",
            "Consider documenting successful patterns for future reference"
        ],
        PerformanceLevel.GOOD: [
            "Fine-tune retrieval for edge cases",
            "Analyze queries with lower scores for patterns"
        ],
        PerformanceLevel.ACCEPTABLE: [
            "Increase retrieval top-k to capture more relevant contexts",
            "Consider query expansion or reformulation techniques",
            "Review embedding model for domain alignment"
        ],
        PerformanceLevel.NEEDS_IMPROVEMENT: [
            "Urgently review embedding model choice",
            "Implement query preprocessing and expansion",
            "Increase retrieval top-k significantly",
            "Consider hybrid search with keyword matching"
        ],
        PerformanceLevel.POOR: [
            "Critical: Redesign retrieval strategy",
            "Switch to a different embedding model",
            "Implement fallback retrieval mechanisms",
            "Add keyword-based search as primary method"
        ]
    }
    
    return MetricInterpretation(
        metric_name="Context Relevance",
        value=value,
        level=level,
        interpretation=interpretations[level],
        recommendations=recommendations_map[level]
    )


def interpret_context_utilization(value: float) -> MetricInterpretation:
    """Interpret context utilization metric for e-commerce RAG."""
    # Note: Lower utilization is often GOOD in e-commerce RAG
    
    if value < 0.1:
        level = PerformanceLevel.ACCEPTABLE
        interpretation = f"Low context utilization ({value:.1%}) is NORMAL for e-commerce RAG. This indicates comprehensive retrieval where only specific details are needed for answers."
        recommendations = [
            "This is expected behavior - no action needed",
            "Ensure retrieved contexts remain comprehensive"
        ]
    elif value < 0.3:
        level = PerformanceLevel.GOOD
        interpretation = f"Moderate context utilization ({value:.1%}) shows balanced retrieval. The system retrieves sufficient context while using relevant portions."
        recommendations = [
            "Monitor for consistency across query types",
            "This is a healthy range for e-commerce systems"
        ]
    elif value < 0.5:
        level = PerformanceLevel.ACCEPTABLE
        interpretation = f"Higher context utilization ({value:.1%}) may indicate either very focused retrieval or verbose answer generation."
        recommendations = [
            "Review if answers are unnecessarily verbose",
            "Check if retrieval is too narrow"
        ]
    else:
        level = PerformanceLevel.NEEDS_IMPROVEMENT
        interpretation = f"Very high context utilization ({value:.1%}) suggests either insufficient retrieval or overly detailed answers."
        recommendations = [
            "Increase retrieval top-k for more context",
            "Review answer generation prompts for conciseness",
            "Check for retrieval diversity"
        ]
    
    return MetricInterpretation(
        metric_name="Context Utilization",
        value=value,
        level=level,
        interpretation=interpretation,
        recommendations=recommendations
    )


def interpret_faithfulness(value: float) -> MetricInterpretation:
    """Interpret faithfulness metric."""
    thresholds = {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7, 'poor': 0.5}
    level = classify_performance(value, thresholds)
    
    interpretations = {
        PerformanceLevel.EXCELLENT: f"Excellent faithfulness ({value:.1%}). Answers are strongly grounded in retrieved contexts with minimal hallucination risk.",
        PerformanceLevel.GOOD: f"Good faithfulness ({value:.1%}). Most answers are well-grounded with occasional minor deviations.",
        PerformanceLevel.ACCEPTABLE: f"Acceptable faithfulness ({value:.1%}). Some answers include information not fully supported by contexts.",
        PerformanceLevel.NEEDS_IMPROVEMENT: f"Concerning faithfulness ({value:.1%}). Significant hallucination detected in answers.",
        PerformanceLevel.POOR: f"Critical faithfulness issue ({value:.1%}). High hallucination rate requiring immediate intervention."
    }
    
    recommendations_map = {
        PerformanceLevel.EXCELLENT: [
            "Maintain current generation settings",
            "Document successful prompt patterns"
        ],
        PerformanceLevel.GOOD: [
            "Review cases with lower scores",
            "Fine-tune temperature settings if needed"
        ],
        PerformanceLevel.ACCEPTABLE: [
            "Lower LLM temperature for more conservative generation",
            "Strengthen grounding instructions in prompts",
            "Increase context quality through better retrieval"
        ],
        PerformanceLevel.NEEDS_IMPROVEMENT: [
            "Urgent: Reduce temperature to 0.3 or lower",
            "Implement strict grounding constraints",
            "Add fact-checking mechanisms",
            "Review and improve prompt engineering"
        ],
        PerformanceLevel.POOR: [
            "Critical: Set temperature to 0",
            "Redesign generation prompts with explicit grounding",
            "Implement answer validation pipeline",
            "Consider switching to more faithful LLM model"
        ]
    }
    
    return MetricInterpretation(
        metric_name="Faithfulness",
        value=value,
        level=level,
        interpretation=interpretations[level],
        recommendations=recommendations_map[level]
    )


def interpret_answer_relevancy(value: float) -> MetricInterpretation:
    """Interpret answer relevancy metric."""
    thresholds = {'excellent': 0.85, 'good': 0.75, 'acceptable': 0.65, 'poor': 0.5}
    level = classify_performance(value, thresholds)
    
    interpretations = {
        PerformanceLevel.EXCELLENT: f"Excellent answer relevancy ({value:.1%}). Responses directly and comprehensively address user queries.",
        PerformanceLevel.GOOD: f"Good answer relevancy ({value:.1%}). Most answers address the query well with minor gaps.",
        PerformanceLevel.ACCEPTABLE: f"Acceptable answer relevancy ({value:.1%}). Answers generally address queries but may miss some aspects.",
        PerformanceLevel.NEEDS_IMPROVEMENT: f"Below-target answer relevancy ({value:.1%}). Many answers partially miss the query intent.",
        PerformanceLevel.POOR: f"Poor answer relevancy ({value:.1%}). Answers frequently fail to address user queries properly."
    }
    
    recommendations_map = {
        PerformanceLevel.EXCELLENT: [
            "Maintain current approach",
            "Use as baseline for future improvements"
        ],
        PerformanceLevel.GOOD: [
            "Analyze lower-scoring samples for patterns",
            "Minor prompt adjustments may help"
        ],
        PerformanceLevel.ACCEPTABLE: [
            "Improve query understanding in prompts",
            "Add explicit instructions to address all query aspects",
            "Consider query reformulation techniques"
        ],
        PerformanceLevel.NEEDS_IMPROVEMENT: [
            "Redesign answer generation prompts",
            "Implement query intent classification",
            "Add answer validation checks",
            "Improve context retrieval relevance"
        ],
        PerformanceLevel.POOR: [
            "Critical: Complete prompt overhaul needed",
            "Implement structured answer generation",
            "Add query-answer alignment validation",
            "Consider different LLM or approach"
        ]
    }
    
    return MetricInterpretation(
        metric_name="Answer Relevancy",
        value=value,
        level=level,
        interpretation=interpretations[level],
        recommendations=recommendations_map[level]
    )


def interpret_search_variant_comparison(variants: Dict[str, Dict[str, float]]) -> str:
    """Generate comparative analysis of search variants."""
    if not variants:
        return "No variant data available for comparison."
    
    interpretation = ["## Variant Performance Comparison\n"]
    
    # Find best variant for each metric
    metrics = set()
    for variant_metrics in variants.values():
        metrics.update(variant_metrics.keys())
    
    best_variants = {}
    for metric in metrics:
        best_score = -1
        best_variant = None
        for variant, scores in variants.items():
            if metric in scores and scores[metric] > best_score:
                best_score = scores[metric]
                best_variant = variant
        if best_variant:
            best_variants[metric] = (best_variant, best_score)
    
    # Overall winner
    variant_wins = {}
    for metric, (variant, _) in best_variants.items():
        variant_wins[variant] = variant_wins.get(variant, 0) + 1
    
    if variant_wins:
        winner = max(variant_wins.items(), key=lambda x: x[1])
        interpretation.append(f"**Overall Best Performer**: `{winner[0]}` (best in {winner[1]}/{len(metrics)} metrics)\n")
    
    # Detailed comparison
    interpretation.append("\n### Variant Analysis:\n")
    
    for variant, scores in sorted(variants.items()):
        interpretation.append(f"\n**{variant.upper()}**:")
        
        # Calculate average performance
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        # Variant-specific insights
        if variant == "bm25":
            interpretation.append(f"- Keyword-based search: {avg_score:.1%} average performance")
            interpretation.append("- Best for: Exact term matching, product names, model numbers")
            interpretation.append("- Limitations: No semantic understanding, misses synonyms")
            
        elif variant == "vec":
            interpretation.append(f"- Pure semantic search: {avg_score:.1%} average performance")
            interpretation.append("- Best for: Conceptual queries, finding similar products")
            interpretation.append("- Limitations: May miss exact keyword matches")
            
        elif variant == "rrf":
            interpretation.append(f"- Hybrid search (BM25 + Vector): {avg_score:.1%} average performance")
            interpretation.append("- Combines keyword and semantic understanding")
            interpretation.append("- Generally more robust than individual methods")
            
        elif variant == "rrf_ce":
            interpretation.append(f"- Hybrid + Cross-encoder reranking: {avg_score:.1%} average performance")
            interpretation.append("- Most sophisticated approach with final relevance scoring")
            interpretation.append("- Best for: Maximum accuracy when latency permits")
        
        # Show metrics for this variant
        if scores:
            interpretation.append("- Metrics:")
            for metric, value in sorted(scores.items()):
                interpretation.append(f"  - {metric}: {value:.3f}")
    
    # Recommendations based on comparison
    interpretation.append("\n### Recommendations Based on Comparison:\n")
    
    # Check if cross-encoder helps
    if "rrf" in variants and "rrf_ce" in variants:
        rrf_avg = sum(variants["rrf"].values()) / len(variants["rrf"]) if variants["rrf"] else 0
        rrf_ce_avg = sum(variants["rrf_ce"].values()) / len(variants["rrf_ce"]) if variants["rrf_ce"] else 0
        improvement = ((rrf_ce_avg - rrf_avg) / rrf_avg * 100) if rrf_avg > 0 else 0
        
        if improvement > 10:
            interpretation.append(f"✅ Cross-encoder reranking provides {improvement:.1f}% improvement - recommended for production")
        elif improvement > 0:
            interpretation.append(f"ℹ️ Cross-encoder provides {improvement:.1f}% improvement - use based on latency requirements")
        else:
            interpretation.append("⚠️ Cross-encoder not improving results - review configuration or skip for better latency")
    
    # Check hybrid vs pure methods
    if all(v in variants for v in ["bm25", "vec", "rrf"]):
        bm25_avg = sum(variants["bm25"].values()) / len(variants["bm25"]) if variants["bm25"] else 0
        vec_avg = sum(variants["vec"].values()) / len(variants["vec"]) if variants["vec"] else 0
        rrf_avg = sum(variants["rrf"].values()) / len(variants["rrf"]) if variants["rrf"] else 0
        
        if rrf_avg > max(bm25_avg, vec_avg):
            improvement = ((rrf_avg - max(bm25_avg, vec_avg)) / max(bm25_avg, vec_avg) * 100)
            interpretation.append(f"✅ Hybrid search (RRF) outperforms pure methods by {improvement:.1f}% - recommended")
        else:
            interpretation.append("⚠️ Hybrid search not improving over pure methods - review fusion parameters")
    
    return "\n".join(interpretation)


def generate_search_interpretation(
    aggregates: Dict[str, Dict[str, float]],
    config: Dict
) -> str:
    """Generate detailed interpretation for search evaluation results."""
    
    interpretation = ["# Detailed Evaluation Interpretation\n"]
    interpretation.append("=" * 50 + "\n")
    
    # Overall summary
    interpretation.append("## Executive Summary\n")
    
    # Calculate overall system health
    all_scores = []
    for variant_scores in aggregates.values():
        all_scores.extend(variant_scores.values())
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        if avg_score >= 0.8:
            interpretation.append(f"✅ **System Health: EXCELLENT** (Average score: {avg_score:.1%})")
        elif avg_score >= 0.7:
            interpretation.append(f"✅ **System Health: GOOD** (Average score: {avg_score:.1%})")
        elif avg_score >= 0.6:
            interpretation.append(f"⚠️ **System Health: ACCEPTABLE** (Average score: {avg_score:.1%})")
        else:
            interpretation.append(f"❌ **System Health: NEEDS ATTENTION** (Average score: {avg_score:.1%})")
    
    interpretation.append(f"\nEvaluated {config.get('max_samples', 'N/A')} samples across {len(aggregates)} search variants.\n")
    
    # Variant comparison
    interpretation.append(interpret_search_variant_comparison(aggregates))
    
    # Individual metric analysis
    interpretation.append("\n## Detailed Metric Analysis\n")
    
    # Get all unique metrics
    all_metrics = set()
    for variant_scores in aggregates.values():
        all_metrics.update(variant_scores.keys())
    
    # Analyze each metric across variants
    for metric in sorted(all_metrics):
        interpretation.append(f"\n### {metric.replace('_', ' ').title()}\n")
        
        # Get scores for this metric across variants
        metric_scores = {}
        for variant, scores in aggregates.items():
            if metric in scores:
                metric_scores[variant] = scores[metric]
        
        if metric_scores:
            best_variant = max(metric_scores.items(), key=lambda x: x[1])
            worst_variant = min(metric_scores.items(), key=lambda x: x[1])
            avg_metric = sum(metric_scores.values()) / len(metric_scores)
            
            interpretation.append(f"- **Average**: {avg_metric:.3f}")
            interpretation.append(f"- **Best**: {best_variant[0]} ({best_variant[1]:.3f})")
            interpretation.append(f"- **Worst**: {worst_variant[0]} ({worst_variant[1]:.3f})")
            interpretation.append(f"- **Range**: {best_variant[1] - worst_variant[1]:.3f}")
            
            # Metric-specific interpretation
            if "relevance" in metric.lower():
                interp = interpret_context_relevance(avg_metric)
                interpretation.append(f"\n{interp.interpretation}")
            elif "utilization" in metric.lower():
                interp = interpret_context_utilization(avg_metric)
                interpretation.append(f"\n{interp.interpretation}")
    
    # Action items
    interpretation.append("\n## Recommended Actions\n")
    interpretation.append("Based on the evaluation results:\n")
    
    # Generate priority actions
    priority_actions = []
    
    # Check context relevance
    avg_relevance = 0
    relevance_count = 0
    for scores in aggregates.values():
        for metric, value in scores.items():
            if "relevance" in metric.lower():
                avg_relevance += value
                relevance_count += 1
    
    if relevance_count > 0:
        avg_relevance /= relevance_count
        if avg_relevance < 0.6:
            priority_actions.append("🔴 **URGENT**: Improve retrieval relevance (currently {:.1%})".format(avg_relevance))
        elif avg_relevance < 0.75:
            priority_actions.append("🟡 **MEDIUM**: Enhance retrieval quality (currently {:.1%})".format(avg_relevance))
    
    # Check if cross-encoder is worth the latency
    if "rrf_ce" in aggregates and "rrf" in aggregates:
        rrf_scores = list(aggregates["rrf"].values())
        rrf_ce_scores = list(aggregates["rrf_ce"].values())
        if rrf_scores and rrf_ce_scores:
            improvement = (sum(rrf_ce_scores) - sum(rrf_scores)) / sum(rrf_scores) * 100
            if improvement < 5:
                priority_actions.append("🟡 Consider removing cross-encoder for better latency (only {:.1f}% improvement)".format(improvement))
    
    if priority_actions:
        for action in priority_actions:
            interpretation.append(f"- {action}")
    else:
        interpretation.append("- ✅ System performing well, maintain current configuration")
    
    # Configuration notes
    interpretation.append(f"\n## Configuration Used\n")
    interpretation.append(f"- Dataset: {config.get('dataset', 'N/A')}")
    interpretation.append(f"- Samples: {config.get('max_samples', 'N/A')}")
    interpretation.append(f"- Top-K: {config.get('top_k', 'N/A')}")
    interpretation.append(f"- RRF-K: {config.get('rrf_k', 'N/A')}")
    interpretation.append(f"- Rerank Top-K: {config.get('rerank_top_k', 'N/A')}")
    
    return "\n".join(interpretation)


def generate_chat_interpretation(
    scores: Dict[str, float],
    config: Dict
) -> str:
    """Generate detailed interpretation for chat evaluation results."""
    
    interpretation = ["# Detailed Chat Evaluation Interpretation\n"]
    interpretation.append("=" * 50 + "\n")
    
    # Overall summary
    interpretation.append("## Executive Summary\n")
    
    if scores:
        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 0.85:
            interpretation.append(f"✅ **Chat Quality: EXCELLENT** (Average score: {avg_score:.1%})")
            interpretation.append("The chat system is performing exceptionally well across all metrics.")
        elif avg_score >= 0.75:
            interpretation.append(f"✅ **Chat Quality: GOOD** (Average score: {avg_score:.1%})")
            interpretation.append("The chat system is performing well with room for minor improvements.")
        elif avg_score >= 0.65:
            interpretation.append(f"⚠️ **Chat Quality: ACCEPTABLE** (Average score: {avg_score:.1%})")
            interpretation.append("The chat system meets minimum requirements but needs optimization.")
        else:
            interpretation.append(f"❌ **Chat Quality: NEEDS ATTENTION** (Average score: {avg_score:.1%})")
            interpretation.append("The chat system requires immediate improvements to meet quality standards.")
    
    interpretation.append(f"\nEvaluated {config.get('max_samples', 'N/A')} chat interactions.\n")
    
    # Individual metric analysis
    interpretation.append("\n## Metric-by-Metric Analysis\n")
    
    # Interpret each metric
    metric_interpretations = []
    
    for metric_name, value in sorted(scores.items()):
        interpretation.append(f"\n### {metric_name.replace('_', ' ').title()}\n")
        interpretation.append(f"**Score**: {value:.3f}\n")
        
        # Get specific interpretation
        if "faithfulness" in metric_name.lower():
            interp = interpret_faithfulness(value)
        elif "relevancy" in metric_name.lower() or "relevance" in metric_name.lower():
            interp = interpret_answer_relevancy(value)
        elif "utilization" in metric_name.lower():
            interp = interpret_context_utilization(value)
        else:
            # Generic interpretation
            if value >= 0.8:
                level = PerformanceLevel.GOOD
                text = f"Strong performance ({value:.1%})"
            elif value >= 0.6:
                level = PerformanceLevel.ACCEPTABLE
                text = f"Acceptable performance ({value:.1%})"
            else:
                level = PerformanceLevel.NEEDS_IMPROVEMENT
                text = f"Below target performance ({value:.1%})"
            
            interp = MetricInterpretation(
                metric_name=metric_name,
                value=value,
                level=level,
                interpretation=text,
                recommendations=[]
            )
        
        metric_interpretations.append(interp)
        interpretation.append(f"{interp.interpretation}\n")
        
        if interp.recommendations:
            interpretation.append("**Recommendations**:")
            for rec in interp.recommendations:
                interpretation.append(f"- {rec}")
    
    # Critical issues identification
    interpretation.append("\n## Critical Issues & Priorities\n")
    
    critical_issues = []
    medium_issues = []
    
    for interp in metric_interpretations:
        if interp.level == PerformanceLevel.POOR:
            critical_issues.append(f"🔴 **CRITICAL**: {interp.metric_name} at {interp.value:.1%}")
        elif interp.level == PerformanceLevel.NEEDS_IMPROVEMENT:
            medium_issues.append(f"🟡 **MEDIUM**: {interp.metric_name} at {interp.value:.1%}")
    
    if critical_issues:
        interpretation.append("### Critical Issues (Immediate Action Required):")
        for issue in critical_issues:
            interpretation.append(f"- {issue}")
    
    if medium_issues:
        interpretation.append("\n### Medium Priority Issues:")
        for issue in medium_issues:
            interpretation.append(f"- {issue}")
    
    if not critical_issues and not medium_issues:
        interpretation.append("✅ No critical or medium priority issues detected.")
    
    # Overall recommendations
    interpretation.append("\n## Strategic Recommendations\n")
    
    # Prioritized action plan
    action_plan = []
    
    # Check for hallucination issues
    if "faithfulness" in scores and scores["faithfulness"] < 0.7:
        action_plan.append("1. **Reduce Hallucination** (Priority: CRITICAL)")
        action_plan.append("   - Set temperature to 0 for deterministic outputs")
        action_plan.append("   - Strengthen grounding instructions in prompts")
        action_plan.append("   - Implement fact-checking validation")
    
    # Check for relevancy issues
    relevancy_metrics = [k for k in scores.keys() if "relevancy" in k.lower() or "relevance" in k.lower()]
    if relevancy_metrics:
        avg_relevancy = sum(scores[m] for m in relevancy_metrics) / len(relevancy_metrics)
        if avg_relevancy < 0.7:
            action_plan.append(f"{len(action_plan)+1}. **Improve Answer Relevancy** (Priority: HIGH)")
            action_plan.append("   - Review and optimize prompt templates")
            action_plan.append("   - Implement query intent classification")
            action_plan.append("   - Add explicit instructions to address all query aspects")
    
    # Check for context issues
    context_metrics = [k for k in scores.keys() if "context" in k.lower()]
    if context_metrics:
        avg_context = sum(scores[m] for m in context_metrics) / len(context_metrics)
        if avg_context < 0.6:
            action_plan.append(f"{len(action_plan)+1}. **Enhance Context Quality** (Priority: MEDIUM)")
            action_plan.append("   - Increase retrieval top-k parameter")
            action_plan.append("   - Improve embedding model or search strategy")
            action_plan.append("   - Implement query expansion techniques")
    
    if action_plan:
        interpretation.append("### Prioritized Action Plan:")
        for action in action_plan:
            interpretation.append(action)
    else:
        interpretation.append("The system is performing well. Focus on:")
        interpretation.append("- Monitoring for regression")
        interpretation.append("- Incremental optimizations")
        interpretation.append("- Expanding test coverage")
    
    # Configuration details
    interpretation.append(f"\n## Evaluation Configuration\n")
    interpretation.append(f"- Dataset: {config.get('dataset', 'N/A')}")
    interpretation.append(f"- Samples Evaluated: {config.get('max_samples', 'N/A')}")
    interpretation.append(f"- Context Top-K: {config.get('top_k', 'N/A')}")
    interpretation.append(f"- Model: {config.get('model', 'gpt-5-mini')}")
    
    # Success criteria
    interpretation.append("\n## Success Criteria Assessment\n")
    
    passed = []
    failed = []
    
    # Define success criteria
    criteria = {
        "Faithfulness > 70%": scores.get("faithfulness", 0) > 0.7,
        "Answer Relevancy > 70%": any(scores.get(k, 0) > 0.7 for k in scores if "relevancy" in k.lower()),
        "Context Precision > 60%": scores.get("context_precision", 1) > 0.6,
        "Overall Average > 65%": (sum(scores.values()) / len(scores) if scores else 0) > 0.65
    }
    
    for criterion, met in criteria.items():
        if met:
            passed.append(f"✅ {criterion}")
        else:
            failed.append(f"❌ {criterion}")
    
    if passed:
        interpretation.append("**Passed Criteria**:")
        for p in passed:
            interpretation.append(f"- {p}")
    
    if failed:
        interpretation.append("\n**Failed Criteria**:")
        for f in failed:
            interpretation.append(f"- {f}")
    
    # Final verdict
    interpretation.append("\n## Final Verdict\n")
    
    if len(passed) >= len(failed):
        interpretation.append("✅ **PASS**: The chat system meets most quality criteria.")
        if failed:
            interpretation.append(f"Note: {len(failed)} criteria need attention for optimal performance.")
    else:
        interpretation.append("❌ **NEEDS IMPROVEMENT**: The chat system fails to meet several quality criteria.")
        interpretation.append(f"Action required on {len(failed)} critical criteria.")
    
    return "\n".join(interpretation)