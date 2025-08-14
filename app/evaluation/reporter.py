"""Evaluation report generation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in various formats."""
    
    def __init__(self, output_dir: Path = Path("eval/results")):
        """Initialize reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        report_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        formats: List[str] = ["json", "markdown"]
    ) -> Dict[str, Path]:
        """Generate evaluation report in multiple formats.
        
        Args:
            results: Evaluation results
            report_name: Base name for report files
            metadata: Additional metadata
            formats: List of formats to generate
            
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = get_timestamp()
        base_name = f"{report_name}_{timestamp}"
        
        # Add metadata
        if metadata:
            results["metadata"] = metadata
        
        results["timestamp"] = timestamp
        results["report_name"] = report_name
        
        output_files = {}
        
        if "json" in formats:
            json_path = self.save_json_report(results, base_name)
            output_files["json"] = json_path
        
        if "markdown" in formats:
            md_path = self.save_markdown_report(results, base_name)
            output_files["markdown"] = md_path
        
        if "html" in formats:
            html_path = self.save_html_report(results, base_name)
            output_files["html"] = html_path
        
        return output_files
    
    def save_json_report(self, results: Dict[str, Any], base_name: str) -> Path:
        """Save JSON report.
        
        Args:
            results: Results to save
            base_name: Base filename
            
        Returns:
            Path to saved file
        """
        path = self.output_dir / f"{base_name}.json"
        with path.open("w") as fp:
            json.dump(results, fp, indent=2, default=str)
        logger.info(f"Saved JSON report: {path}")
        return path
    
    def save_markdown_report(self, results: Dict[str, Any], base_name: str) -> Path:
        """Save Markdown report.
        
        Args:
            results: Results to format
            base_name: Base filename
            
        Returns:
            Path to saved file
        """
        path = self.output_dir / f"{base_name}.md"
        
        md_content = self.format_markdown_report(results)
        
        with path.open("w") as fp:
            fp.write(md_content)
        
        logger.info(f"Saved Markdown report: {path}")
        return path
    
    def format_markdown_report(self, results: Dict[str, Any]) -> str:
        """Format results as Markdown.
        
        Args:
            results: Results to format
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report: {results.get('report_name', 'Unknown')}")
        lines.append(f"\n**Generated:** {results.get('timestamp', 'N/A')}\n")
        
        # Metadata
        if "metadata" in results:
            lines.append("## Configuration")
            lines.append("```json")
            lines.append(json.dumps(results["metadata"], indent=2))
            lines.append("```\n")
        
        # Metrics
        lines.append("## Metrics Summary")
        
        # Check if this is a multi-variant result
        if any(isinstance(v, dict) and "variant" in v for v in results.values()):
            # Search evaluation with variants
            lines.extend(self._format_variant_metrics(results))
        else:
            # Single evaluation
            lines.extend(self._format_single_metrics(results))
        
        # Sample results
        if "detailed_results" in results:
            lines.append("\n## Sample Results")
            lines.extend(self._format_sample_results(results["detailed_results"][:3]))
        
        return "\n".join(lines)
    
    def _format_variant_metrics(self, results: Dict) -> List[str]:
        """Format metrics for multi-variant evaluation."""
        lines = []
        
        # Create comparison table
        headers = ["Variant", "Success Rate", "Avg Results", "Avg Time"]
        rows = []
        
        for key, value in results.items():
            if isinstance(value, dict) and "variant" in value:
                variant = value["variant"]
                success_rate = value.get("success_rate", 0)
                avg_results = value.get("avg_contexts_retrieved", 0)
                avg_time = value.get("avg_search_time", 0)
                
                rows.append([
                    variant,
                    f"{success_rate:.2%}",
                    f"{avg_results:.1f}",
                    f"{avg_time:.3f}s"
                ])
        
        if rows:
            lines.append(write_markdown_table(headers, rows))
        
        return lines
    
    def _format_single_metrics(self, results: Dict) -> List[str]:
        """Format metrics for single evaluation."""
        lines = []
        
        metrics_to_show = [
            ("num_queries", "Total Queries"),
            ("num_questions", "Total Questions"),
            ("success_rate", "Success Rate"),
            ("avg_contexts_retrieved", "Avg Contexts"),
            ("avg_generation_time", "Avg Generation Time"),
            ("avg_retrieval_time", "Avg Retrieval Time"),
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall")
        ]
        
        for key, label in metrics_to_show:
            if key in results:
                value = results[key]
                if isinstance(value, float):
                    if "rate" in key or key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        lines.append(f"- **{label}:** {value:.2%}")
                    elif "time" in key:
                        lines.append(f"- **{label}:** {value:.3f}s")
                    else:
                        lines.append(f"- **{label}:** {value:.2f}")
                else:
                    lines.append(f"- **{label}:** {value}")
        
        return lines
    
    def _format_sample_results(self, samples: List[Dict]) -> List[str]:
        """Format sample results."""
        lines = []
        
        for i, sample in enumerate(samples, 1):
            lines.append(f"\n### Sample {i}")
            
            if "query" in sample:
                lines.append(f"**Query:** {sample['query']}")
            elif "question" in sample:
                lines.append(f"**Question:** {sample['question']}")
            
            if "answer" in sample:
                lines.append(f"**Answer:** {sample['answer'][:200]}...")
            
            if "contexts_retrieved" in sample:
                lines.append(f"**Contexts Retrieved:** {sample.get('num_results', len(sample['contexts_retrieved']))}")
            
            if "ragas_scores" in sample and sample["ragas_scores"]:
                lines.append("**RAGAS Scores:**")
                for metric, score in sample["ragas_scores"].items():
                    lines.append(f"  - {metric}: {score:.3f}")
        
        return lines
    
    def save_html_report(self, results: Dict[str, Any], base_name: str) -> Path:
        """Save HTML report.
        
        Args:
            results: Results to format
            base_name: Base filename
            
        Returns:
            Path to saved file
        """
        path = self.output_dir / f"{base_name}.html"
        
        # Convert Markdown to HTML
        md_content = self.format_markdown_report(results)
        html_content = markdown_to_html(md_content, results.get("report_name", "Report"))
        
        with path.open("w") as fp:
            fp.write(html_content)
        
        logger.info(f"Saved HTML report: {path}")
        return path


def generate_evaluation_report(
    results: Dict[str, Any],
    report_name: str,
    output_dir: Path = Path("eval/results"),
    metadata: Optional[Dict[str, Any]] = None,
    formats: List[str] = ["json", "markdown"]
) -> Dict[str, Path]:
    """Generate evaluation report.
    
    Args:
        results: Evaluation results
        report_name: Report name
        output_dir: Output directory
        metadata: Additional metadata
        formats: Output formats
        
    Returns:
        Dictionary of output paths
    """
    reporter = EvaluationReporter(output_dir)
    return reporter.generate_report(results, report_name, metadata, formats)


def get_timestamp() -> str:
    """Get timestamp string for filenames.
    
    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create a Markdown table.
    
    Args:
        headers: Table headers
        rows: Table rows
        
    Returns:
        Markdown table string
    """
    lines = []
    
    # Headers
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|")
    
    # Rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join(lines)


def markdown_to_html(markdown_content: str, title: str = "Report") -> str:
    """Convert Markdown to basic HTML.
    
    Args:
        markdown_content: Markdown content
        title: HTML title
        
    Returns:
        HTML string
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        h3 {{ color: #999; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="content">
        {markdown_content.replace('#', 'h1').replace('**', 'strong').replace('`', 'code')}
    </div>
</body>
</html>"""
    return html