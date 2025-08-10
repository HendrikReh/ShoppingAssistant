"""Continuous evaluation and monitoring for RAG retrieval quality."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
import schedule
import threading

app = typer.Typer()
console = Console()


class EvaluationMetrics:
    """Track evaluation metrics over time."""
    
    def __init__(self, history_file: Path = Path("eval_history.json")):
        self.history_file = history_file
        self.current_metrics = {}
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load historical metrics."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_history(self):
        """Save metrics history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_evaluation(self, metrics: Dict):
        """Add new evaluation results."""
        metrics["timestamp"] = datetime.now().isoformat()
        self.history.append(metrics)
        self.current_metrics = metrics
        self.save_history()
        
        # Keep only last 100 evaluations
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_trend(self, metric_name: str, window: int = 10) -> str:
        """Get trend for a specific metric."""
        if len(self.history) < 2:
            return "→"
        
        recent = [h.get(metric_name, 0) for h in self.history[-window:]]
        if len(recent) < 2:
            return "→"
        
        # Calculate trend
        avg_recent = np.mean(recent[-5:])
        avg_previous = np.mean(recent[:-5]) if len(recent) > 5 else recent[0]
        
        if avg_recent > avg_previous * 1.05:
            return "↑"
        elif avg_recent < avg_previous * 0.95:
            return "↓"
        else:
            return "→"
    
    def get_statistics(self) -> Dict:
        """Get statistical summary of metrics."""
        if not self.history:
            return {}
        
        metrics_df = pd.DataFrame(self.history)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                stats[col] = {
                    "current": values.iloc[-1] if len(values) > 0 else 0,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "trend": self.get_trend(col)
                }
        
        return stats


class ContinuousEvaluator:
    """Continuous evaluation of retrieval quality."""
    
    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer,
        collection_name: str,
        test_queries_file: Optional[Path] = None
    ):
        self.client = client
        self.model = model
        self.collection_name = collection_name
        self.metrics = EvaluationMetrics()
        self.test_queries = self._load_test_queries(test_queries_file)
        
    def _load_test_queries(self, file_path: Optional[Path]) -> List[Dict]:
        """Load test queries for evaluation."""
        if file_path and file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        
        # Default test queries
        return [
            {"query": "Fire TV Stick", "expected_terms": ["fire", "tv", "stick"], "category": "streaming"},
            {"query": "wireless earbuds", "expected_terms": ["earbud", "wireless", "bluetooth"], "category": "audio"},
            {"query": "USB cable", "expected_terms": ["usb", "cable", "cord"], "category": "accessories"},
            {"query": "laptop", "expected_terms": ["laptop", "notebook", "computer"], "category": "electronics"},
            {"query": "Echo Dot", "expected_terms": ["echo", "dot", "alexa"], "category": "smart_home"},
            {"query": "gaming keyboard", "expected_terms": ["keyboard", "gaming", "mechanical"], "category": "gaming"},
            {"query": "tablet", "expected_terms": ["tablet", "ipad", "tab"], "category": "electronics"},
            {"query": "router", "expected_terms": ["router", "wifi", "wireless"], "category": "networking"},
            {"query": "monitor", "expected_terms": ["monitor", "display", "screen"], "category": "electronics"},
            {"query": "headphones", "expected_terms": ["headphone", "headset", "audio"], "category": "audio"}
        ]
    
    def evaluate_single_query(self, query_info: Dict) -> Dict:
        """Evaluate a single query."""
        query = query_info["query"]
        expected_terms = query_info.get("expected_terms", [])
        
        # Perform search
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec.tolist(),
            limit=10,
            with_payload=True
        )
        
        # Calculate metrics
        metrics = {
            "query": query,
            "num_results": len(results),
            "avg_score": np.mean([r.score for r in results]) if results else 0,
            "max_score": max([r.score for r in results]) if results else 0,
            "min_score": min([r.score for r in results]) if results else 0,
        }
        
        # Check relevance
        relevant_count = 0
        for result in results[:5]:  # Check top 5
            title = result.payload.get('title', '').lower()
            if any(term.lower() in title for term in expected_terms):
                relevant_count += 1
        
        metrics["relevance_rate"] = relevant_count / 5 if results else 0
        
        # Check diversity (unique product types in results)
        unique_products = len(set(r.payload.get('title', '')[:20] for r in results[:5]))
        metrics["diversity"] = unique_products / 5 if results else 0
        
        return metrics
    
    def run_evaluation(self) -> Dict:
        """Run full evaluation suite."""
        console.print("[yellow]Running evaluation suite...[/yellow]")
        
        all_metrics = []
        for query_info in self.test_queries:
            metrics = self.evaluate_single_query(query_info)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(all_metrics),
            "avg_relevance": np.mean([m["relevance_rate"] for m in all_metrics]),
            "avg_score": np.mean([m["avg_score"] for m in all_metrics]),
            "avg_diversity": np.mean([m["diversity"] for m in all_metrics]),
            "min_relevance": min([m["relevance_rate"] for m in all_metrics]),
            "max_relevance": max([m["relevance_rate"] for m in all_metrics]),
            "queries_with_perfect_relevance": sum(1 for m in all_metrics if m["relevance_rate"] == 1.0),
            "queries_with_poor_relevance": sum(1 for m in all_metrics if m["relevance_rate"] < 0.4),
        }
        
        # Calculate health score
        health_score = (
            aggregated["avg_relevance"] * 0.5 +
            aggregated["avg_diversity"] * 0.2 +
            (aggregated["avg_score"] if aggregated["avg_score"] > 0 else 0) * 0.3
        )
        aggregated["health_score"] = min(health_score, 1.0)
        
        # Save metrics
        self.metrics.add_evaluation(aggregated)
        
        return aggregated
    
    def generate_dashboard(self) -> Layout:
        """Generate dashboard layout."""
        layout = Layout()
        
        # Get current metrics and statistics
        current = self.metrics.current_metrics
        stats = self.metrics.get_statistics()
        
        # Create header
        header = Panel(
            f"[bold cyan]RAG Retrieval Quality Monitor[/bold cyan]\n"
            f"Last Update: {current.get('timestamp', 'Never')[:19]}",
            style="bold blue"
        )
        
        # Create metrics table
        metrics_table = Table(title="Current Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        metrics_table.add_column("Trend", style="white")
        metrics_table.add_column("Target", style="green")
        metrics_table.add_column("Status", style="bold")
        
        # Define targets
        targets = {
            "avg_relevance": 0.8,
            "avg_diversity": 0.7,
            "health_score": 0.75,
            "queries_with_poor_relevance": 2  # max allowed
        }
        
        for metric, target in targets.items():
            if metric in current:
                value = current[metric]
                trend = stats.get(metric, {}).get("trend", "→") if stats else "→"
                
                # Determine status
                if metric == "queries_with_poor_relevance":
                    status = "✅" if value <= target else "⚠️" if value <= target * 1.5 else "❌"
                else:
                    status = "✅" if value >= target else "⚠️" if value >= target * 0.8 else "❌"
                
                metrics_table.add_row(
                    metric.replace("_", " ").title(),
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    trend,
                    f"{target:.2f}" if isinstance(target, float) else str(target),
                    status
                )
        
        # Create alerts panel
        alerts = []
        if current.get("avg_relevance", 0) < 0.6:
            alerts.append("🔴 Critical: Average relevance below 60%")
        if current.get("queries_with_poor_relevance", 0) > 3:
            alerts.append("🟡 Warning: Multiple queries with poor relevance")
        if current.get("health_score", 0) < 0.5:
            alerts.append("🔴 Critical: Overall health score below 50%")
        
        if not alerts:
            alerts.append("🟢 All systems operational")
        
        alerts_panel = Panel(
            "\n".join(alerts),
            title="System Alerts",
            style="red" if any("🔴" in a for a in alerts) else "yellow" if any("🟡" in a for a in alerts) else "green"
        )
        
        # Create statistics panel
        if stats:
            stats_text = []
            for metric in ["avg_relevance", "health_score"]:
                if metric in stats:
                    s = stats[metric]
                    stats_text.append(
                        f"{metric.replace('_', ' ').title()}:\n"
                        f"  Mean: {s['mean']:.3f} ± {s['std']:.3f}\n"
                        f"  Range: [{s['min']:.3f}, {s['max']:.3f}]"
                    )
            
            stats_panel = Panel(
                "\n\n".join(stats_text),
                title="Historical Statistics",
                style="blue"
            )
        else:
            stats_panel = Panel("No historical data available", title="Historical Statistics")
        
        # Arrange layout
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main"),
            Layout(alerts_panel, size=5)
        )
        
        layout["main"].split_row(
            Layout(metrics_table),
            Layout(stats_panel)
        )
        
        return layout


class MonitoringService:
    """Background monitoring service."""
    
    def __init__(self, evaluator: ContinuousEvaluator, interval_minutes: int = 30):
        self.evaluator = evaluator
        self.interval_minutes = interval_minutes
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring service."""
        self.running = True
        self.thread = threading.Thread(target=self._run_monitor)
        self.thread.daemon = True
        self.thread.start()
        console.print(f"[green]Monitoring started (interval: {self.interval_minutes} minutes)[/green]")
    
    def stop(self):
        """Stop monitoring service."""
        self.running = False
        if self.thread:
            self.thread.join()
        console.print("[red]Monitoring stopped[/red]")
    
    def _run_monitor(self):
        """Run monitoring loop."""
        # Schedule evaluations
        schedule.every(self.interval_minutes).minutes.do(self.evaluator.run_evaluation)
        
        # Run initial evaluation
        self.evaluator.run_evaluation()
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


@app.command()
def monitor(
    collection: str = typer.Option("products_minilm", help="Collection name"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Model name"),
    interval: int = typer.Option(30, help="Evaluation interval in minutes"),
    dashboard: bool = typer.Option(True, help="Show live dashboard")
):
    """Start continuous monitoring."""
    
    # Initialize
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)
    evaluator = ContinuousEvaluator(client, model, collection)
    
    if dashboard:
        # Show live dashboard
        with Live(evaluator.generate_dashboard(), refresh_per_second=0.5, console=console) as live:
            service = MonitoringService(evaluator, interval)
            service.start()
            
            try:
                while True:
                    time.sleep(5)
                    live.update(evaluator.generate_dashboard())
            except KeyboardInterrupt:
                service.stop()
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    else:
        # Run in background
        service = MonitoringService(evaluator, interval)
        service.start()
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            service.stop()


@app.command()
def evaluate_once(
    collection: str = typer.Option("products_minilm", help="Collection name"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Model name"),
    save: bool = typer.Option(True, help="Save results to history")
):
    """Run evaluation once."""
    
    # Initialize
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(model_name)
    evaluator = ContinuousEvaluator(client, model, collection)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Display results
    console.print("\n[bold cyan]Evaluation Results[/bold cyan]\n")
    
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in results.items():
        if key != "timestamp":
            if isinstance(value, float):
                table.add_row(key.replace("_", " ").title(), f"{value:.3f}")
            else:
                table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)
    
    # Health assessment
    health = results.get("health_score", 0)
    if health > 0.8:
        console.print("\n✅ [green]System health: EXCELLENT[/green]")
    elif health > 0.6:
        console.print("\n✅ [yellow]System health: GOOD[/yellow]")
    elif health > 0.4:
        console.print("\n⚠️ [orange]System health: NEEDS ATTENTION[/orange]")
    else:
        console.print("\n❌ [red]System health: CRITICAL[/red]")
    
    if save:
        console.print("\n💾 Results saved to eval_history.json")


@app.command()
def show_history(
    last_n: int = typer.Option(10, help="Number of recent evaluations to show")
):
    """Show evaluation history."""
    
    metrics = EvaluationMetrics()
    
    if not metrics.history:
        console.print("[yellow]No evaluation history found[/yellow]")
        return
    
    # Get recent history
    recent = metrics.history[-last_n:]
    
    # Create table
    table = Table(title=f"Last {len(recent)} Evaluations")
    table.add_column("Time", style="cyan")
    table.add_column("Health", style="bold")
    table.add_column("Relevance", style="yellow")
    table.add_column("Diversity", style="blue")
    table.add_column("Poor Queries", style="red")
    
    for entry in recent:
        timestamp = entry.get("timestamp", "")[:19]
        health = entry.get("health_score", 0)
        relevance = entry.get("avg_relevance", 0)
        diversity = entry.get("avg_diversity", 0)
        poor = entry.get("queries_with_poor_relevance", 0)
        
        health_color = "green" if health > 0.7 else "yellow" if health > 0.5 else "red"
        
        table.add_row(
            timestamp,
            f"[{health_color}]{health:.2f}[/{health_color}]",
            f"{relevance:.2f}",
            f"{diversity:.2f}",
            str(poor)
        )
    
    console.print(table)
    
    # Show trends
    stats = metrics.get_statistics()
    if stats:
        console.print("\n[bold]Trends:[/bold]")
        for metric in ["health_score", "avg_relevance"]:
            if metric in stats:
                trend = stats[metric]["trend"]
                trend_emoji = "📈" if trend == "↑" else "📉" if trend == "↓" else "➡️"
                console.print(f"  {trend_emoji} {metric.replace('_', ' ').title()}: {trend}")


if __name__ == "__main__":
    app()