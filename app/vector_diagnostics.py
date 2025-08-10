"""Comprehensive vector search diagnostics tool.

Implements all debugging checklist items to automatically diagnose and report
issues with vector search and embeddings.
"""

import json
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from collections import defaultdict
import typer

app = typer.Typer()
console = Console()

class VectorDiagnostics:
    """Comprehensive diagnostics for vector search quality."""
    
    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer,
        collection_name: str,
        data_path: Optional[Path] = None
    ):
        self.client = client
        self.model = model
        self.collection_name = collection_name
        self.data_path = data_path
        self.results = {}
        
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests and return results."""
        console.print("\n[bold cyan]🔍 Running Vector Search Diagnostics[/bold cyan]\n")
        
        # 1. Check collection configuration
        self.check_collection_config()
        
        # 2. Check embedding similarity ranges
        self.check_similarity_ranges()
        
        # 3. Test exact product matches
        self.test_exact_matches()
        
        # 4. Verify model consistency
        self.verify_model_consistency()
        
        # 5. Test category discrimination
        self.test_category_discrimination()
        
        # 6. Check embedding normalization
        self.check_normalization()
        
        # 7. Verify vector dimensions
        self.verify_dimensions()
        
        # 8. Test query performance
        self.test_query_performance()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def check_collection_config(self):
        """Check Qdrant collection configuration."""
        console.print("[yellow]1. Checking collection configuration...[/yellow]")
        
        try:
            info = self.client.get_collection(self.collection_name)
            config = {
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else "unknown",
                "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params, 'vectors') else "unknown"
            }
            
            # Checks
            issues = []
            if info.points_count == 0:
                issues.append("❌ Collection is empty")
            if info.status != "green":
                issues.append(f"⚠️ Collection status is {info.status}")
            if "cosine" not in str(config["distance"]).lower():
                issues.append("⚠️ Not using cosine distance (recommended for normalized embeddings)")
                
            self.results["collection_config"] = {
                "status": "pass" if not issues else "warning",
                "config": config,
                "issues": issues
            }
            
            if issues:
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print(f"  ✅ Collection healthy: {info.points_count} points")
                
        except Exception as e:
            self.results["collection_config"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def check_similarity_ranges(self, sample_size: int = 100):
        """Check if embedding similarities are in a healthy range."""
        console.print("[yellow]2. Checking embedding similarity ranges...[/yellow]")
        
        try:
            # Sample random points
            sample = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(sample_size, 100),
                with_vectors=True,
                with_payload=True
            )[0]
            
            if len(sample) < 2:
                self.results["similarity_ranges"] = {
                    "status": "skip",
                    "reason": "Not enough points to compare"
                }
                console.print("  ⏭️ Skipped: Not enough points")
                return
            
            # Calculate pairwise similarities
            vectors = [np.array(p.vector) for p in sample]
            titles = [p.payload.get('title', 'Unknown')[:30] for p in sample]
            
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = np.dot(vectors[i], vectors[j])
                    similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # Statistics
            stats = {
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "range": float(np.max(similarities) - np.min(similarities))
            }
            
            # Check for issues
            issues = []
            if stats["range"] < 0.3:
                issues.append("❌ Very narrow similarity range - poor discrimination")
            elif stats["range"] < 0.5:
                issues.append("⚠️ Narrow similarity range - moderate discrimination")
            
            if stats["mean"] > 0.7:
                issues.append("⚠️ High average similarity - vectors may be too similar")
            
            self.results["similarity_ranges"] = {
                "status": "fail" if any("❌" in i for i in issues) else "warning" if issues else "pass",
                "stats": stats,
                "issues": issues,
                "sample_size": len(sample)
            }
            
            # Display results
            console.print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f} (spread: {stats['range']:.3f})")
            console.print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            
            if issues:
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print("  ✅ Healthy similarity distribution")
                
        except Exception as e:
            self.results["similarity_ranges"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def test_exact_matches(self, num_tests: int = 5):
        """Test if exact product matches return themselves first."""
        console.print("[yellow]3. Testing exact product matches...[/yellow]")
        
        try:
            # Sample some products
            sample = self.client.scroll(
                collection_name=self.collection_name,
                limit=num_tests,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not sample:
                self.results["exact_matches"] = {
                    "status": "skip",
                    "reason": "No products to test"
                }
                console.print("  ⏭️ Skipped: No products")
                return
            
            test_results = []
            failures = []
            
            for point in sample:
                title = point.payload.get('title', '')
                if not title:
                    continue
                    
                # Search for exact title
                query_vec = self.model.encode([title], normalize_embeddings=True)[0]
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec.tolist(),
                    limit=1,
                    with_payload=True
                )
                
                if results:
                    top_title = results[0].payload.get('title', '')
                    score = results[0].score
                    is_exact = top_title == title
                    
                    test_results.append({
                        "query": title[:50],
                        "matched": is_exact,
                        "score": score
                    })
                    
                    if not is_exact:
                        failures.append(f"'{title[:30]}...' returned '{top_title[:30]}...' (score: {score:.3f})")
                    elif score < 0.95:
                        failures.append(f"'{title[:30]}...' low self-similarity: {score:.3f}")
            
            # Calculate pass rate
            pass_rate = sum(1 for t in test_results if t["matched"]) / len(test_results) if test_results else 0
            
            self.results["exact_matches"] = {
                "status": "pass" if pass_rate == 1.0 else "warning" if pass_rate > 0.7 else "fail",
                "pass_rate": pass_rate,
                "test_count": len(test_results),
                "failures": failures
            }
            
            console.print(f"  Pass rate: {pass_rate:.1%} ({len(test_results)} tests)")
            if failures:
                console.print("  ❌ Failed matches:")
                for f in failures[:3]:  # Show first 3 failures
                    console.print(f"    - {f}")
            else:
                console.print("  ✅ All exact matches successful")
                
        except Exception as e:
            self.results["exact_matches"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def verify_model_consistency(self):
        """Verify the model used for search matches ingestion."""
        console.print("[yellow]4. Verifying model consistency...[/yellow]")
        
        try:
            # Get model info
            model_info = {
                "name": self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else "unknown",
                "embedding_dim": self.model.get_sentence_embedding_dimension()
            }
            
            # Get collection vector size
            info = self.client.get_collection(self.collection_name)
            collection_dim = info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None
            
            # Check consistency
            issues = []
            if collection_dim and collection_dim != model_info["embedding_dim"]:
                issues.append(f"❌ Dimension mismatch: model={model_info['embedding_dim']}, collection={collection_dim}")
            
            # Test with a sample embedding
            test_text = "test product"
            test_vec = self.model.encode([test_text], normalize_embeddings=True)[0]
            
            if len(test_vec) != collection_dim:
                issues.append(f"❌ Generated vector size {len(test_vec)} doesn't match collection {collection_dim}")
            
            self.results["model_consistency"] = {
                "status": "fail" if issues else "pass",
                "model_info": model_info,
                "collection_dim": collection_dim,
                "issues": issues
            }
            
            console.print(f"  Model: {model_info['name']}")
            console.print(f"  Dimensions: model={model_info['embedding_dim']}, collection={collection_dim}")
            
            if issues:
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print("  ✅ Model and collection are consistent")
                
        except Exception as e:
            self.results["model_consistency"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def test_category_discrimination(self):
        """Test if the model can discriminate between different product categories."""
        console.print("[yellow]5. Testing category discrimination...[/yellow]")
        
        # Define test categories
        test_queries = {
            "electronics": ["laptop", "smartphone", "tablet", "computer"],
            "audio": ["headphones", "earbuds", "speaker", "microphone"],
            "accessories": ["cable", "charger", "case", "adapter"],
            "streaming": ["Fire TV", "Roku", "Chromecast", "streaming device"]
        }
        
        try:
            discrimination_scores = {}
            
            for category, queries in test_queries.items():
                category_scores = []
                
                for query in queries:
                    query_vec = self.model.encode([query], normalize_embeddings=True)[0]
                    results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vec.tolist(),
                        limit=5,
                        with_payload=True
                    )
                    
                    if results:
                        # Check if results are relevant to the query
                        relevant = 0
                        for hit in results:
                            title = hit.payload.get('title', '').lower()
                            # Simple relevance check
                            if any(term in title for term in query.lower().split()):
                                relevant += 1
                        
                        relevance_rate = relevant / len(results)
                        category_scores.append(relevance_rate)
                
                if category_scores:
                    discrimination_scores[category] = np.mean(category_scores)
            
            # Overall discrimination score
            overall_score = np.mean(list(discrimination_scores.values())) if discrimination_scores else 0
            
            issues = []
            if overall_score < 0.3:
                issues.append("❌ Very poor category discrimination")
            elif overall_score < 0.5:
                issues.append("⚠️ Moderate category discrimination")
            
            self.results["category_discrimination"] = {
                "status": "fail" if overall_score < 0.3 else "warning" if overall_score < 0.5 else "pass",
                "overall_score": overall_score,
                "category_scores": discrimination_scores,
                "issues": issues
            }
            
            console.print(f"  Overall discrimination: {overall_score:.1%}")
            for cat, score in discrimination_scores.items():
                emoji = "✅" if score > 0.5 else "⚠️" if score > 0.3 else "❌"
                console.print(f"    {emoji} {cat}: {score:.1%}")
                
        except Exception as e:
            self.results["category_discrimination"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def check_normalization(self, sample_size: int = 20):
        """Check if embeddings are properly normalized."""
        console.print("[yellow]6. Checking embedding normalization...[/yellow]")
        
        try:
            # Sample vectors
            sample = self.client.scroll(
                collection_name=self.collection_name,
                limit=sample_size,
                with_vectors=True,
                with_payload=False
            )[0]
            
            if not sample:
                self.results["normalization"] = {
                    "status": "skip",
                    "reason": "No vectors to check"
                }
                console.print("  ⏭️ Skipped: No vectors")
                return
            
            # Check norms
            norms = []
            not_normalized = []
            
            for i, point in enumerate(sample):
                vector = np.array(point.vector)
                norm = np.linalg.norm(vector)
                norms.append(norm)
                
                if abs(norm - 1.0) > 0.01:  # Tolerance for floating point
                    not_normalized.append((i, norm))
            
            # Statistics
            norm_stats = {
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms))
            }
            
            issues = []
            if not_normalized:
                issues.append(f"❌ {len(not_normalized)}/{len(sample)} vectors not normalized")
            
            self.results["normalization"] = {
                "status": "fail" if not_normalized else "pass",
                "stats": norm_stats,
                "not_normalized_count": len(not_normalized),
                "sample_size": len(sample),
                "issues": issues
            }
            
            console.print(f"  Norm range: {norm_stats['min']:.4f} - {norm_stats['max']:.4f}")
            console.print(f"  Mean: {norm_stats['mean']:.4f}, Std: {norm_stats['std']:.6f}")
            
            if not_normalized:
                console.print(f"  ❌ {len(not_normalized)} vectors not normalized")
            else:
                console.print("  ✅ All vectors properly normalized")
                
        except Exception as e:
            self.results["normalization"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def verify_dimensions(self):
        """Verify vector dimensions match configuration."""
        console.print("[yellow]7. Verifying vector dimensions...[/yellow]")
        
        try:
            # Get expected dimensions
            info = self.client.get_collection(self.collection_name)
            expected_dim = info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None
            
            # Sample a vector
            sample = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_vectors=True,
                with_payload=False
            )[0]
            
            if not sample:
                self.results["dimensions"] = {
                    "status": "skip",
                    "reason": "No vectors to check"
                }
                console.print("  ⏭️ Skipped: No vectors")
                return
            
            actual_dim = len(sample[0].vector)
            model_dim = self.model.get_sentence_embedding_dimension()
            
            issues = []
            if actual_dim != expected_dim:
                issues.append(f"❌ Dimension mismatch: actual={actual_dim}, expected={expected_dim}")
            if actual_dim != model_dim:
                issues.append(f"❌ Model dimension mismatch: vector={actual_dim}, model={model_dim}")
            
            self.results["dimensions"] = {
                "status": "fail" if issues else "pass",
                "expected": expected_dim,
                "actual": actual_dim,
                "model": model_dim,
                "issues": issues
            }
            
            console.print(f"  Expected: {expected_dim}, Actual: {actual_dim}, Model: {model_dim}")
            
            if issues:
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print("  ✅ Dimensions match correctly")
                
        except Exception as e:
            self.results["dimensions"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def test_query_performance(self):
        """Test query performance and latency."""
        console.print("[yellow]8. Testing query performance...[/yellow]")
        
        test_queries = [
            "laptop",
            "wireless earbuds with noise cancellation",
            "USB-C cable for fast charging",
            "4K streaming device with voice control",
            "gaming keyboard mechanical switches RGB"
        ]
        
        try:
            latencies = []
            
            for query in test_queries:
                # Measure encoding time
                start = time.time()
                query_vec = self.model.encode([query], normalize_embeddings=True)[0]
                encode_time = time.time() - start
                
                # Measure search time
                start = time.time()
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec.tolist(),
                    limit=10,
                    with_payload=False
                )
                search_time = time.time() - start
                
                total_time = encode_time + search_time
                latencies.append({
                    "query": query[:30],
                    "encode_ms": encode_time * 1000,
                    "search_ms": search_time * 1000,
                    "total_ms": total_time * 1000
                })
            
            # Statistics
            avg_total = np.mean([l["total_ms"] for l in latencies])
            max_total = np.max([l["total_ms"] for l in latencies])
            
            issues = []
            if avg_total > 100:
                issues.append(f"⚠️ High average latency: {avg_total:.1f}ms")
            if max_total > 200:
                issues.append(f"⚠️ High max latency: {max_total:.1f}ms")
            
            self.results["query_performance"] = {
                "status": "warning" if issues else "pass",
                "avg_latency_ms": avg_total,
                "max_latency_ms": max_total,
                "latencies": latencies,
                "issues": issues
            }
            
            console.print(f"  Average latency: {avg_total:.1f}ms")
            console.print(f"  Max latency: {max_total:.1f}ms")
            
            if avg_total < 50:
                console.print("  ✅ Excellent performance")
            elif avg_total < 100:
                console.print("  ✅ Good performance")
            else:
                console.print("  ⚠️ Performance could be improved")
                
        except Exception as e:
            self.results["query_performance"] = {
                "status": "fail",
                "error": str(e)
            }
            console.print(f"  ❌ Error: {e}")
    
    def generate_report(self):
        """Generate a comprehensive diagnostic report."""
        console.print("\n[bold cyan]📊 Diagnostic Report Summary[/bold cyan]\n")
        
        # Count statuses
        status_counts = defaultdict(int)
        for test_name, test_result in self.results.items():
            status_counts[test_result.get("status", "unknown")] += 1
        
        # Overall health score
        total_tests = len(self.results)
        pass_count = status_counts["pass"]
        health_score = (pass_count / total_tests * 100) if total_tests > 0 else 0
        
        # Create summary table
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        status_emojis = {
            "pass": "✅",
            "warning": "⚠️",
            "fail": "❌",
            "skip": "⏭️"
        }
        
        for test_name, test_result in self.results.items():
            status = test_result.get("status", "unknown")
            emoji = status_emojis.get(status, "❓")
            
            # Get key detail for each test
            detail = ""
            if status == "fail" and "error" in test_result:
                detail = f"Error: {test_result['error'][:50]}..."
            elif status == "fail" and "issues" in test_result:
                detail = test_result["issues"][0] if test_result["issues"] else ""
            elif status == "pass":
                if "pass_rate" in test_result:
                    detail = f"Pass rate: {test_result['pass_rate']:.1%}"
                elif "overall_score" in test_result:
                    detail = f"Score: {test_result['overall_score']:.1%}"
            
            table.add_row(
                test_name.replace("_", " ").title(),
                f"{emoji} {status}",
                detail
            )
        
        console.print(table)
        
        # Health score panel
        health_emoji = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        console.print(Panel(
            f"{health_emoji} Overall Health Score: {health_score:.0f}%\n"
            f"✅ Passed: {status_counts['pass']}/{total_tests}\n"
            f"⚠️ Warnings: {status_counts['warning']}/{total_tests}\n"
            f"❌ Failed: {status_counts['fail']}/{total_tests}",
            title="System Health",
            border_style="bold green" if health_score > 80 else "yellow" if health_score > 60 else "red"
        ))
        
        # Recommendations
        console.print("\n[bold cyan]💡 Recommendations[/bold cyan]\n")
        
        recommendations = []
        
        # Check for critical issues
        if self.results.get("exact_matches", {}).get("status") == "fail":
            recommendations.append("🔴 Critical: Exact matches failing - rebuild vector index")
        
        if self.results.get("model_consistency", {}).get("status") == "fail":
            recommendations.append("🔴 Critical: Model inconsistency - ensure same model for ingestion and search")
        
        if self.results.get("similarity_ranges", {}).get("status") == "fail":
            recommendations.append("🔴 Critical: Poor similarity distribution - consider different embedding model")
        
        # Check for improvements
        if self.results.get("category_discrimination", {}).get("overall_score", 0) < 0.5:
            recommendations.append("🟡 Consider fine-tuning embeddings on your product data")
        
        if self.results.get("query_performance", {}).get("avg_latency_ms", 0) > 100:
            recommendations.append("🟡 Consider optimizing with smaller model or indexing")
        
        if not recommendations:
            recommendations.append("🟢 System is healthy! Consider regular monitoring.")
        
        for rec in recommendations:
            console.print(f"  • {rec}")
        
        # Save detailed report
        report_path = Path("diagnostics_report.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n📄 Detailed report saved to: {report_path}")


@app.command()
def diagnose(
    collection: str = typer.Option("products_minilm", help="Qdrant collection name"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Embedding model"),
    host: str = typer.Option("localhost", help="Qdrant host"),
    port: int = typer.Option(6333, help="Qdrant port"),
    data_path: Optional[Path] = typer.Option(None, help="Path to product data for additional tests")
):
    """Run comprehensive vector search diagnostics."""
    
    # Initialize
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(model_name)
    
    # Run diagnostics
    diagnostics = VectorDiagnostics(client, model, collection, data_path)
    results = diagnostics.run_all_diagnostics()
    
    # Return status code based on health
    health_score = sum(1 for r in results.values() if r.get("status") == "pass") / len(results) * 100
    
    if health_score < 60:
        raise typer.Exit(code=1)


@app.command()
def quick_check(
    collection: str = typer.Option("products_minilm", help="Qdrant collection name"),
    query: str = typer.Argument(..., help="Test query to check")
):
    """Quick check of a specific query."""
    
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    console.print(f"\n[bold]Testing query: '{query}'[/bold]\n")
    
    # Encode and search
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    results = client.search(
        collection_name=collection,
        query_vector=query_vec.tolist(),
        limit=5,
        with_payload=True
    )
    
    if results:
        table = Table(title="Search Results")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="yellow")
        table.add_column("Product", style="white")
        
        for i, hit in enumerate(results, 1):
            title = hit.payload.get('title', 'Unknown')[:60]
            table.add_row(str(i), f"{hit.score:.4f}", title)
        
        console.print(table)
        
        # Quick relevance check
        query_terms = query.lower().split()
        relevant = sum(1 for hit in results[:3] 
                      if any(term in hit.payload.get('title', '').lower() 
                            for term in query_terms))
        
        if relevant >= 2:
            console.print("\n✅ Results look relevant!")
        elif relevant >= 1:
            console.print("\n⚠️ Results partially relevant")
        else:
            console.print("\n❌ Results don't look relevant - consider running full diagnostics")
    else:
        console.print("❌ No results found!")


if __name__ == "__main__":
    app()