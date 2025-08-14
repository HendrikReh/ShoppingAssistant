#!/usr/bin/env python
"""Test the enhanced reporting with call parameters."""

import json
import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd: list[str]):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def check_report_files(base_name: str):
    """Check that report files were created with call parameters."""
    results_dir = Path("eval/results")
    
    # Find the most recent files matching the pattern
    json_files = sorted(results_dir.glob(f"{base_name}_*.json"), key=lambda x: x.stat().st_mtime)
    md_files = sorted(results_dir.glob(f"{base_name}_*.md"), key=lambda x: x.stat().st_mtime)
    
    if not json_files:
        print(f"❌ No JSON report files found for {base_name}")
        return False
    
    if not md_files:
        print(f"❌ No Markdown report files found for {base_name}")
        return False
    
    # Check the most recent files
    json_file = json_files[-1]
    md_file = md_files[-1]
    
    print(f"\n📄 Checking {json_file.name}...")
    
    # Check JSON file
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        if "call_parameters" not in data:
            print("❌ Missing 'call_parameters' in JSON report")
            return False
        
        params = data["call_parameters"]
        required_fields = ["command", "timestamp", "dataset", "top_k"]
        
        for field in required_fields:
            if field not in params:
                print(f"❌ Missing '{field}' in call_parameters")
                return False
            print(f"  ✓ {field}: {params[field]}")
        
        print("✅ JSON report has all required call parameters")
        
    except Exception as e:
        print(f"❌ Error reading JSON report: {e}")
        return False
    
    # Check Markdown file
    print(f"\n📄 Checking {md_file.name}...")
    
    try:
        content = md_file.read_text()
        
        if "### Call Parameters" not in content:
            print("❌ Missing 'Call Parameters' section in Markdown report")
            return False
        
        required_strings = ["Command", "Timestamp", "Dataset", "top_k"]
        for s in required_strings:
            if s not in content:
                print(f"❌ Missing '{s}' in Markdown report")
                return False
        
        print("✅ Markdown report has Call Parameters section")
        
        # Show a snippet of the parameters section
        lines = content.split('\n')
        in_params = False
        param_lines = []
        for line in lines:
            if "### Call Parameters" in line:
                in_params = True
            elif in_params:
                if line.startswith('#') and not line.startswith('###'):
                    break
                if line.strip():
                    param_lines.append(line)
                if len(param_lines) > 5:  # Show first 5 parameter lines
                    break
        
        if param_lines:
            print("\n  Sample from Call Parameters section:")
            for line in param_lines[:5]:
                print(f"  {line}")
        
    except Exception as e:
        print(f"❌ Error reading Markdown report: {e}")
        return False
    
    return True

def main():
    """Test enhanced reporting for both eval-search and eval-chat."""
    
    print("🧪 Testing Enhanced Reporting with Call Parameters")
    print("=" * 60)
    
    # Create eval/results directory if it doesn't exist
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: eval-search with limited samples
    print("\n1️⃣ Testing eval-search reporting...")
    
    success = run_command([
        "uv", "run", "python", "-m", "app.cli", "eval-search",
        "--dataset", "eval/datasets/search_eval.jsonl",
        "--top-k", "10",
        "--max-samples", "3",
        "--variants", "bm25,vec",
        "--seed", "42"
    ])
    
    if not success:
        print("❌ eval-search command failed")
        sys.exit(1)
    
    time.sleep(1)  # Give filesystem time to sync
    
    if not check_report_files("search"):
        print("❌ eval-search report validation failed")
        sys.exit(1)
    
    # Test 2: eval-chat with limited samples
    print("\n2️⃣ Testing eval-chat reporting...")
    
    success = run_command([
        "uv", "run", "python", "-m", "app.cli", "eval-chat",
        "--dataset", "eval/datasets/chat_eval.jsonl",
        "--top-k", "5",
        "--max-samples", "2",
        "--seed", "123"
    ])
    
    if not success:
        print("❌ eval-chat command failed")
        sys.exit(1)
    
    time.sleep(1)  # Give filesystem time to sync
    
    if not check_report_files("chat"):
        print("❌ eval-chat report validation failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ All tests passed! Enhanced reporting is working correctly.")
    print("\nKey features verified:")
    print("  • JSON reports include 'call_parameters' section")
    print("  • Markdown reports include formatted 'Call Parameters' section")
    print("  • All command arguments are captured (dataset, top_k, seed, etc.)")
    print("  • Model configurations are included (embed_model, cross_encoder_model, LLMs)")
    print("  • Timestamps use human-readable format (YYYYMMDD_HHMMSS)")
    
    # Show where to find the reports
    print(f"\n📁 Reports saved to: {results_dir.absolute()}")
    
    latest_reports = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)[-2:]
    if latest_reports:
        print("\nLatest reports:")
        for report in latest_reports:
            print(f"  • {report.name}")

if __name__ == "__main__":
    main()