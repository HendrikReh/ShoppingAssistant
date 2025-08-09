#!/usr/bin/env python
"""Test that RAGAS metrics are correctly imported and configured."""

import os
import sys

def test_ragas_import():
    """Test that we can import the correct RAGAS metrics."""
    try:
        from ragas.metrics import ContextRelevance, ContextUtilization
        print("✓ RAGAS metrics imported successfully")
        print(f"  - ContextRelevance: {ContextRelevance}")
        print(f"  - ContextUtilization: {ContextUtilization}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RAGAS metrics: {e}")
        return False

def test_llm_config():
    """Test LLM configuration module."""
    try:
        from app.llm_config import get_llm_config, LLMConfig
        
        # Test with no API key (should fail)
        config = LLMConfig()
        try:
            config.validate()
            print("✗ LLM config should have failed without API key")
            return False
        except ValueError as e:
            print(f"✓ LLM config correctly fails without API key: {e}")
        
        # Test with dummy API key
        config = LLMConfig(api_key="test-key-123")
        try:
            config.validate()
            print("✓ LLM config validates with API key")
        except Exception as e:
            print(f"✗ LLM config failed with API key: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed to test LLM config: {e}")
        return False

def test_metric_instantiation():
    """Test that metrics can be instantiated."""
    try:
        from ragas.metrics import ContextRelevance, ContextUtilization
        
        # Try to instantiate metrics
        metric1 = ContextRelevance()
        metric2 = ContextUtilization()
        
        print("✓ RAGAS metrics instantiated successfully")
        print(f"  - ContextRelevance instance: {metric1}")
        print(f"  - ContextUtilization instance: {metric2}")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate metrics: {e}")
        return False

def main():
    print("Testing RAGAS evaluation fix...\n")
    
    all_passed = True
    all_passed &= test_ragas_import()
    all_passed &= test_llm_config()
    all_passed &= test_metric_instantiation()
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed! The evaluation should work with proper API key.")
        print("\nTo run evaluation, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  uv run python -m app.cli eval-search --dataset eval/datasets/search_eval.jsonl")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()