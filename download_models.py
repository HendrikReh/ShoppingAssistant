#!/usr/bin/env python
"""
Pre-download models for offline use.

Run this script when you have internet connection to download all required models:
    python download_models.py
"""

import sys
from pathlib import Path


def download_models():
    """Download all required models for the Shopping Assistant."""
    
    print("🔽 Downloading models for offline use...\n")
    
    # Embedding model
    try:
        print("1. Downloading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        cache_path = Path.home() / '.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2'
        if cache_path.exists():
            print(f"   ✅ Saved to: {cache_path}")
        else:
            print("   ✅ Downloaded successfully")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Cross-encoder model
    try:
        print("\n2. Downloading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-12-v2")
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        cache_path = Path.home() / '.cache/torch/sentence_transformers/cross-encoder_ms-marco-MiniLM-L-12-v2'
        if cache_path.exists():
            print(f"   ✅ Saved to: {cache_path}")
        else:
            print("   ✅ Downloaded successfully")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Legacy GTE-large model (if still needed for existing data)
    try:
        print("\n3. Downloading legacy model: thenlper/gte-large")
        print("   (Optional - only needed if you have existing Qdrant collections)")
        model = SentenceTransformer('thenlper/gte-large')
        cache_path = Path.home() / '.cache/torch/sentence_transformers/thenlper_gte-large'
        if cache_path.exists():
            print(f"   ✅ Saved to: {cache_path}")
        else:
            print("   ✅ Downloaded successfully")
    except Exception as e:
        print(f"   ⚠️  Skipped (not critical): {e}")
    
    print("\n✨ All models downloaded successfully!")
    print("\nYou can now use the Shopping Assistant offline.")
    print("The models are cached in: ~/.cache/torch/sentence_transformers/")
    
    return True


if __name__ == "__main__":
    try:
        success = download_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)