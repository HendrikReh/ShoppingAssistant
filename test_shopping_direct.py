#!/usr/bin/env python
"""
Direct test of ShoppingAssistant MCP Server functions.

This tests the server functions directly without MCP protocol.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_direct():
    """Test server functions directly."""
    print("\n" + "="*60)
    print("Testing ShoppingAssistant Search Functions Directly")
    print("="*60)
    
    # Import server module
    from app.mcp import shopping_assistant_mcp_server as server
    
    try:
        # Initialize components
        print("\n1. Initializing search components...")
        server.ensure_initialized()
        print("✅ Components initialized")
        
        # Test product search
        print("\n2. Testing product search...")
        response = await server.search_products_impl(
            query="wireless earbuds",
            top_k=5,
            use_reranking=True
        )
        print(f"✅ Found {response.total_found} products")
        if response.results:
            print(f"   Top result: {response.results[0].title[:80]}")
            print(f"   Score: {response.results[0].score:.3f}")
        
        # Test review search
        print("\n3. Testing review search...")
        response = await server.search_reviews_impl(
            query="battery life",
            top_k=5
        )
        print(f"✅ Found {response.total_found} reviews")
        if response.results:
            print(f"   Top review: {response.results[0].title[:80]}")
        
        # Test hybrid search
        print("\n4. Testing hybrid search...")
        response = await server.hybrid_search_impl(
            query="Fire TV Stick",
            top_k=10
        )
        print(f"✅ Found {response.total_found} results")
        products = sum(1 for r in response.results if r.type == "product")
        reviews = sum(1 for r in response.results if r.type == "review")
        print(f"   Products: {products}, Reviews: {reviews}")
        
        # Test stats resource
        print("\n5. Testing stats resource...")
        stats_json = await server.get_stats_impl()
        import json
        stats = json.loads(stats_json)
        print(f"✅ System stats:")
        print(f"   Products: {stats['products_indexed']}")
        print(f"   Reviews: {stats['reviews_indexed']}")
        print(f"   Device: {stats['device']}")
        
        print("\n✨ All direct tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run direct tests."""
    print("\n🚀 Direct ShoppingAssistant Search Test")
    
    # Check for data files
    from app.cli import DATA_PRODUCTS, DATA_REVIEWS
    
    if not DATA_PRODUCTS.exists() or not DATA_REVIEWS.exists():
        print("\n❌ Data files not found. Please run:")
        print("   uv run python -m app.cli ingest")
        return
    
    # Run tests
    asyncio.run(test_direct())

if __name__ == "__main__":
    main()