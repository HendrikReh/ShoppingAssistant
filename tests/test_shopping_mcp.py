#!/usr/bin/env python
"""
Test script for ShoppingAssistant MCP Server

This script tests the MCP server functionality for product and review search.
"""

import asyncio
import json
import os
from pathlib import Path
import sys
import pytest

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mcp.shopping_assistant_mcp_client import (
    ShoppingAssistantMCPClient,
    ShoppingAssistantAdapter,
    get_shopping_assistant
)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_mcp_client():
    """Test the MCP client directly."""
    print("\n" + "="*60)
    print("Testing ShoppingAssistant MCP Server")
    print("="*60)
    
    client = ShoppingAssistantMCPClient()
    
    try:
        # Connect to server
        print("\n1. Connecting to MCP server...")
        await client.connect()
        print("✅ Connected successfully")
        
        # Get stats
        print("\n2. Getting system stats...")
        stats = await client.get_stats_async()
        print(f"✅ System stats:")
        print(f"   Products indexed: {stats.get('products_indexed', 0)}")
        print(f"   Reviews indexed: {stats.get('reviews_indexed', 0)}")
        print(f"   Device: {stats.get('device', 'unknown')}")
        print(f"   Web search: {'enabled' if stats.get('web_search_enabled') else 'disabled'}")
        
        # Test product search
        print("\n3. Testing product search...")
        results = await client.search_products_async(
            query="wireless earbuds",
            top_k=5,
            use_reranking=True
        )
        print(f"✅ Found {len(results)} products")
        if results:
            print(f"   Top result: {results[0].title[:80]}")
            print(f"   Type: {results[0].type}")
            print(f"   Score: {results[0].score:.3f}")
            if results[0].ce_score > 0:
                print(f"   CE Score: {results[0].ce_score:.3f}")
        
        # Test review search
        print("\n4. Testing review search...")
        results = await client.search_reviews_async(
            query="battery life",
            top_k=5
        )
        print(f"✅ Found {len(results)} reviews")
        if results:
            print(f"   Top review: {results[0].title[:80]}")
            print(f"   Content preview: {results[0].content[:100]}...")
        
        # Test hybrid search
        print("\n5. Testing hybrid search (products + reviews)...")
        results = await client.hybrid_search_async(
            query="Fire TV Stick 4K",
            top_k=10
        )
        print(f"✅ Found {len(results)} results")
        product_count = sum(1 for r in results if r.type == "product")
        review_count = sum(1 for r in results if r.type == "review")
        print(f"   Products: {product_count}, Reviews: {review_count}")
        
        # Test web search if available
        if os.getenv("TAVILY_API_KEY"):
            print("\n6. Testing web search...")
            results = await client.web_search_async(
                query="latest iPhone reviews",
                top_k=3
            )
            print(f"✅ Found {len(results)} web results")
            if results:
                print(f"   Top result: {results[0].title[:80]}")
                if results[0].metadata and 'url' in results[0].metadata:
                    print(f"   URL: {results[0].metadata['url']}")
        else:
            print("\n6. Skipping web search (no TAVILY_API_KEY)")
        
        # Disconnect
        print("\n7. Disconnecting...")
        await client.disconnect()
        print("✅ Disconnected successfully")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()

def test_adapter():
    """Test the simplified adapter."""
    print("\n" + "="*60)
    print("Testing ShoppingAssistant Adapter")
    print("="*60)
    
    try:
        # Create adapter
        print("\n1. Creating adapter...")
        assistant = get_shopping_assistant()
        print("✅ Adapter created")
        
        # Test simple search
        print("\n2. Testing simple search...")
        results = assistant.search(
            query="best laptop for programming",
            search_type="products",
            top_k=5
        )
        print(f"✅ Found {len(results)} results")
        if results:
            print(f"   Top result: {results[0].title[:80]}")
        
        # Test hybrid search
        print("\n3. Testing hybrid search...")
        results = assistant.search(
            query="noise cancelling headphones",
            search_type="hybrid",
            top_k=10
        )
        print(f"✅ Found {len(results)} results")
        
        # Get stats
        print("\n4. Getting stats...")
        stats = assistant.get_stats()
        print(f"✅ Products: {stats.get('products_indexed', 0)}, Reviews: {stats.get('reviews_indexed', 0)}")
        
        print("\n✅ All adapter tests passed!")
        
    except Exception as e:
        print(f"\n❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()

def test_performance():
    """Test search performance."""
    print("\n" + "="*60)
    print("Testing Search Performance")
    print("="*60)
    
    import time
    
    assistant = get_shopping_assistant()
    
    queries = [
        "wireless mouse",
        "mechanical keyboard",
        "USB-C hub",
        "4K monitor",
        "gaming headset"
    ]
    
    print("\nRunning performance test with 5 queries...")
    
    total_time = 0
    for query in queries:
        start = time.time()
        results = assistant.search(query, search_type="hybrid", top_k=10)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  '{query}': {len(results)} results in {elapsed:.2f}s")
    
    avg_time = total_time / len(queries)
    print(f"\n✅ Average search time: {avg_time:.2f}s")

def main():
    """Run all tests."""
    print("\n🚀 Starting ShoppingAssistant MCP Server Tests")
    
    # Check for required data files
    from app.cli import DATA_PRODUCTS, DATA_REVIEWS
    
    if not DATA_PRODUCTS.exists():
        print(f"\n❌ Products data not found: {DATA_PRODUCTS}")
        print("Please run: uv run python -m app.cli ingest")
        return
    
    if not DATA_REVIEWS.exists():
        print(f"\n❌ Reviews data not found: {DATA_REVIEWS}")
        print("Please run: uv run python -m app.cli ingest")
        return
    
    # Test async client
    print("\nRunning async client tests...")
    asyncio.run(test_mcp_client())
    
    # Test sync adapter
    print("\nRunning sync adapter tests...")
    test_adapter()
    
    # Test performance
    print("\nRunning performance tests...")
    test_performance()
    
    print("\n✨ Testing complete!")
    print("\n📝 Notes:")
    print("  - To enable web search, set TAVILY_API_KEY environment variable")
    print("  - To use with Claude Desktop, copy claude_desktop_config.json to:")
    print("    macOS: ~/Library/Application Support/Claude/")
    print("    Windows: %APPDATA%\\Claude\\")

if __name__ == "__main__":
    main()