#!/usr/bin/env python
"""
Test script for Tavily MCP Server

This script tests the MCP server functionality by connecting as a client
and executing various search operations.
"""

import asyncio
import json
import os
from pathlib import Path
import sys
import pytest

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mcp.tavily_mcp_client import TavilyMCPClient, TavilyMCPAdapter

@pytest.mark.asyncio
@pytest.mark.integration
async def test_mcp_client():
    """Test the MCP client directly."""
    print("\n" + "="*60)
    print("Testing Tavily MCP Server")
    print("="*60)
    
    # Check for API key
    if not os.getenv("TAVILY_API_KEY"):
        print("\n❌ TAVILY_API_KEY not set in environment")
        print("Please set: export TAVILY_API_KEY='your-key-here'")
        return
    
    client = TavilyMCPClient()
    
    try:
        # Connect to server
        print("\n1. Connecting to MCP server...")
        await client.connect()
        print("✅ Connected successfully")
        
        # Test web search
        print("\n2. Testing web search...")
        results = await client.search_async(
            query="best wireless earbuds 2024",
            search_type="general",
            max_results=5
        )
        print(f"✅ Found {len(results)} results")
        if results:
            print(f"   First result: {results[0].title[:80]}")
            print(f"   URL: {results[0].url}")
        
        # Test product price search
        print("\n3. Testing product price search...")
        price_info = await client.search_product_info_async(
            product_name="Apple AirPods Pro",
            info_type="price"
        )
        if "prices" in price_info:
            print(f"✅ Found {len(price_info['prices'])} price results")
            if price_info['prices']:
                first_price = price_info['prices'][0]
                print(f"   Retailer: {first_price.get('retailer', 'N/A')}")
                print(f"   Price: {first_price.get('price_text', 'N/A')}")
        
        # Test finding alternatives
        print("\n4. Testing find alternatives...")
        alternatives = await client.find_alternatives_async(
            product_name="Samsung Galaxy S24",
            max_results=3
        )
        print(f"✅ Found {len(alternatives)} alternatives")
        
        # Disconnect
        print("\n5. Disconnecting...")
        await client.disconnect()
        print("✅ Disconnected successfully")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()

def test_adapter():
    """Test the backward compatibility adapter."""
    print("\n" + "="*60)
    print("Testing Backward Compatibility Adapter")
    print("="*60)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n❌ TAVILY_API_KEY not set")
        return
    
    try:
        # Create adapter (drop-in replacement for TavilyWebSearchAgent)
        print("\n1. Creating adapter...")
        adapter = TavilyMCPAdapter()
        print("✅ Adapter created")
        
        # Test synchronous search
        print("\n2. Testing synchronous search...")
        results = adapter.search(
            query="Fire TV Stick 4K reviews",
            search_type="review",
            use_cache=True
        )
        print(f"✅ Found {len(results)} results")
        if results:
            print(f"   First result: {results[0].title[:80]}")
        
        # Test product info
        print("\n3. Testing product info search...")
        info = adapter.search_product_info(
            product_name="Echo Dot",
            info_type="all"
        )
        print(f"✅ Got product info with {len(info)} categories")
        for key in info:
            if isinstance(info[key], list):
                print(f"   {key}: {len(info[key])} items")
        
        # Test parallel search
        print("\n4. Testing parallel search...")
        queries = [
            "best laptop for programming",
            "mechanical keyboard recommendations",
            "ergonomic office chair"
        ]
        results = adapter.parallel_search(queries, "general")
        print(f"✅ Completed {len(results)} parallel searches")
        for query, res in results.items():
            print(f"   '{query[:40]}...': {len(res)} results")
        
        print("\n✅ All adapter tests passed!")
        
    except Exception as e:
        print(f"\n❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("\n🚀 Starting Tavily MCP Server Tests")
    
    # Test async client
    print("\nRunning async client tests...")
    asyncio.run(test_mcp_client())
    
    # Test sync adapter
    print("\nRunning sync adapter tests...")
    test_adapter()
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    main()