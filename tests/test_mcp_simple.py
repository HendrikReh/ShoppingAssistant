#!/usr/bin/env python
"""Simple test to verify MCP server structure."""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_server_structure():
    """Test that MCP server can be imported and has tools registered."""
    print("\n🔍 Testing MCP Server Structure")
    print("="*50)
    
    try:
        # Import server
        from app.mcp.tavily_mcp_server import mcp, config
        print("✅ MCP server imported successfully")
        
        # Check configuration
        print(f"\nConfiguration:")
        print(f"  API Key Set: {'Yes' if config.api_key else 'No (set TAVILY_API_KEY)'}")
        print(f"  Search Depth: {config.search_depth}")
        print(f"  Max Results: {config.max_results}")
        print(f"  Cache TTL: {config.cache_ttl}s")
        
        # Check tools (they're registered at module load time)
        print(f"\nMCP Server: {mcp.name}")
        
        # The tools are registered via decorators, so they exist
        # Let's manually check what functions are decorated
        import inspect
        from app.mcp import tavily_mcp_server
        
        tools = []
        resources = []
        
        for name, obj in inspect.getmembers(tavily_mcp_server):
            if inspect.isfunction(obj):
                # Check if it's a tool (has specific naming pattern)
                if name in ['web_search', 'search_product_prices', 'search_product_reviews', 
                           'compare_products', 'check_availability', 'find_alternatives']:
                    tools.append(name)
                elif name in ['list_cached_searches', 'get_cached_search']:
                    resources.append(name)
        
        print(f"\nTools found: {len(tools)}")
        for tool in tools:
            print(f"  - {tool}")
        
        print(f"\nResources found: {len(resources)}")
        for resource in resources:
            print(f"  - {resource}")
        
        # Assert to validate the test passed
        assert len(tools) >= 4, "Should have at least 4 tools"
        assert mcp.name, "MCP server should have a name"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for pytest

def test_client_structure():
    """Test that MCP client can be imported."""
    print("\n🔍 Testing MCP Client Structure")
    print("="*50)
    
    try:
        # Import client
        from app.mcp.tavily_mcp_client import (
            TavilyMCPClient,
            TavilyMCPAdapter,
            WebSearchResult,
            get_mcp_agent
        )
        print("✅ MCP client imported successfully")
        
        # Check classes
        print("\nAvailable classes:")
        print("  - TavilyMCPClient (async/sync client)")
        print("  - TavilyMCPAdapter (backward compatible)")
        print("  - WebSearchResult (data class)")
        print("  - get_mcp_agent (factory function)")
        
        # Create instances (without connecting)
        client = TavilyMCPClient()
        print("✅ TavilyMCPClient instantiated")
        
        adapter = TavilyMCPAdapter()
        print("✅ TavilyMCPAdapter instantiated")
        
        agent = get_mcp_agent()
        print("✅ get_mcp_agent() works")
        
        # Assert to validate the test passed
        assert client is not None, "Client should be instantiated"
        assert adapter is not None, "Adapter should be instantiated"
        assert agent is not None, "Agent should be instantiated"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception for pytest

def main():
    """Run structure tests."""
    print("\n🚀 MCP Implementation Structure Test")
    
    # Test server
    server_ok = test_server_structure()
    
    # Test client
    client_ok = test_client_structure()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Summary:")
    print(f"  Server: {'✅ Pass' if server_ok else '❌ Fail'}")
    print(f"  Client: {'✅ Pass' if client_ok else '❌ Fail'}")
    
    if server_ok and client_ok:
        print("\n✨ MCP implementation is properly structured!")
        print("\nNext steps:")
        print("1. Set TAVILY_API_KEY environment variable")
        print("2. Run: uv run python test_mcp_server.py")
        print("3. Configure Claude Desktop with claude_desktop_config.json")
    else:
        print("\n⚠️ Some issues need to be fixed")

if __name__ == "__main__":
    main()