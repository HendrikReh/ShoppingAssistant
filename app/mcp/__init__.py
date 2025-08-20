"""
MCP (Model Context Protocol) implementations for ShoppingAssistant.

This module contains MCP servers and clients that expose ShoppingAssistant
functionality to MCP-compatible clients like Claude Desktop.
"""

from .tavily_mcp_client import (
    TavilyMCPClient,
    TavilyMCPAdapter,
    WebSearchResult,
    get_mcp_agent
)

from .shopping_assistant_mcp_client import (
    ShoppingAssistantMCPClient,
    ShoppingAssistantAdapter,
    ShoppingSearchResult,
    get_shopping_assistant
)

__all__ = [
    # Tavily MCP
    "TavilyMCPClient",
    "TavilyMCPAdapter", 
    "WebSearchResult",
    "get_mcp_agent",
    # ShoppingAssistant MCP
    "ShoppingAssistantMCPClient",
    "ShoppingAssistantAdapter",
    "ShoppingSearchResult",
    "get_shopping_assistant"
]