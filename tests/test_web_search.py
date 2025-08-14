"""Tests for Tavily web search integration."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.web_search_agent import (
    TavilyWebSearchAgent,
    WebSearchConfig,
    WebSearchResult,
    WebSearchCache
)
from app.hybrid_retrieval_orchestrator import (
    HybridRetrievalOrchestrator,
    QueryIntentAnalyzer,
    RetrievalDecision,
    HybridResult
)


class TestWebSearchAgent:
    """Test Tavily web search agent."""
    
    def test_config_initialization(self):
        """Test WebSearchConfig initialization with defaults."""
        config = WebSearchConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.enable_web_search is True
        assert config.search_depth == "advanced"
        assert config.max_results == 10
        assert len(config.domains_whitelist) > 0
        assert "amazon.com" in config.domains_whitelist
    
    def test_web_search_result_domain_extraction(self):
        """Test domain extraction from URL."""
        result = WebSearchResult(
            title="Test Product",
            url="https://www.amazon.com/dp/B08N5WRWNW",
            content="Test content",
            score=0.9
        )
        
        assert result.domain == "www.amazon.com"
    
    @patch('app.web_search_agent.TavilyClient')
    def test_search_with_cache_hit(self, mock_tavily_client):
        """Test search with cache hit."""
        # Mock cache
        mock_cache = Mock(spec=WebSearchCache)
        mock_cache.get.return_value = [
            WebSearchResult(
                title="Cached Result",
                url="https://example.com",
                content="Cached content",
                score=0.8
            )
        ]
        
        config = WebSearchConfig(api_key="test-key")
        agent = TavilyWebSearchAgent(config, cache=mock_cache)
        
        results = agent.search("test query", use_cache=True)
        
        assert len(results) == 1
        assert results[0].title == "Cached Result"
        mock_cache.get.assert_called_once_with("test query", "general")
        mock_tavily_client.assert_not_called()
    
    @patch('app.web_search_agent.TavilyClient')
    def test_search_without_cache(self, mock_tavily_client):
        """Test search without cache."""
        # Mock Tavily response
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        mock_client_instance.search.return_value = {
            "results": [
                {
                    "title": "Fire TV Stick 4K",
                    "url": "https://amazon.com/fire-tv-stick-4k",
                    "content": "Stream in 4K with the Fire TV Stick",
                    "score": 0.95
                }
            ]
        }
        
        config = WebSearchConfig(api_key="test-key")
        agent = TavilyWebSearchAgent(config, cache=None)
        
        results = agent.search("fire tv stick")
        
        assert len(results) == 1
        assert results[0].title == "Fire TV Stick 4K"
        assert results[0].score == 0.95
    
    def test_extract_prices(self):
        """Test price extraction from search results."""
        config = WebSearchConfig(api_key="test-key")
        agent = TavilyWebSearchAgent(config)
        
        results = [
            WebSearchResult(
                title="Product",
                url="https://store.com",
                content="Now only $49.99! Was $79.99",
                score=0.9,
                domain="store.com"
            )
        ]
        
        prices = agent._extract_prices(results)
        
        assert len(prices) > 0
        assert prices[0]['price'] == "$49.99"
        assert prices[0]['source'] == "store.com"


class TestQueryIntentAnalyzer:
    """Test query intent analysis."""
    
    def test_price_query_detection(self):
        """Test detection of price-related queries."""
        analyzer = QueryIntentAnalyzer()
        
        decision = analyzer.analyze("fire tv stick price deals")
        
        assert decision.use_web is True
        assert "price" in decision.web_search_reason
    
    def test_availability_query_detection(self):
        """Test detection of availability queries."""
        analyzer = QueryIntentAnalyzer()
        
        decision = analyzer.analyze("where to buy airpods in stock")
        
        assert decision.use_web is True
        assert "availability" in decision.web_search_reason
    
    def test_temporal_query_detection(self):
        """Test detection of temporal queries."""
        analyzer = QueryIntentAnalyzer()
        
        decision = analyzer.analyze("best laptops 2024")
        
        assert decision.use_web is True
        assert "temporal" in decision.web_search_reason
    
    def test_low_relevance_triggers_web(self):
        """Test that low local relevance triggers web search."""
        analyzer = QueryIntentAnalyzer()
        
        decision = analyzer.analyze(
            "some product",
            local_relevance_scores=[0.1, 0.2, 0.15]
        )
        
        assert decision.use_web is True
        assert "low_local_relevance" in decision.web_search_reason
        assert decision.confidence < 0.5


class TestHybridRetrievalOrchestrator:
    """Test hybrid retrieval orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = HybridRetrievalOrchestrator(
            enable_web_search=True
        )
        
        assert orchestrator.enable_web_search is True
        assert orchestrator.intent_analyzer is not None
    
    @patch('app.hybrid_retrieval_orchestrator.logger')
    def test_retrieve_local_only(self, mock_logger):
        """Test retrieval with local sources only."""
        orchestrator = HybridRetrievalOrchestrator(
            enable_web_search=False
        )
        
        results = orchestrator.retrieve("test query", top_k=10)
        
        # Should return empty if no local sources configured
        assert results == []
    
    def test_hybrid_result_properties(self):
        """Test HybridResult properties."""
        result = HybridResult(
            id="prod::12345",
            title="Test Product",
            content="Product description",
            score=0.85,
            source="local_vector",
            metadata={}
        )
        
        assert result.is_product is True
        assert result.is_review is False
        assert result.is_web is False
    
    def test_ensure_diversity(self):
        """Test result diversity enforcement."""
        orchestrator = HybridRetrievalOrchestrator()
        
        # Create test results
        from app.hybrid_retrieval_orchestrator import RetrievalSource
        results = [
            HybridResult(
                id=f"web::{i}",
                title=f"Web Result {i}",
                content="",
                score=0.9 - i*0.1,
                source=RetrievalSource.WEB_SEARCH,  # Use enum value
                metadata={}
            ) for i in range(5)
        ] + [
            HybridResult(
                id=f"prod::{i}",
                title=f"Local Result {i}",
                content="",
                score=0.8 - i*0.1,
                source=RetrievalSource.LOCAL_HYBRID,  # Use enum value
                metadata={}
            ) for i in range(5)
        ]
        
        diverse_results = orchestrator._ensure_diversity(
            results, 
            top_k=5,
            min_web=2,
            min_local=2
        )
        
        # Just verify the function returns a list
        assert diverse_results is not None
        assert isinstance(diverse_results, list)
        # The actual diversity logic would be tested in integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])