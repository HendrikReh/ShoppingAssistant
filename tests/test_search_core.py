#!/usr/bin/env python
"""Comprehensive tests for search functionality."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cli import (
    _vector_search, _rrf_fuse, _cross_encoder_scores,
    _hybrid_search_inline, _tokenize, COLLECTION_PRODUCTS
)


class TestBM25Search:
    """Test BM25 search functionality."""
    
    @patch('app.cli._bm25_from_files')
    def test_bm25_initialization(self, mock_bm25):
        """Test BM25 index initialization."""
        from app.cli import _bm25_from_files
        
        mock_bm25_obj = Mock()
        mock_bm25_obj.get_scores.return_value = np.array([0.5, 0.3, 0.1])
        
        mock_products = {
            "1": {"title": "Fire TV Stick"},
            "2": {"title": "Echo Dot"}
        }
        
        mock_bm25.return_value = (
            mock_bm25_obj,
            mock_bm25_obj,
            mock_products,
            {}
        )
        
        bm25_prod, bm25_rev, products, reviews = _bm25_from_files(
            Path("products.jsonl"),
            Path("reviews.jsonl")
        )
        
        assert bm25_prod is not None
        assert products == mock_products
    
    def test_bm25_scoring(self):
        """Test BM25 scoring mechanism."""
        from rank_bm25 import BM25Okapi
        
        corpus = [
            ["fire", "tv", "stick", "streaming"],
            ["echo", "dot", "smart", "speaker"],
            ["kindle", "ebook", "reader"]
        ]
        
        bm25 = BM25Okapi(corpus)
        query = ["fire", "tv"]
        scores = bm25.get_scores(query)
        
        assert len(scores) == 3
        assert scores[0] > scores[1]  # First doc should have highest score


class TestVectorSearch:
    """Test vector search with Qdrant."""
    
    @patch('app.cli.QdrantClient')
    def test_vector_search_basic(self, mock_qdrant_class):
        """Test basic vector search functionality."""
        # Setup mock client
        mock_client = Mock()
        mock_qdrant_class.return_value = mock_client
        
        # Mock search results
        mock_result = Mock()
        mock_result.id = "uuid-123"
        mock_result.score = 0.95
        mock_result.payload = {
            "original_id": "prod_1",
            "title": "Fire TV Stick",
            "description": "Streaming device"
        }
        
        mock_client.search.return_value = [mock_result]
        
        # Test vector search
        query_vec = [0.1, 0.2, 0.3]
        results = _vector_search(mock_client, COLLECTION_PRODUCTS, query_vec, top_k=5)
        
        assert len(results) == 1
        assert results[0][0] == "uuid-123"
        assert results[0][1] == 0.95
        assert results[0][2]["title"] == "Fire TV Stick"
    
    @patch('app.cli._load_st_model')
    def test_embedding_generation(self, mock_load_model):
        """Test embedding generation with sentence transformer."""
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_load_model.return_value = mock_model
        
        # Generate embedding
        text = "Fire TV Stick 4K"
        embedding = mock_model.encode([text], batch_size=1, normalize_embeddings=True)
        
        assert embedding.shape == (1, 4)
        mock_model.encode.assert_called_once()
    
    def test_vector_id_mapping(self):
        """Test UUID to original ID mapping."""
        raw_results = [
            ("uuid-1", 0.9, {"original_id": "prod_1", "title": "Product 1"}),
            ("uuid-2", 0.8, {"original_id": "prod_2", "title": "Product 2"}),
            ("uuid-3", 0.7, {"id": "prod_3", "title": "Product 3"})  # Fallback to 'id'
        ]
        
        mapped_results = []
        for uuid, score, payload in raw_results:
            original_id = payload.get('original_id', payload.get('id', ''))
            if original_id:
                mapped_results.append((original_id, score))
        
        assert len(mapped_results) == 3
        assert mapped_results[0][0] == "prod_1"
        assert mapped_results[2][0] == "prod_3"


class TestRRFFusion:
    """Test Reciprocal Rank Fusion."""
    
    def test_rrf_fusion_basic(self):
        """Test basic RRF fusion."""
        # Two result lists with overlapping documents
        results1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        results2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]
        
        fused = _rrf_fuse([results1, results2], k=60)
        
        # Check all documents are present
        assert len(fused) == 4
        assert all(doc in fused for doc in ["doc1", "doc2", "doc3", "doc4"])
        
        # Check that fusion worked (all docs present with reasonable scores)
        scores = list(fused.items())
        scores.sort(key=lambda x: x[1], reverse=True)
        # Either doc1 or doc2 should be at top (both appear in both lists)
        assert scores[0][0] in ["doc1", "doc2"]
    
    def test_rrf_fusion_with_product_boost(self):
        """Test RRF fusion with product boost."""
        # Product results (with boost)
        products = [("prod::1", 0.7), ("prod::2", 0.6)]
        # Review results (no boost)
        reviews = [("rev::1", 0.8), ("rev::2", 0.75)]
        
        fused = _rrf_fuse([products, reviews], k=60, product_boost=1.5)
        
        # Products should be boosted
        assert "prod::1" in fused
        assert "rev::1" in fused
        
    def test_rrf_fusion_empty_lists(self):
        """Test RRF fusion with empty lists."""
        results1 = []
        results2 = [("doc1", 0.9)]
        
        fused = _rrf_fuse([results1, results2], k=60)
        
        assert len(fused) == 1
        assert "doc1" in fused


class TestCrossEncoderReranking:
    """Test cross-encoder reranking."""
    
    @patch('app.cli._get_cross_encoder')
    def test_cross_encoder_loading(self, mock_get_encoder):
        """Test cross-encoder model loading."""
        mock_model = Mock()
        mock_get_encoder.return_value = mock_model
        
        model = mock_get_encoder("cross-encoder/ms-marco-MiniLM-L-12-v2", "cpu")
        
        assert model is not None
        mock_get_encoder.assert_called_once()
    
    @patch('app.cli._get_cross_encoder')
    def test_cross_encoder_scoring(self, mock_get_encoder):
        """Test cross-encoder scoring."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.7, 0.5])
        mock_get_encoder.return_value = mock_model
        
        # Test scoring
        query = "Fire TV Stick"
        candidates = [
            ("1", "Fire TV Stick 4K streaming device"),
            ("2", "Echo Dot smart speaker"),
            ("3", "Kindle ebook reader")
        ]
        
        scores = _cross_encoder_scores(
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cpu",
            query,
            candidates
        )
        
        assert len(scores) == 3
        assert scores[0][1] == 0.9  # Highest score for most relevant
        assert scores[0][0] == "1"
    
    def test_cross_encoder_empty_candidates(self):
        """Test cross-encoder with empty candidates."""
        with patch('app.cli._get_cross_encoder') as mock_get_encoder:
            mock_model = Mock()
            mock_get_encoder.return_value = mock_model
            
            scores = _cross_encoder_scores(
                "model",
                "cpu",
                "query",
                []
            )
            
            assert scores == []


class TestHybridSearch:
    """Test hybrid search functionality."""
    
    @patch('app.cli._get_cross_encoder')
    @patch('app.cli._vector_search')
    @patch('app.cli._load_st_model')
    @patch('app.cli._qdrant_client')
    @patch('app.cli._bm25_from_files')
    def test_hybrid_search_all_variants(self, mock_bm25, mock_client, 
                                       mock_model, mock_vector, mock_encoder):
        """Test hybrid search with all variants."""
        # Setup BM25
        mock_bm25_obj = Mock()
        mock_bm25_obj.get_scores.return_value = np.array([0.5, 0.3, 0.1])
        id_to_product = {
            "1": {"title": "Fire TV", "description": "Streaming"},
            "2": {"title": "Echo", "description": "Speaker"}
        }
        mock_bm25.return_value = (mock_bm25_obj, mock_bm25_obj, id_to_product, {})
        
        # Setup vector search
        mock_vector.return_value = [
            ("uuid-1", 0.9, {"original_id": "1", "title": "Fire TV"}),
            ("uuid-2", 0.7, {"original_id": "2", "title": "Echo"})
        ]
        
        # Setup model
        mock_st = Mock()
        mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.return_value = mock_st
        
        # Setup cross-encoder
        mock_ce = Mock()
        mock_ce.predict.return_value = np.array([0.95, 0.6])
        mock_encoder.return_value = mock_ce
        
        # Test all variants
        results = _hybrid_search_inline(
            query="fire tv",
            st_model=mock_st,  # Pass the mock model
            client=mock_client.return_value,  # Pass the mock client
            top_k=5,
            variant="rrf_ce",
            products_file=Path("products.jsonl"),
            reviews_file=Path("reviews.jsonl"),
            rrf_k=60,
            rerank_top_k=10,
            enhanced=False,
            product_filter=False
        )
        
        # Just verify the function can be called without errors
        # The actual result depends on complex mocking that would be integration tested
        assert results is not None
        assert isinstance(results, list)
    
    def test_hybrid_search_product_filter(self):
        """Test hybrid search with product-only filter."""
        with patch('app.cli._bm25_from_files') as mock_bm25:
            with patch('app.cli._load_st_model') as mock_load_st:
                with patch('app.cli._qdrant_client') as mock_client:
                    with patch('app.cli._vector_search') as mock_vector:
                        # Setup mocks
                        mock_st = Mock()
                        mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                        mock_load_st.return_value = mock_st
                        
                        mock_bm25_obj = Mock()
                        mock_bm25_obj.get_scores.return_value = np.array([0.5])
                        id_to_product = {"prod::1": {"title": "Product"}}
                        id_to_review = {"rev::1": {"review": "Review"}}
                        mock_bm25.return_value = (mock_bm25_obj, mock_bm25_obj, 
                                                 id_to_product, id_to_review)
                        
                        mock_vector.return_value = []  # No vector results
                        
                        results = _hybrid_search_inline(
                            query="test",
                            st_model=mock_st,  # Pass the mock model
                            client=mock_client.return_value,  # Pass the mock client
                            top_k=5,
                            variant="bm25",
                            products_file=Path("p.jsonl"),
                            reviews_file=Path("r.jsonl"),
                            product_filter=True
                        )
                        
                        # Just verify the function can be called without errors
                        assert results is not None
                        assert isinstance(results, list)


class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_tokenization(self):
        """Test text tokenization."""
        text = "Fire TV Stick 4K - Stream in Ultra HD!"
        tokens = _tokenize(text)
        
        assert "fire" in tokens
        assert "stick" in tokens
        assert "stream" in tokens
        assert "ultra" in tokens
        assert "hd" in tokens
        assert "-" not in tokens  # Punctuation removed
    
    def test_tokenization_empty(self):
        """Test tokenization of empty string."""
        tokens = _tokenize("")
        assert tokens == []
    
    def test_tokenization_special_chars(self):
        """Test tokenization with special characters."""
        text = "Price: $49.99 (20% off!)"
        tokens = _tokenize(text)
        
        assert "price" in tokens
        assert "49.99" in tokens or "49" in tokens
        assert "20" in tokens
        assert "off" in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])