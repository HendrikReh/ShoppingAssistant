"""Vector search implementation using Qdrant."""

import logging
from typing import List, Tuple, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


class VectorSearch:
    """Vector search using Qdrant vector database."""
    
    def __init__(self, client: Optional[QdrantClient] = None):
        """Initialize vector search.
        
        Args:
            client: Qdrant client instance (creates new if None)
        """
        self.client = client or QdrantClient(host="localhost", port=6333)
    
    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors.
        
        Args:
            collection: Collection name to search
            query_vector: Query vector
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of (id, score, payload) tuples
        """
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append((
                    result.id,
                    result.score,
                    result.payload or {}
                ))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_with_id_mapping(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """Search and map UUIDs to original IDs.
        
        Args:
            collection: Collection name
            query_vector: Query vector
            top_k: Number of results
            
        Returns:
            List of (original_id, score) tuples
        """
        raw_results = self.search(collection, query_vector, top_k)
        
        mapped_results = []
        for uuid, score, payload in raw_results:
            # Extract original ID from payload
            original_id = payload.get('original_id', payload.get('id', ''))
            if original_id:
                mapped_results.append((original_id, score))
        
        return mapped_results
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """Create a new collection.
        
        Args:
            collection_name: Name of collection
            vector_size: Dimension of vectors
            distance: Distance metric
            
        Returns:
            True if successful
        """
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Collection name
            
        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
    
    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """Ensure collection exists, create if not.
        
        Args:
            collection_name: Collection name
            vector_size: Vector dimension
            distance: Distance metric
            
        Returns:
            True if collection exists or was created
        """
        if self.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return True
        
        logger.info(f"Creating collection {collection_name}")
        return self.create_collection(collection_name, vector_size, distance)
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct]
    ) -> bool:
        """Upsert points to collection.
        
        Args:
            collection_name: Collection name
            points: Points to upsert
            
        Returns:
            True if successful
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False


def create_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Create Qdrant client.
    
    Args:
        host: Qdrant host
        port: Qdrant port
        
    Returns:
        QdrantClient instance
    """
    return QdrantClient(host=host, port=port)