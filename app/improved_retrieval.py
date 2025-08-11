"""Improved retrieval pipeline with query preprocessing, expansion, and fallback strategies."""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import logging

# NLTK imports are now optional - only loaded if needed
logger = logging.getLogger(__name__)

def _ensure_nltk_data():
    """Ensure NLTK data is available, download if needed."""
    try:
        import nltk
        nltk.data.find('wordnet')
        nltk.data.find('averaged_perceptron_tagger')
        nltk.data.find('punkt')
        return True
    except (ImportError, LookupError):
        logger.warning("NLTK data not available. Synonym expansion disabled.")
        return False


@dataclass
class RetrievalConfig:
    """Configuration for improved retrieval."""
    top_k: int = 30
    rrf_k: int = 100
    use_query_expansion: bool = True
    use_spelling_correction: bool = True
    use_synonyms: bool = True
    use_fallback: bool = True
    min_relevance_score: float = 0.3
    enable_reranking: bool = True
    
    
@dataclass
class QueryAnalysis:
    """Analysis of user query."""
    original: str
    preprocessed: str
    expanded_terms: List[str]
    intent: str  # search, comparison, recommendation, technical
    entities: Dict[str, List[str]]  # brand, product_type, features
    confidence: float


class QueryPreprocessor:
    """Advanced query preprocessing and expansion."""
    
    def __init__(self):
        self.lemmatizer = None
        if _ensure_nltk_data():
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                pass
        
        # Common misspellings in e-commerce
        self.spelling_corrections = {
            "airpods": "airpods",
            "air pods": "airpods",
            "firetv": "fire tv",
            "fire-tv": "fire tv",
            "usbc": "usb-c",
            "usb c": "usb-c",
            "iphone": "iphone",
            "i phone": "iphone",
            "bluetooth": "bluetooth",
            "bluethooth": "bluetooth",
            "wirless": "wireless",
            "speker": "speaker",
            "keybord": "keyboard",
            "mose": "mouse",
            "moniter": "monitor",
            "leptop": "laptop",
            "teblet": "tablet",
            "hedphones": "headphones",
            "ereader": "e-reader",
            "ebook": "e-book",
        }
        
        # Product type synonyms and related terms
        self.product_synonyms = {
            "laptop": ["notebook", "computer", "pc", "macbook", "chromebook"],
            "earbuds": ["earphones", "in-ear", "airpods", "buds", "wireless earbuds"],
            "headphones": ["headset", "over-ear", "on-ear", "cans"],
            "tv": ["television", "smart tv", "display", "screen"],
            "tablet": ["ipad", "tab", "slate"],
            "phone": ["smartphone", "mobile", "cell phone", "iphone", "android"],
            "cable": ["cord", "wire", "connector", "charger cable"],
            "speaker": ["audio", "sound system", "bluetooth speaker"],
            "router": ["wifi", "wireless router", "modem", "gateway"],
            "keyboard": ["keys", "typing", "mechanical keyboard"],
            "mouse": ["mice", "pointer", "trackpad"],
            "monitor": ["display", "screen", "lcd", "led monitor"],
            "streaming": ["streaming device", "media player", "fire tv", "roku", "chromecast"],
        }
        
        # Brand variations
        self.brand_variations = {
            "amazon": ["amazon", "amazon basics", "amazonbasics"],
            "apple": ["apple", "iphone", "ipad", "macbook", "airpods"],
            "samsung": ["samsung", "galaxy"],
            "google": ["google", "pixel", "chromecast", "nest"],
            "microsoft": ["microsoft", "surface", "xbox"],
            "sony": ["sony", "playstation", "ps5", "ps4"],
            "bose": ["bose"],
            "jbl": ["jbl"],
            "logitech": ["logitech", "logi"],
            "anker": ["anker"],
        }
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis."""
        
        # Basic preprocessing
        preprocessed = self._preprocess(query)
        
        # Detect intent
        intent = self._detect_intent(preprocessed)
        
        # Extract entities
        entities = self._extract_entities(preprocessed)
        
        # Generate expanded terms
        expanded_terms = self._expand_query(preprocessed, entities)
        
        # Calculate confidence
        confidence = self._calculate_confidence(preprocessed, entities, expanded_terms)
        
        return QueryAnalysis(
            original=query,
            preprocessed=preprocessed,
            expanded_terms=expanded_terms,
            intent=intent,
            entities=entities,
            confidence=confidence
        )
    
    def _preprocess(self, query: str) -> str:
        """Basic query preprocessing."""
        # Lowercase
        query = query.lower().strip()
        
        # Apply spelling corrections
        for wrong, correct in self.spelling_corrections.items():
            query = re.sub(r'\b' + wrong + r'\b', correct, query)
        
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent."""
        
        # Comparison intent
        if any(word in query for word in ["compare", "vs", "versus", "or", "better", "difference"]):
            return "comparison"
        
        # Recommendation intent
        if any(word in query for word in ["best", "recommend", "top", "good", "cheap", "budget", "premium"]):
            return "recommendation"
        
        # Technical intent
        if any(word in query for word in ["specs", "specifications", "features", "compatible", "support", "work with"]):
            return "technical"
        
        # Problem-solving intent
        if any(word in query for word in ["fix", "problem", "issue", "not working", "broken", "help"]):
            return "problem_solving"
        
        # Default to search
        return "search"
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query."""
        entities = {
            "brands": [],
            "product_types": [],
            "features": [],
            "modifiers": []
        }
        
        # Extract brands
        for brand, variations in self.brand_variations.items():
            if any(var in query for var in variations):
                entities["brands"].append(brand)
        
        # Extract product types
        for product, synonyms in self.product_synonyms.items():
            if product in query or any(syn in query for syn in synonyms):
                entities["product_types"].append(product)
        
        # Extract features
        feature_keywords = ["wireless", "bluetooth", "usb", "hdmi", "4k", "hd", "portable", 
                          "waterproof", "noise cancelling", "fast charging", "rgb", "mechanical"]
        for feature in feature_keywords:
            if feature in query:
                entities["features"].append(feature)
        
        # Extract modifiers
        modifier_keywords = ["cheap", "budget", "premium", "best", "new", "latest", "small", "large", "mini"]
        for modifier in modifier_keywords:
            if modifier in query:
                entities["modifiers"].append(modifier)
        
        return entities
    
    def _expand_query(self, query: str, entities: Dict[str, List[str]]) -> List[str]:
        """Expand query with synonyms and related terms."""
        expanded = [query]
        
        # Add product type synonyms
        for product_type in entities["product_types"]:
            if product_type in self.product_synonyms:
                expanded.extend(self.product_synonyms[product_type][:3])  # Limit expansion
        
        # Add WordNet synonyms for key terms
        tokens = query.split()
        # Only use WordNet if available
        if _ensure_nltk_data():
            try:
                from nltk.corpus import wordnet
                for token in tokens:
                    if len(token) > 3:  # Skip short words
                        synsets = wordnet.synsets(token)[:2]  # Limit to 2 synsets
                        for synset in synsets:
                            for lemma in synset.lemmas()[:2]:  # Limit lemmas
                                synonym = lemma.name().replace('_', ' ')
                                if synonym != token and synonym not in expanded:
                                    expanded.append(synonym)
            except ImportError:
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        expanded_unique = []
        for term in expanded:
            if term not in seen:
                seen.add(term)
                expanded_unique.append(term)
        
        return expanded_unique[:5]  # Limit total expansion
    
    def _calculate_confidence(self, query: str, entities: Dict, expanded: List) -> float:
        """Calculate confidence in query understanding."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for detected entities
        if entities["brands"]:
            confidence += 0.15
        if entities["product_types"]:
            confidence += 0.15
        if entities["features"]:
            confidence += 0.1
        
        # Increase confidence for successful expansion
        if len(expanded) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)


class ImprovedRetriever:
    """Enhanced retrieval with multiple strategies."""
    
    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer,
        bm25: Optional[BM25Okapi] = None,
        config: Optional[RetrievalConfig] = None
    ):
        self.client = client
        self.model = model
        self.bm25 = bm25
        self.config = config or RetrievalConfig()
        self.preprocessor = QueryPreprocessor()
        
    def retrieve(
        self,
        query: str,
        collection_name: str = "products_minilm",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Main retrieval method with all improvements."""
        
        top_k = top_k or self.config.top_k
        
        # Analyze query
        analysis = self.preprocessor.analyze_query(query)
        logger.info(f"Query analysis: intent={analysis.intent}, confidence={analysis.confidence:.2f}")
        
        # Try primary retrieval
        results = self._primary_retrieval(analysis, collection_name, top_k)
        
        # Check quality and apply fallback if needed
        if self.config.use_fallback:
            quality_score = self._assess_result_quality(results, analysis)
            
            if quality_score < self.config.min_relevance_score:
                logger.info(f"Low quality score ({quality_score:.2f}), applying fallback strategies")
                results = self._fallback_retrieval(analysis, collection_name, top_k, results)
        
        # Score and rank results
        results = self._score_and_rank(results, analysis)
        
        return results[:top_k]
    
    def _primary_retrieval(
        self,
        analysis: QueryAnalysis,
        collection_name: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Primary retrieval using hybrid search."""
        
        all_results = []
        
        # Vector search with original query
        vec_results = self._vector_search(
            analysis.preprocessed,
            collection_name,
            top_k * 2  # Retrieve more for fusion
        )
        all_results.extend(vec_results)
        
        # Vector search with expanded queries if enabled
        if self.config.use_query_expansion and analysis.expanded_terms:
            for expanded_query in analysis.expanded_terms[:2]:  # Limit expansion
                exp_results = self._vector_search(
                    expanded_query,
                    collection_name,
                    top_k
                )
                all_results.extend(exp_results)
        
        # BM25 search if available
        if self.bm25:
            bm25_results = self._bm25_search(analysis.preprocessed, top_k)
            all_results.extend(bm25_results)
        
        # Deduplicate and merge
        merged = self._merge_results(all_results)
        
        return merged
    
    def _vector_search(
        self,
        query: str,
        collection_name: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform vector search."""
        
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vec.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        return [
            {
                "id": hit.payload.get("original_id", str(hit.id)),
                "score": hit.score,
                "payload": hit.payload,
                "source": "vector"
            }
            for hit in results
        ]
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search (placeholder - needs actual implementation)."""
        # This would need actual BM25 index
        return []
    
    def _fallback_retrieval(
        self,
        analysis: QueryAnalysis,
        collection_name: str,
        top_k: int,
        initial_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Fallback strategies for poor initial results."""
        
        fallback_results = initial_results.copy()
        
        # Strategy 1: Broaden search with individual terms
        if len(analysis.preprocessed.split()) > 1:
            for term in analysis.preprocessed.split():
                if len(term) > 3:  # Skip short terms
                    term_results = self._vector_search(term, collection_name, top_k // 2)
                    fallback_results.extend(term_results)
        
        # Strategy 2: Use only product type if detected
        if analysis.entities["product_types"]:
            for product_type in analysis.entities["product_types"]:
                type_results = self._vector_search(product_type, collection_name, top_k)
                fallback_results.extend(type_results)
        
        # Strategy 3: Use only brand if detected
        if analysis.entities["brands"]:
            for brand in analysis.entities["brands"]:
                brand_results = self._vector_search(brand, collection_name, top_k // 2)
                fallback_results.extend(brand_results)
        
        # Merge and deduplicate
        merged = self._merge_results(fallback_results)
        
        return merged
    
    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results using RRF."""
        
        # Group by ID
        id_to_results = {}
        for result in results:
            result_id = result["id"]
            if result_id not in id_to_results:
                id_to_results[result_id] = []
            id_to_results[result_id].append(result)
        
        # Apply Reciprocal Rank Fusion
        merged = []
        for result_id, result_list in id_to_results.items():
            # Calculate RRF score
            rrf_score = sum(1.0 / (self.config.rrf_k + r["score"]) for r in result_list)
            
            # Use the first result's payload
            merged.append({
                "id": result_id,
                "score": rrf_score,
                "payload": result_list[0]["payload"],
                "sources": list(set(r.get("source", "unknown") for r in result_list))
            })
        
        # Sort by score
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged
    
    def _assess_result_quality(
        self,
        results: List[Dict],
        analysis: QueryAnalysis
    ) -> float:
        """Assess quality of retrieval results."""
        
        if not results:
            return 0.0
        
        quality_scores = []
        
        for result in results[:5]:  # Check top 5
            score = 0.0
            title = result["payload"].get("title", "").lower()
            
            # Check for query terms
            query_terms = analysis.preprocessed.lower().split()
            matches = sum(1 for term in query_terms if term in title)
            score += (matches / len(query_terms)) * 0.5 if query_terms else 0
            
            # Check for entities
            if analysis.entities["brands"]:
                if any(brand in title for brand in analysis.entities["brands"]):
                    score += 0.2
            
            if analysis.entities["product_types"]:
                if any(ptype in title for ptype in analysis.entities["product_types"]):
                    score += 0.2
            
            # Check result confidence (vector similarity)
            if result.get("score", 0) > 0.7:
                score += 0.1
            
            quality_scores.append(score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _score_and_rank(
        self,
        results: List[Dict],
        analysis: QueryAnalysis
    ) -> List[Dict]:
        """Final scoring and ranking of results."""
        
        for result in results:
            # Base score from retrieval
            final_score = result["score"]
            
            # Boost for exact matches
            title = result["payload"].get("title", "").lower()
            if analysis.preprocessed.lower() in title:
                final_score *= 1.5
            
            # Boost for brand matches
            if analysis.entities["brands"]:
                if any(brand in title for brand in analysis.entities["brands"]):
                    final_score *= 1.2
            
            # Boost for high ratings
            rating = result["payload"].get("average_rating", 0)
            if rating > 4.5:
                final_score *= 1.1
            
            # Boost for popular products
            num_reviews = result["payload"].get("num_reviews", 0)
            if num_reviews > 1000:
                final_score *= 1.05
            
            result["final_score"] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return results


def create_improved_retriever(
    host: str = "localhost",
    port: int = 6333,
    collection: str = "products_minilm",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> ImprovedRetriever:
    """Factory function to create improved retriever."""
    
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(model_name)
    config = RetrievalConfig()
    
    return ImprovedRetriever(client, model, None, config)


# Example usage
if __name__ == "__main__":
    retriever = create_improved_retriever()
    
    # Test queries
    test_queries = [
        "fire tv stick",
        "wirless earbud",  # Misspelling
        "best budget laptop",
        "usb c cable fast charging",
        "compare echo dot and google home"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        results = retriever.retrieve(query, top_k=5)
        
        for i, result in enumerate(results, 1):
            title = result["payload"].get("title", "Unknown")[:60]
            score = result.get("final_score", result["score"])
            sources = result.get("sources", ["unknown"])
            print(f"{i}. [{score:.3f}] {title}... (via: {', '.join(sources)})")