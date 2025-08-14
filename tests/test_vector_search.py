#!/usr/bin/env python
"""Test that vector search fix works correctly."""

from app.cli import (
    _load_st_model, _qdrant_client, _vector_search,
    _bm25_from_files, EMBED_MODEL, COLLECTION_PRODUCTS,
    DATA_PRODUCTS, DATA_REVIEWS
)

def test_vector_search_with_fix():
    """Test the fixed vector search mapping."""
    print("Testing Vector Search ID Mapping Fix")
    print("=" * 50)
    
    # Load components
    device = "cpu"
    st_model = _load_st_model(EMBED_MODEL, device=device)
    client = _qdrant_client()
    _, _, id_to_product, _ = _bm25_from_files(DATA_PRODUCTS, DATA_REVIEWS)
    
    # Test query
    query = "wireless earbuds"
    print(f"Query: '{query}'")
    
    # Generate embedding
    q_vec = st_model.encode([query], batch_size=1, normalize_embeddings=True,
                           device=device, convert_to_numpy=True)[0].tolist()
    
    # Get raw vector search results
    prod_vec_raw = _vector_search(client, COLLECTION_PRODUCTS, q_vec, top_k=5)
    
    print("\n--- Raw Vector Search Results (UUIDs) ---")
    for uuid, score, payload in prod_vec_raw[:3]:
        print(f"UUID: {uuid}")
        print(f"  Score: {score:.4f}")
        print(f"  original_id: {payload.get('original_id', 'MISSING')}")
        print(f"  Title: {payload.get('title', 'N/A')[:60]}...")
    
    # Apply the fix - map UUIDs to original IDs
    print("\n--- After Mapping to Original IDs ---")
    prod_vec = []
    successful_mappings = 0
    for uuid, score, payload in prod_vec_raw:
        original_id = payload.get('original_id', payload.get('id', ''))
        if original_id:
            prod_vec.append((original_id, score))
            if original_id in id_to_product:
                successful_mappings += 1
                print(f"✓ {original_id} -> Found in id_to_product dict")
            else:
                print(f"✗ {original_id} -> NOT found in dict")
    
    print("\nSummary:")
    print(f"  Total vector results: {len(prod_vec_raw)}")
    print(f"  Successfully mapped: {successful_mappings}")
    print(f"  Mapping success rate: {successful_mappings/len(prod_vec_raw)*100:.1f}%")
    
    if successful_mappings == len(prod_vec_raw):
        print("\n✅ FIX SUCCESSFUL! All vector search results now map correctly.")
    else:
        print("\n⚠️  PARTIAL SUCCESS. Some IDs still don't map correctly.")
    
    return successful_mappings == len(prod_vec_raw)

if __name__ == "__main__":
    success = test_vector_search_with_fix()
    exit(0 if success else 1)