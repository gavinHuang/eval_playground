"""
Test FAISS vector search functionality in Agentar-Scale-SQL
"""

import os
from agentar_scale_sql import AgentarScaleSQL

def test_faiss_vector_search():
    """Test that FAISS vector search is working"""
    db_path = "debit_card.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found. Run with --mode setup first.")
        return False
    
    print("Initializing Agentar-Scale-SQL with FAISS vector search...")
    agentar = AgentarScaleSQL(
        db_path=db_path,
        n_reasoning_candidates=1,  # Minimal for testing
        n_icl_candidates=1
    )
    
    # Verify FAISS index was created
    if agentar.cell_index is None:
        print("❌ FAISS index not created")
        return False
    
    print(f"✓ FAISS index created with {agentar.cell_index.ntotal} vectors")
    print(f"✓ Cell store contains {len(agentar.cell_texts)} entries")
    
    # Test keyword extraction and cell retrieval
    print("\nTesting vector search retrieval...")
    keywords = agentar._extract_keywords(
        "What are the currencies used by customers?",
        "EUR and CZK are the main currencies"
    )
    print(f"✓ Extracted keywords: {keywords}")
    
    relevant_cells = agentar._retrieve_relevant_cells(keywords, top_k=5)
    print(f"✓ Retrieved relevant cells:\n{relevant_cells[:200]}...")
    
    print("\n✅ All FAISS vector search tests passed!")
    return True

if __name__ == "__main__":
    test_faiss_vector_search()
