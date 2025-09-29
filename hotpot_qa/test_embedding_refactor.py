#!/usr/bin/env python3
"""
Test script for the refactored embedding generator module.
Tests both EmbeddingService for query embedding and DocumentEmbeddingManager for document processing.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_generator import (
    EmbeddingService, 
    DocumentEmbeddingManager, 
    EmbeddingConfig, 
    DocumentEmbeddingConfig,
    create_embedding_service
)


def test_embedding_service():
    """Test the EmbeddingService for query embedding."""
    print("Testing EmbeddingService...")
    
    # Create a simple config using SentenceTransformers (no API keys needed)
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",  # Small, fast model for testing
        batch_size=2
    )
    
    # Initialize service
    embedding_service = EmbeddingService(config)
    
    # Test single query embedding
    query = "What is the capital of France?"
    embedding = embedding_service.embed_query(query)
    
    print(f"Single query embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedding_service.get_embedding_dimension()}")
    
    # Test batch query embedding
    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is machine learning?"
    ]
    batch_embeddings = embedding_service.embed_queries(queries)
    
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    assert batch_embeddings.shape[0] == len(queries)
    assert batch_embeddings.shape[1] == embedding_service.get_embedding_dimension()
    
    print("✅ EmbeddingService tests passed!")
    return embedding_service


def test_convenience_function():
    """Test the convenience function for creating embedding service."""
    print("\nTesting create_embedding_service convenience function...")
    
    service = create_embedding_service(
        model_name="all-MiniLM-L6-v2",
        batch_size=2
    )
    
    # Test with a simple query
    query = "Test query for convenience function"
    embedding = service.embed_query(query)
    
    print(f"Convenience function embedding shape: {embedding.shape}")
    print("✅ Convenience function test passed!")
    return service


def test_document_embedding_manager():
    """Test the DocumentEmbeddingManager (backward compatibility)."""
    print("\nTesting DocumentEmbeddingManager...")
    
    # Create config
    config = DocumentEmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=2,
        max_documents=3,
        output_format="json"
    )
    
    # Initialize manager
    manager = DocumentEmbeddingManager(config)
    
    # Create some sample documents
    sample_docs = [
        {"id": "1", "title": "Test Doc 1", "content": "This is the first test document."},
        {"id": "2", "title": "Test Doc 2", "content": "This is the second test document."},
        {"id": "3", "title": "Test Doc 3", "content": "This is the third test document."}
    ]
    
    # Test embedding generation
    embeddings = manager.generate_embeddings(sample_docs)
    
    print(f"Generated embeddings for {len(embeddings)} documents")
    print(f"Document IDs: {list(embeddings.keys())}")
    
    # Test individual document retrieval
    doc1_embedding = manager.get_embedding_by_id("1")
    doc1_metadata = manager.get_metadata_by_id("1")
    
    print(f"Doc 1 embedding shape: {doc1_embedding.shape}")
    print(f"Doc 1 metadata: {doc1_metadata}")
    
    # Test stats
    stats = manager.get_stats()
    print(f"Manager stats: {stats}")
    
    print("✅ DocumentEmbeddingManager tests passed!")
    return manager


def test_backward_compatibility():
    """Test backward compatibility with EmbeddingGenerator alias."""
    print("\nTesting backward compatibility...")
    
    # Import the alias
    from embedding_generator import EmbeddingGenerator
    
    # Should be the same as DocumentEmbeddingManager
    config = DocumentEmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=2
    )
    
    generator = EmbeddingGenerator(config)
    
    # Test that it has all the expected methods
    assert hasattr(generator, 'embed_query')
    assert hasattr(generator, 'embed_queries')
    assert hasattr(generator, 'generate_embeddings')
    assert hasattr(generator, 'load_documents')
    
    # Test query embedding
    query = "Backward compatibility test"
    embedding = generator.embed_query(query)
    print(f"Backward compatibility embedding shape: {embedding.shape}")
    
    print("✅ Backward compatibility test passed!")


def main():
    """Run all tests."""
    print("Running refactored embedding generator tests...\n")
    
    try:
        # Test core functionality
        embedding_service = test_embedding_service()
        convenience_service = test_convenience_function()
        document_manager = test_document_embedding_manager()
        test_backward_compatibility()
        
        print("\n🎉 All tests passed! The refactoring is successful.")
        print("\nUsage examples:")
        print("1. For query embedding in evaluation:")
        print("   from embedding_generator import create_embedding_service")
        print("   service = create_embedding_service()")
        print("   embedding = service.embed_query('your query here')")
        print()
        print("2. For document processing (original functionality):")
        print("   from embedding_generator import DocumentEmbeddingManager, DocumentEmbeddingConfig")
        print("   config = DocumentEmbeddingConfig()")
        print("   manager = DocumentEmbeddingManager(config)")
        print("   manager.load_documents('documents.json')")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())