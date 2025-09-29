"""
Azure AI Search implementation for HotpotQA evaluation.

Provides functionality to index documents and perform retrieval
using Azure Cognitive Search service with SentenceTransformers embeddings.
Uses Qwen/Qwen3-Embedding-0.6B model by default for generating embeddings.

Configuration is loaded from environment variables or .env file:
- AZURE_SEARCH_SERVICE_NAME: Azure Search service name
- AZURE_SEARCH_API_KEY: Azure Search API key
- EMBEDDING_MODEL: SentenceTransformer model name (optional, defaults to Qwen/Qwen3-Embedding-0.6B)
- INDEX_NAME: Search index name (optional, defaults to hotpotqa-index)
- PRECOMPUTED_EMBEDDINGS_PATH: Path to precomputed embeddings (optional, defaults to hotpot_embeddings.pkl)
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.models import VectorizedQuery
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        VectorSearchAlgorithmConfiguration,
        HnswAlgorithmConfiguration,
    )
    from azure.core.credentials import AzureKeyCredential
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv
    import numpy as np
    from embedding_generator import EmbeddingGenerator, EmbeddingConfig
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install azure-search-documents sentence-transformers python-dotenv")


@dataclass 
class SearchResult:
    """Represents a search result."""
    doc_id: str
    title: str
    content: str
    score: float
    

class AzureAISearchRetriever:
    """Azure AI Search implementation for document retrieval."""
    
    @classmethod
    def create_simple(cls,
                     search_service_name: str,
                     search_api_key: str,
                     model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                     index_name: str = "hotpotqa-index",
                     precomputed_embeddings_path: Optional[str] = "hotpot_embeddings.pkl") -> 'AzureAISearchRetriever':
        """Create a simple Azure AI Search retriever with default settings.
        
        Args:
            search_service_name: Azure Search service name
            search_api_key: Azure Search API key
            model_name: Name of the embedding model
            index_name: Search index name
            precomputed_embeddings_path: Path to pre-computed embeddings
            
        Returns:
            AzureAISearchRetriever instance with default configuration
        """
        config = EmbeddingConfig(model_name=model_name)
        return cls(
            search_service_name=search_service_name,
            search_api_key=search_api_key,
            embedding_config=config,
            index_name=index_name,
            precomputed_embeddings_path=precomputed_embeddings_path
        )
    
    def __init__(self, 
                 search_service_name: str,
                 search_api_key: str,
                 embedding_config: Optional[EmbeddingConfig] = None,
                 index_name: str = "hotpotqa-index",
                 precomputed_embeddings_path: Optional[str] = "hotpot_embeddings.pkl"):
        """Initialize Azure AI Search client.
        
        Args:
            search_service_name: Azure Search service name
            search_api_key: Azure Search API key
            embedding_config: EmbeddingConfig instance (if None, uses default config)
            index_name: Search index name
            precomputed_embeddings_path: Path to pre-computed embeddings
        """
        self.search_service_name = search_service_name
        self.search_api_key = search_api_key
        self.index_name = index_name
        self.precomputed_embeddings_path = precomputed_embeddings_path
        
        # Handle embedding configuration
        if embedding_config is None:
            # Create default config - provider details handled internally
            self.embedding_config = EmbeddingConfig()
        else:
            self.embedding_config = embedding_config
            
        self.embedding_model = self.embedding_config.model_name
        
        # Initialize clients
        endpoint = f"https://{search_service_name}.search.windows.net"
        credential = AzureKeyCredential(search_api_key)
        
        self.search_client = SearchClient(endpoint, index_name, credential)
        self.index_client = SearchIndexClient(endpoint, credential)
        self.embedding_service = None
        self.embedding_generator = None
        
        # Try to load pre-computed embeddings first
        if precomputed_embeddings_path and os.path.exists(precomputed_embeddings_path):
            try:
                self.embedding_generator = self.embedding_config.create_document_embedding_manager()
                self.embedding_generator.load_embeddings(precomputed_embeddings_path)
                print(f"Loaded pre-computed embeddings from {precomputed_embeddings_path}")
            except Exception as e:
                print(f"Failed to load pre-computed embeddings: {e}")
                print("Falling back to on-demand embedding generation")
                self.embedding_generator = None
        
        # Initialize on-demand embedding service if no pre-computed embeddings
        if not self.embedding_generator:
            if precomputed_embeddings_path and not os.path.exists(precomputed_embeddings_path):
                print(f"Warning: Precomputed embeddings file not found at {precomputed_embeddings_path}")
                print("Falling back to on-demand embedding generation")
            self.embedding_service = self.embedding_config.create_embedding_service()
            print(f"Initialized embedding service with model: {embedding_model}")
            
        # Keep backward compatibility attributes
        self.encoder = None
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        if self.embedding_generator:
            # Get dimension from pre-computed embeddings
            if self.embedding_generator.embeddings:
                sample_embedding = next(iter(self.embedding_generator.embeddings.values()))
                return sample_embedding.shape[0]
            return self.embedding_generator.embedding_dim
        elif self.embedding_service:
            return self.embedding_service.get_embedding_dimension() or 512
        else:
            # Default fallback - this shouldn't happen in normal usage
            return 512
    
    def create_index(self):
        """Create the search index with vector search capabilities."""
        
        # Define the fields for the search index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.get_embedding_dimension(),
                vector_search_profile_name="my-vector-profile",
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-algorithms-config",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="my-algorithms-config")
            ],
        )
        
        # Create the search index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            result = self.index_client.create_or_update_index(index)
            print(f"Index '{self.index_name}' created successfully")
            return result
        except Exception as e:
            print(f"Failed to create index: {e}")
            raise
    
    def get_embedding(self, text: str, doc_id: Optional[str] = None) -> List[float]:
        """Get embedding for text."""
        # Use pre-computed embedding if available and doc_id provided
        if self.embedding_generator and doc_id:
            embedding = self.embedding_generator.get_embedding_by_id(doc_id)
            if embedding is not None:
                return embedding.tolist()
        
        # Fall back to on-demand generation using embedding service
        if self.embedding_service:
            try:
                embedding = self.embedding_service.get_embedding(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Failed to get embedding: {e}")
                return []
        
        print("No embedding service available")
        return []
    
    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Index documents in Azure AI Search with embeddings."""
        
        print(f"Indexing {len(documents)} documents...")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            search_docs = []
            
            for doc in batch:
                # Get embedding for content (try pre-computed first)
                content_vector = self.get_embedding(doc['content'], doc['id'])
                if not content_vector:
                    print(f"Skipping document {doc['id']} - failed to get embedding")
                    continue
                
                search_doc = {
                    'id': doc['id'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'source': doc.get('source', 'hotpotqa'),
                    'content_vector': content_vector
                }
                search_docs.append(search_doc)
            
            # Upload batch
            try:
                result = self.search_client.upload_documents(documents=search_docs)
                print(f"Indexed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                # Add small delay to avoid throttling
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to index batch: {e}")
                print(f"Error details: {str(e)}")
                # If error is about invalid keys, suggest solutions
                if "InvalidDocumentKey" in str(e):
                    print("Document keys contain invalid characters.")
                    print("Solution 1: Reprocess documents with --force to get sanitized IDs")
                    print("Solution 2: Check Azure Search service configuration for allowUnsafeKeys")
                continue
        
        print("Document indexing completed")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               use_semantic_search: bool = True,
               use_vector_search: bool = True) -> List[SearchResult]:
        """
        Search for documents using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_semantic_search: Whether to use semantic search
            use_vector_search: Whether to use vector search
            
        Returns:
            List of SearchResult objects
        """
        
        try:
            search_params = {
                'search_text': query if use_semantic_search else None,
                'top': top_k,
                'select': ['id', 'title', 'content']
            }
            
            # Add vector search if enabled
            if use_vector_search:
                query_vector = self.get_embedding(query)  # No doc_id for queries
                if query_vector:
                    vector_query = VectorizedQuery(
                        vector=query_vector,
                        k_nearest_neighbors=top_k,
                        fields="content_vector"
                    )
                    search_params['vector_queries'] = [vector_query]
            
            # Perform search
            results = self.search_client.search(**search_params)
            
            search_results = []
            for result in results:
                search_result = SearchResult(
                    doc_id=result['id'],
                    title=result['title'],
                    content=result['content'],
                    score=result.get('@search.score', 0.0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def search_with_embedding(self, 
                             query: str, 
                             query_embedding: np.ndarray,
                             top_k: int = 10,
                             use_semantic_search: bool = True) -> List[SearchResult]:
        """
        Search for documents using precomputed query embedding.
        
        Args:
            query: Search query (for semantic search and reference)
            query_embedding: Precomputed query embedding
            top_k: Number of results to return
            use_semantic_search: Whether to use semantic search in addition to vector search
            
        Returns:
            List of SearchResult objects
        """
        
        try:
            search_params = {
                'search_text': query if use_semantic_search else None,
                'top': top_k,
                'select': ['id', 'title', 'content']
            }
            
            # Use precomputed embedding for vector search
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            if query_vector:
                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector"
                )
                search_params['vector_queries'] = [vector_query]
            
            # Perform search
            results = self.search_client.search(**search_params)
            
            search_results = []
            for result in results:
                search_result = SearchResult(
                    doc_id=result['id'],
                    title=result['title'],
                    content=result['content'],
                    score=result.get('@search.score', 0.0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Search Azure with embedding failed: {e}")
            return []
    
    def index_exists(self) -> bool:
        """Check if the search index exists."""
        try:
            self.index_client.get_index(self.index_name)
            return True
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        try:
            results = self.search_client.search(search_text="*", top=1, include_total_count=True)
            return results.get_count() or 0
        except Exception as e:
            print(f"Failed to get document count: {e}")
            return 0
    
    def delete_index(self):
        """Delete the search index."""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted")
        except Exception as e:
            print(f"Failed to delete index: {e}")


def main():
    """Main function to test Azure AI Search functionality."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    search_service = os.getenv('AZURE_SEARCH_SERVICE_NAME')
    search_key = os.getenv('AZURE_SEARCH_API_KEY')
    
    if not all([search_service, search_key]):
        print("Missing required environment variables:")
        print("- AZURE_SEARCH_SERVICE_NAME")
        print("- AZURE_SEARCH_API_KEY")
        print("\nPlease create a .env file with your configuration.")
        print("See .env.example for a template.")
        return
    
    # Get optional parameters from environment with defaults
    embedding_model = os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')
    index_name = os.getenv('INDEX_NAME', 'hotpotqa-index')
    precomputed_embeddings = os.getenv('PRECOMPUTED_EMBEDDINGS_PATH', 'hotpot_embeddings.pkl')
    
    # Initialize retriever
    retriever = AzureAISearchRetriever(
        search_service_name=search_service,
        search_api_key=search_key,
        embedding_model=embedding_model,
        index_name=index_name,
        precomputed_embeddings_path=precomputed_embeddings
    )
    
    # Example usage
    print("Azure AI Search Retriever initialized")
    print("Use this class in your evaluation pipeline")


if __name__ == "__main__":
    main()