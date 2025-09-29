"""
FAISS implementation for HotpotQA evaluation.

Provides functionality to index documents and perform retrieval
using Facebook AI Similarity Search (FAISS) library.
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import faiss
    import openai
    from sentence_transformers import SentenceTransformer
    from embedding_generator import EmbeddingGenerator, EmbeddingConfig
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install faiss-cpu sentence-transformers openai")


@dataclass 
class SearchResult:
    """Represents a search result."""
    doc_id: str
    title: str
    content: str
    score: float


class FAISSRetriever:
    """FAISS implementation for document retrieval."""
    
    @classmethod
    def create_simple(cls, 
                     model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                     precomputed_embeddings_path: Optional[str] = "hotpot_embeddings.pkl") -> 'FAISSRetriever':
        """Create a simple FAISS retriever with default settings.
        
        Args:
            model_name: Name of the embedding model
            precomputed_embeddings_path: Path to pre-computed embeddings
            
        Returns:
            FAISSRetriever instance with default configuration
        """
        config = EmbeddingConfig(model_name=model_name)
        return cls(embedding_config=config, precomputed_embeddings_path=precomputed_embeddings_path)
    
    def __init__(self, 
                 embedding_config: Optional[EmbeddingConfig] = None,
                 precomputed_embeddings_path: Optional[str] = "hotpot_embeddings.pkl"):
        """
        Initialize FAISS retriever.
        
        Args:
            embedding_config: EmbeddingConfig instance (if None, uses default config)
            precomputed_embeddings_path: Path to pre-computed embeddings file
        """
        # Handle embedding configuration
        if embedding_config is None:
            # Create default config - provider details handled by EmbeddingConfig internally
            self.embedding_config = EmbeddingConfig()
        else:
            self.embedding_config = embedding_config
            
        self.embedding_model_name = self.embedding_config.model_name
        self.precomputed_embeddings_path = precomputed_embeddings_path
        
        # Initialize embedding services
        self.embedding_generator = None
        self.embedding_service = None
        
        # Try to load pre-computed embeddings first
        if precomputed_embeddings_path and os.path.exists(precomputed_embeddings_path):
            try:
                self.embedding_generator = self.embedding_config.create_document_embedding_manager()
                self.embedding_generator.load_embeddings(precomputed_embeddings_path)
                self.embedding_dim = list(self.embedding_generator.embeddings.values())[0].shape[0]
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
            self.embedding_dim = self.embedding_service.get_embedding_dimension() or 512
            print(f"Initialized embedding service with model: {self.embedding_model_name}")
        
        # Backward compatibility attributes
        self.embedding_provider = self.embedding_config.embedding_provider
        self.openai_api_key = self.embedding_config.openai_api_key
        self.gemini_api_key = self.embedding_config.gemini_api_key
        self.use_openai = self.embedding_config.use_openai
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.document_metadata = []
        
    def get_embedding(self, text: str, doc_id: Optional[str] = None) -> np.ndarray:
        """Get embedding for text."""
        # Use pre-computed embedding if available and doc_id provided
        if self.embedding_generator and doc_id:
            embedding = self.embedding_generator.get_embedding_by_id(doc_id)
            if embedding is not None:
                return embedding
        
        # Fall back to on-demand generation using embedding service
        if self.embedding_service:
            try:
                return self.embedding_service.get_embedding(text)
            except Exception as e:
                print(f"Failed to get embedding: {e}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
        
        print("No embedding service available")
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        if self.embedding_service:
            try:
                return self.embedding_service.get_embeddings_batch(texts)
            except Exception as e:
                print(f"Failed to get batch embeddings: {e}")
                # Return zero vectors as fallback
                return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        print("No embedding service available")
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
    
    def build_index(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document dictionaries
            save_path: Optional path to save the index
        """
        print(f"Building FAISS index for {len(documents)} documents...")
        
        # Store documents and metadata
        self.documents = documents
        self.document_metadata = [
            {
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content']
            }
            for doc in documents
        ]
        
        # Get embeddings for all documents
        if self.embedding_generator:
            print("Using pre-computed embeddings...")
            embeddings = []
            missing_docs = []
            
            for doc in documents:
                embedding = self.embedding_generator.get_embedding_by_id(doc['id'])
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    missing_docs.append(doc)
                    print(f"Warning: No pre-computed embedding found for document {doc['id']}")
            
            if missing_docs:
                print(f"Generating embeddings for {len(missing_docs)} missing documents...")
                missing_texts = [doc['content'] for doc in missing_docs]
                missing_embeddings = self.get_embeddings_batch(missing_texts)
                embeddings.extend(missing_embeddings)
            
            embeddings = np.vstack(embeddings)
        else:
            # Extract texts for embedding
            texts = [doc['content'] for doc in documents]
            print("Generating embeddings...")
            embeddings = self.get_embeddings_batch(texts)
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} documents")
        
        # Save index if path provided
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, save_path: str):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{save_path}.faiss")
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'embedding_model_name': self.embedding_model_name,
                'embedding_provider': self.embedding_provider,
                'use_openai': self.use_openai,  # Backward compatibility
                'embedding_dim': self.embedding_dim,
                'precomputed_embeddings_path': self.precomputed_embeddings_path
            }
            
            with open(f"{save_path}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"Index saved to {save_path}.faiss and {save_path}.pkl")
            
        except Exception as e:
            print(f"Failed to save index: {e}")
    
    def load_index(self, save_path: str):
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{save_path}.faiss")
            
            # Load metadata
            with open(f"{save_path}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.documents = metadata['documents']
            self.document_metadata = metadata['document_metadata']
            self.embedding_model_name = metadata['embedding_model_name']
            self.embedding_provider = metadata.get('embedding_provider', 'gemini')
            self.use_openai = metadata.get('use_openai', False)  # Backward compatibility
            self.embedding_dim = metadata['embedding_dim']
            self.precomputed_embeddings_path = metadata.get('precomputed_embeddings_path')
            
            # Load pre-computed embeddings if path is available
            if self.precomputed_embeddings_path and os.path.exists(self.precomputed_embeddings_path):
                # Recreate embedding config from loaded metadata
                self.embedding_config = EmbeddingConfig.from_args(
                    model_name=self.embedding_model_name,
                    embedding_provider=self.embedding_provider,
                    use_openai=self.use_openai
                )
                self.embedding_generator = self.embedding_config.create_document_embedding_manager()
                try:
                    self.embedding_generator.load_embeddings(self.precomputed_embeddings_path)
                    print(f"Loaded pre-computed embeddings from {self.precomputed_embeddings_path}")
                except Exception as e:
                    print(f"Failed to load pre-computed embeddings: {e}")
                    self.embedding_generator = None
            
            print(f"Index loaded from {save_path}")
            print(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            print("Index not built or loaded")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Convert to SearchResult objects
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_metadata):
                    doc_metadata = self.document_metadata[idx]
                    result = SearchResult(
                        doc_id=doc_metadata['id'],
                        title=doc_metadata['title'],
                        content=doc_metadata['content'],
                        score=float(score)
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def search_with_embedding(self, query: str, query_embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents using precomputed query embedding.
        
        Args:
            query: Search query (for reference/logging)
            query_embedding: Precomputed query embedding
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            print("Index not built or loaded")
            return []
        
        try:
            # Reshape and normalize the precomputed embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Convert to SearchResult objects
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_metadata):
                    doc_metadata = self.document_metadata[idx]
                    result = SearchResult(
                        doc_id=doc_metadata['id'],
                        title=doc_metadata['title'],
                        content=doc_metadata['content'],
                        score=float(score)
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Search with embedding failed: {e}")
            return []
    
    def load_documents_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from hotpot_documents.json format.
        
        Args:
            json_path: Path to the JSON file containing documents
            
        Returns:
            List of document dictionaries
        """
        try:
            print(f"Loading documents from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Validate and normalize document format
            normalized_docs = []
            for doc in documents:
                # Ensure required fields exist
                if not all(key in doc for key in ['id', 'title', 'content']):
                    print(f"Warning: Document missing required fields: {doc.get('id', 'unknown')}")
                    continue
                
                # Normalize document structure
                normalized_doc = {
                    'id': str(doc['id']),
                    'title': str(doc['title']),
                    'content': str(doc['content']),
                    'source': doc.get('source', 'unknown')
                }
                normalized_docs.append(normalized_doc)
            
            print(f"Loaded and normalized {len(normalized_docs)} documents from JSON file")
            return normalized_docs
            
        except Exception as e:
            print(f"Failed to load documents from {json_path}: {e}")
            return []
    
    def save_documents_to_json(self, documents: List[Dict[str, Any]], json_path: str):
        """
        Save documents to hotpot_documents.json format.
        
        Args:
            documents: List of document dictionaries
            json_path: Path to save the JSON file
        """
        try:
            print(f"Saving {len(documents)} documents to {json_path}...")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            print(f"Documents saved to {json_path}")
            
        except Exception as e:
            print(f"Failed to save documents to {json_path}: {e}")
    
    def create_sample_documents(self, input_json_path: str, output_json_path: str, 
                               sample_size: int = 1000):
        """
        Create a sample subset of documents for testing.
        
        Args:
            input_json_path: Path to the full hotpot_documents.json file
            output_json_path: Path to save the sample documents
            sample_size: Number of documents to sample
        """
        try:
            documents = self.load_documents_from_json(input_json_path)
            
            if len(documents) <= sample_size:
                print(f"Document count ({len(documents)}) is less than or equal to sample size ({sample_size})")
                sample_docs = documents
            else:
                import random
                random.seed(42)  # For reproducible sampling
                sample_docs = random.sample(documents, sample_size)
                print(f"Created sample of {len(sample_docs)} documents from {len(documents)} total")
            
            self.save_documents_to_json(sample_docs, output_json_path)
            return sample_docs
            
        except Exception as e:
            print(f"Failed to create sample documents: {e}")
            return []
    
    def build_index_from_json(self, json_path: str, save_path: Optional[str] = None):
        """
        Build FAISS index from hotpot_documents.json file.
        
        Args:
            json_path: Path to the JSON file containing documents
            save_path: Optional path to save the index
        """
        documents = self.load_documents_from_json(json_path)
        if documents:
            self.build_index(documents, save_path)
        else:
            print("Failed to load documents, index not built")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {}
        
        return {
            'total_documents': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'embedding_model': self.embedding_model_name,
            'embedding_provider': self.embedding_provider,
            'use_openai': self.use_openai,  # Backward compatibility
            'index_type': 'IndexFlatIP'
        }


def main():
    """Main function to test FAISS functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FAISS Retriever for HotpotQA')
    parser.add_argument('--json-file', default='hotpot_documents.json',
                       help='Path to hotpot documents JSON file')
    parser.add_argument('--index-path', default='hotpot_faiss_index',
                       help='Path to save/load FAISS index')
    parser.add_argument('--build-index', action='store_true',
                       help='Build new index from JSON file')
    parser.add_argument('--load-index', action='store_true',
                       help='Load existing index')
    parser.add_argument('--query', type=str,
                       help='Search query to test retrieval')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI embeddings instead of SentenceTransformers')
    parser.add_argument('--openai-key', type=str,
                       help='OpenAI API key')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample subset of documents')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Size of sample to create (default: 1000)')
    parser.add_argument('--sample-output', type=str, default='hotpot_sample.json',
                       help='Output path for sample documents')
    parser.add_argument('--precomputed-embeddings', type=str,
                       default='hotpot_embeddings.pkl',
                       help='Path to pre-computed embeddings file')
    
    args = parser.parse_args()
    
    print("FAISS Retriever for HotpotQA")
    print(f"JSON file: {args.json_file}")
    print(f"Index path: {args.index_path}")
    
    # Initialize retriever with clean interface
    if args.use_openai:
        # Create embedding config explicitly for OpenAI
        config = EmbeddingConfig.from_args(
            use_openai=args.use_openai,
            openai_api_key=args.openai_key
        )
        retriever = FAISSRetriever(
            embedding_config=config,
            precomputed_embeddings_path=args.precomputed_embeddings
        )
    else:
        # Use default config
        retriever = FAISSRetriever(
            precomputed_embeddings_path=args.precomputed_embeddings
        )
    
    # Handle sample creation
    if args.create_sample:
        print(f"Creating sample of {args.sample_size} documents...")
        sample_docs = retriever.create_sample_documents(
            args.json_file, 
            args.sample_output, 
            args.sample_size
        )
        if sample_docs:
            print(f"Sample saved to {args.sample_output}")
        return
    
    if args.build_index:
        # Build index from JSON file
        if os.path.exists(args.json_file):
            retriever.build_index_from_json(args.json_file, args.index_path)
        else:
            print(f"JSON file not found: {args.json_file}")
            return
    
    elif args.load_index:
        # Load existing index
        if os.path.exists(f"{args.index_path}.faiss"):
            retriever.load_index(args.index_path)
        else:
            print(f"Index not found: {args.index_path}.faiss")
            return
    
    else:
        # Default: try to load index, if not exists, build from JSON
        if os.path.exists(f"{args.index_path}.faiss"):
            print("Loading existing index...")
            retriever.load_index(args.index_path)
        elif os.path.exists(args.json_file):
            print("Building new index from JSON file...")
            retriever.build_index_from_json(args.json_file, args.index_path)
        else:
            print(f"Neither index nor JSON file found")
            return
    
    # Print index statistics
    stats = retriever.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search if query provided
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        results = retriever.search(args.query, top_k=args.top_k)
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title} (Score: {result.score:.4f})")
            print(f"   ID: {result.doc_id}")
            print(f"   Content: {result.content[:200]}..." if len(result.content) > 200 else f"   Content: {result.content}")
    
    else:
        print("\nUse --query 'your search term' to test retrieval")
        print("Example: python faiss_retriever.py --query 'machine learning' --top-k 3")


if __name__ == "__main__":
    main()