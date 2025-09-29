"""
Embedding generator for HotpotQA documents.

This module extracts documents from hotpot_documents.json, generates embeddings
for each document, and saves them to a plain file format that can be looked up
by document ID. Supports multiple embedding models (SentenceTransformers, OpenAI, 
Google Gemini) and storage formats.
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
import openai
from google import genai
from dotenv import load_dotenv

@dataclass
class EmbeddingConfig:
    """Base configuration for embedding generation."""
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_provider: str = "gemini"  # "gemini", "openai", "sentence_transformers"
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    batch_size: int = 32
    
    def __post_init__(self):
        """Validate and set embedding provider from environment if not specified."""
        # Load API keys from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            
        # Load from environment if default value is used
        if self.embedding_provider == "gemini":
            env_provider = os.getenv('EMBEDDING_PROVIDER', 'gemini').lower()
            if env_provider in ['gemini', 'openai', 'sentence_transformers']:
                self.embedding_provider = env_provider
        
        # Validate provider
        valid_providers = ['gemini', 'openai', 'sentence_transformers']
        if self.embedding_provider not in valid_providers:
            raise ValueError(f"embedding_provider must be one of {valid_providers}, got: {self.embedding_provider}")
    
    @property
    def use_gemini(self) -> bool:
        """Backward compatibility property."""
        return self.embedding_provider == "gemini"
    
    @property
    def use_openai(self) -> bool:
        """Backward compatibility property."""
        return self.embedding_provider == "openai"
    
    @property
    def use_sentence_transformers(self) -> bool:
        """Check if using SentenceTransformers."""
        return self.embedding_provider == "sentence_transformers"
    
    def create_embedding_service(self) -> 'EmbeddingService':
        """Factory method to create an EmbeddingService instance.
        
        This encapsulates all provider-specific logic and returns a ready-to-use
        embedding service. Other modules should use this instead of directly
        instantiating EmbeddingService.
        
        Returns:
            EmbeddingService configured with this config
        """
        return EmbeddingService(self)
    
    def create_document_embedding_manager(self, 
                                        max_documents: Optional[int] = None,
                                        output_format: str = "pickle") -> 'DocumentEmbeddingManager':
        """Factory method to create a DocumentEmbeddingManager instance.
        
        Args:
            max_documents: Maximum number of documents to process
            output_format: Output format for saved embeddings
            
        Returns:
            DocumentEmbeddingManager configured with this config
        """
        doc_config = DocumentEmbeddingConfig(
            model_name=self.model_name,
            embedding_provider=self.embedding_provider,
            openai_api_key=self.openai_api_key, 
            gemini_api_key=self.gemini_api_key,
            batch_size=self.batch_size,
            max_documents=max_documents,
            output_format=output_format
        )
        return DocumentEmbeddingManager(doc_config)
    
    @classmethod
    def from_args(cls, 
                  model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                  embedding_provider: Optional[str] = None,
                  use_openai: bool = False,  # Backward compatibility
                  use_gemini: bool = False,  # Backward compatibility
                  openai_api_key: Optional[str] = None,
                  gemini_api_key: Optional[str] = None,
                  batch_size: int = 32) -> 'EmbeddingConfig':
        """Create EmbeddingConfig from legacy arguments.
        
        This method provides backward compatibility with the old API where
        modules passed provider-specific arguments.
        
        Args:
            model_name: Name of the embedding model
            embedding_provider: Embedding provider name
            use_openai: [DEPRECATED] Whether to use OpenAI embeddings
            use_gemini: [DEPRECATED] Whether to use Gemini embeddings
            openai_api_key: OpenAI API key
            gemini_api_key: Gemini API key
            batch_size: Batch size for processing
            
        Returns:
            EmbeddingConfig instance
        """
        # Handle backward compatibility
        if embedding_provider is None:
            if use_openai:
                embedding_provider = 'openai'
            elif use_gemini:
                embedding_provider = 'gemini'
            else:
                embedding_provider = 'gemini'  # Default
        
        return cls(
            model_name=model_name,
            embedding_provider=embedding_provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            batch_size=batch_size
        )


@dataclass
class DocumentEmbeddingConfig(EmbeddingConfig):
    """Configuration for document embedding generation and storage."""
    max_documents: Optional[int] = None
    output_format: str = "pickle"  # "pickle", "json", "npz"


class EmbeddingService:
    """Base service for generating embeddings using various models."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding service.
        
        Args:
            config: EmbeddingConfig object with generation parameters
        """
        self.config = config
        self.embedding_dim = None
        self.embedding_model = None
        self.openai_client = None
        self.genai_client = None
        self.gemini_model = "gemini-embedding-001"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""
        if self.config.embedding_provider == "gemini":
            if not self.config.gemini_api_key:
                raise ValueError("Gemini API key is required when using gemini provider")
            self.genai_client = genai.Client(api_key=self.config.gemini_api_key)
            self.embedding_dim = None  # Will be determined from API response
            print(f"Using Google Gemini embeddings with model: {self.gemini_model}")
            print("Embedding dimension will be determined from API response")
        elif self.config.embedding_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required when using openai provider")
            self.openai_client = openai.OpenAI(api_key=self.config.openai_api_key)
            self.embedding_dim = 1536  # text-embedding-3-small
            print(f"Using OpenAI embeddings with model: text-embedding-3-small")
        elif self.config.embedding_provider == "sentence_transformers":
            self.embedding_model = SentenceTransformer(self.config.model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Using SentenceTransformers model: {self.config.model_name}")
            print(f"Embedding dimension: {self.embedding_dim}")
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.config.embedding_provider == "gemini":
            try:
                response = self.genai_client.models.embed_content(
                    model=self.gemini_model,
                    contents=text
                )
                # Access the first embedding from the response
                embedding = np.array(response.embeddings[0].values, dtype=np.float32)
                # Set dimension on first successful response
                if self.embedding_dim is None:
                    self.embedding_dim = len(embedding)
                    print(f"Detected Gemini embedding dimension: {self.embedding_dim}")
                return embedding
            except Exception as e:
                print(f"Failed to get Gemini embedding: {e}")
                # Return zero vector with detected dimension, or fallback to 768
                dim = self.embedding_dim if self.embedding_dim is not None else 768
                return np.zeros(dim, dtype=np.float32)
        elif self.config.embedding_provider == "openai":
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"Failed to get OpenAI embedding: {e}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
        elif self.config.embedding_provider == "sentence_transformers":
            return self.embedding_model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convenience method for embedding a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.get_embedding(query)
    
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Convenience method for embedding multiple queries.
        
        Args:
            queries: List of query texts to embed
            
        Returns:
            Array of embedding vectors with shape (num_queries, embedding_dim)
        """
        return self.get_embeddings_batch(queries)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors with shape (num_texts, embedding_dim)
        """
        if self.config.embedding_provider == "gemini":
            embeddings = []
            batch_size = 100  # Fixed batch size for Gemini API
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.gemini_model,
                        contents=batch
                    )
                    # Process all embeddings from batch response
                    for embedding_obj in response.embeddings:
                        embedding = np.array(embedding_obj.values, dtype=np.float32)
                        # Set dimension on first successful response
                        if self.embedding_dim is None:
                            self.embedding_dim = len(embedding)
                            print(f"Detected Gemini embedding dimension: {self.embedding_dim}")
                        embeddings.append(embedding)
                    
                    print(f"Processed Gemini batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
                except Exception as e:
                    print(f"Failed Gemini batch {i//batch_size + 1}: {e}")
                    # Add zero vectors for failed batch
                    dim = self.embedding_dim if self.embedding_dim is not None else 768
                    batch_embeddings = [np.zeros(dim, dtype=np.float32) for _ in batch]
                    embeddings.extend(batch_embeddings)
            
            return np.vstack(embeddings)
        elif self.config.embedding_provider == "openai":
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    response = self.openai_client.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small"
                    )
                    batch_embeddings = [np.array(data.embedding, dtype=np.float32) 
                                      for data in response.data]
                    embeddings.extend(batch_embeddings)
                    print(f"Processed OpenAI batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                except Exception as e:
                    print(f"Failed batch {i//batch_size + 1}: {e}")
                    # Add zero vectors for failed batch
                    batch_embeddings = [np.zeros(self.embedding_dim, dtype=np.float32) 
                                      for _ in batch]
                    embeddings.extend(batch_embeddings)
            
            return np.vstack(embeddings)
        elif self.config.embedding_provider == "sentence_transformers":
            # Use SentenceTransformers batch encoding
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=True
                ).astype(np.float32)
                embeddings.append(batch_embeddings)
                print(f"Processed SentenceTransformers batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            return np.vstack(embeddings)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension.
        
        Returns:
            Embedding dimension if known, None otherwise
        """
        return self.embedding_dim
    

class DocumentEmbeddingManager(EmbeddingService):
    """Manager for document embeddings with storage and metadata handling."""
    
    def __init__(self, config: DocumentEmbeddingConfig):
        """
        Initialize document embedding manager.
        
        Args:
            config: DocumentEmbeddingConfig object with generation parameters
        """
        super().__init__(config)
        self.config = config  # Override with DocumentEmbeddingConfig
        self.embeddings = {}  # doc_id -> embedding
        self.metadata = {}    # doc_id -> document metadata
    
    def load_documents(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from hotpot_documents.json.
        
        Args:
            json_path: Path to the JSON file containing documents
            
        Returns:
            List of document dictionaries
        """
        try:
            print(f"Loading documents from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Apply document limit if specified
            if self.config.max_documents:
                documents = documents[:self.config.max_documents]
                print(f"Limited to first {len(documents)} documents")
            
            # Validate and normalize document format
            normalized_docs = []
            for doc in documents:
                if not all(key in doc for key in ['id', 'title', 'content']):
                    print(f"Warning: Document missing required fields: {doc.get('id', 'unknown')}")
                    continue
                
                normalized_doc = {
                    'id': str(doc['id']),
                    'title': str(doc['title']),
                    'content': str(doc['content']),
                    'source': doc.get('source', 'hotpotqa')
                }
                normalized_docs.append(normalized_doc)
            
            print(f"Loaded {len(normalized_docs)} valid documents")
            return normalized_docs
            
        except Exception as e:
            print(f"Failed to load documents from {json_path}: {e}")
            return []
    
    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary mapping document_id -> embedding
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Extract texts and document IDs
        texts = [doc['content'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        # Generate embeddings in batches
        embeddings = self.get_embeddings_batch(texts)
        
        # Store embeddings and metadata
        embedding_dict = {}
        metadata_dict = {}
        
        for doc_id, embedding, doc in zip(doc_ids, embeddings, documents):
            embedding_dict[doc_id] = embedding
            metadata_dict[doc_id] = {
                'title': doc['title'],
                'content': doc['content'],
                'source': doc['source']
            }
        
        self.embeddings = embedding_dict
        self.metadata = metadata_dict
        
        print(f"Generated embeddings for {len(embedding_dict)} documents")
        return embedding_dict
    
    def save_embeddings_pickle(self, output_path: str):
        """Save embeddings and metadata to pickle format."""
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'config': {
                'model_name': self.config.model_name,
                'embedding_provider': self.config.embedding_provider,
                'use_openai': self.config.use_openai,  # Backward compatibility
                'use_gemini': self.config.use_gemini,  # Backward compatibility
                'embedding_dim': self.embedding_dim,
                'total_documents': len(self.embeddings)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Embeddings saved to {output_path} (pickle format)")
    
    def save_embeddings_json(self, output_path: str):
        """Save embeddings and metadata to JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        embeddings_json = {
            doc_id: embedding.tolist() 
            for doc_id, embedding in self.embeddings.items()
        }
        
        data = {
            'embeddings': embeddings_json,
            'metadata': self.metadata,
            'config': {
                'model_name': self.config.model_name,
                'embedding_provider': self.config.embedding_provider,
                'use_openai': self.config.use_openai,  # Backward compatibility
                'use_gemini': self.config.use_gemini,  # Backward compatibility
                'embedding_dim': self.embedding_dim,
                'total_documents': len(self.embeddings)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Embeddings saved to {output_path} (JSON format)")
    
    def save_embeddings_npz(self, output_path: str):
        """Save embeddings in NumPy compressed format with separate metadata."""
        # Prepare arrays
        doc_ids = list(self.embeddings.keys())
        embeddings_array = np.vstack([self.embeddings[doc_id] for doc_id in doc_ids])
        
        # Save embeddings as NPZ
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            doc_ids=doc_ids,
            config=np.array([{
                'model_name': self.config.model_name,
                'embedding_provider': self.config.embedding_provider,
                'use_openai': self.config.use_openai,  # Backward compatibility
                'use_gemini': self.config.use_gemini,  # Backward compatibility
                'embedding_dim': self.embedding_dim,
                'total_documents': len(self.embeddings)
            }], dtype=object)
        )
        
        # Save metadata separately
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Embeddings saved to {output_path} (NPZ format)")
        print(f"Metadata saved to {metadata_path}")
    
    def save_embeddings(self, output_path: str):
        """Save embeddings using the configured format."""
        if self.config.output_format == "pickle":
            self.save_embeddings_pickle(output_path)
        elif self.config.output_format == "json":
            self.save_embeddings_json(output_path)
        elif self.config.output_format == "npz":
            self.save_embeddings_npz(output_path)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
    
    def load_embeddings_pickle(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from pickle format."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        config = data['config']
        
        print(f"Loaded {config['total_documents']} embeddings from {input_path}")
        print(f"Model: {config['model_name']}, Dimension: {config['embedding_dim']}")
        
        return self.embeddings
    
    def load_embeddings_json(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from JSON format."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        self.embeddings = {
            doc_id: np.array(embedding, dtype=np.float32)
            for doc_id, embedding in data['embeddings'].items()
        }
        self.metadata = data['metadata']
        config = data['config']
        
        print(f"Loaded {config['total_documents']} embeddings from {input_path}")
        print(f"Model: {config['model_name']}, Dimension: {config['embedding_dim']}")
        
        return self.embeddings
    
    def load_embeddings_npz(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from NPZ format."""
        data = np.load(input_path, allow_pickle=True)
        
        embeddings_array = data['embeddings']
        doc_ids = data['doc_ids']
        config = data['config'].item()
        
        # Reconstruct embeddings dictionary
        self.embeddings = {
            doc_id: embeddings_array[i]
            for i, doc_id in enumerate(doc_ids)
        }
        
        # Load metadata
        metadata_path = input_path.replace('.npz', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print(f"Warning: Metadata file not found: {metadata_path}")
            self.metadata = {}
        
        print(f"Loaded {config['total_documents']} embeddings from {input_path}")
        print(f"Model: {config['model_name']}, Dimension: {config['embedding_dim']}")
        
        return self.embeddings
    
    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from file, auto-detecting format."""
        if input_path.endswith('.pkl') or input_path.endswith('.pickle'):
            return self.load_embeddings_pickle(input_path)
        elif input_path.endswith('.json'):
            return self.load_embeddings_json(input_path)
        elif input_path.endswith('.npz'):
            return self.load_embeddings_npz(input_path)
        else:
            raise ValueError(f"Unsupported file format for {input_path}")
    
    def get_embedding_by_id(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific document ID."""
        return self.embeddings.get(doc_id)
    
    def get_metadata_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document ID."""
        return self.metadata.get(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded embeddings."""
        if not self.embeddings:
            return {}
        
        return {
            'total_documents': len(self.embeddings),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.config.model_name,
            'embedding_provider': self.config.embedding_provider,
            'use_openai': self.config.use_openai,  # Backward compatibility
            'use_gemini': self.config.use_gemini,  # Backward compatibility
            'sample_doc_ids': list(self.embeddings.keys())[:5]
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate embeddings for HotpotQA documents')
    
    parser.add_argument('--input', '-i', 
                       default='hotpot_documents.json',
                       help='Input JSON file path')
    parser.add_argument('--output', '-o',
                       default='hotpot_embeddings.pkl',
                       help='Output file path')
    parser.add_argument('--model', '-m',
                       default='intfloat/multilingual-e5-large-instruct',
                       help='SentenceTransformers model name')
    parser.add_argument('--embedding-provider', choices=['gemini', 'openai', 'sentence_transformers'],
                       default='gemini',
                       help='Embedding provider to use (default: gemini)')
    # Backward compatibility arguments
    parser.add_argument('--use-openai', action='store_true',
                       help='[DEPRECATED] Use --embedding-provider openai instead')
    parser.add_argument('--use-gemini', action='store_true',
                       help='[DEPRECATED] Use --embedding-provider gemini instead')
    parser.add_argument('--openai-key',
                       help='OpenAI API key (or set OPENAI_API_KEY in .env file)')
    parser.add_argument('--gemini-key',
                       help='Google Gemini API key (or set GEMINI_API_KEY in .env file)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--max-docs', type=int,
                       help='Maximum number of documents to process (for testing)')
    parser.add_argument('--format', choices=['pickle', 'json', 'npz'],
                       default='pickle',
                       help='Output format')
    parser.add_argument('--load-test',
                       help='Test loading embeddings from specified file')
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys from command line args or environment variables
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    gemini_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    
    # Determine embedding provider (handle backward compatibility)
    embedding_provider = args.embedding_provider
    if args.use_openai:
        print("Warning: --use-openai is deprecated. Use --embedding-provider openai instead.")
        embedding_provider = 'openai'
    elif args.use_gemini:
        print("Warning: --use-gemini is deprecated. Use --embedding-provider gemini instead.")
        embedding_provider = 'gemini'
    
    # Create configuration
    config = DocumentEmbeddingConfig(
        model_name=args.model,
        embedding_provider=embedding_provider,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
        batch_size=args.batch_size,
        max_documents=args.max_docs,
        output_format=args.format
    )
    
    # Initialize generator
    generator = DocumentEmbeddingManager(config)
    
    # Test loading if specified
    if args.load_test:
        print(f"Testing loading from {args.load_test}...")
        embeddings = generator.load_embeddings(args.load_test)
        stats = generator.get_stats()
        print("Load test successful!")
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Load documents
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    documents = generator.load_documents(args.input)
    if not documents:
        print("No documents loaded, exiting")
        return
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(documents)
    
    # Save embeddings
    generator.save_embeddings(args.output)
    
    # Print statistics
    stats = generator.get_stats()
    print("\nEmbedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nEmbedding generation completed!")
    print(f"Use this file with updated FAISS and Azure AI Search retrievers.")


# Backward compatibility alias
EmbeddingGenerator = DocumentEmbeddingManager


def create_embedding_service(model_name: str = "intfloat/multilingual-e5-large-instruct",
                           embedding_provider: Optional[str] = None,
                           use_openai: bool = False,
                           use_gemini: bool = False,
                           openai_api_key: Optional[str] = None,
                           gemini_api_key: Optional[str] = None,
                           batch_size: int = 32) -> EmbeddingService:
    """Convenience function to create an EmbeddingService for query embedding.
    
    This is a convenience wrapper around EmbeddingConfig.from_args().create_embedding_service()
    that maintains backward compatibility with the old API.
    
    Args:
        model_name: Name of the SentenceTransformers model
        embedding_provider: Embedding provider ('gemini', 'openai', 'sentence_transformers')
        use_openai: [DEPRECATED] Whether to use OpenAI embeddings
        use_gemini: [DEPRECATED] Whether to use Google Gemini embeddings
        openai_api_key: OpenAI API key
        gemini_api_key: Google Gemini API key
        batch_size: Batch size for processing
        
    Returns:
        Configured EmbeddingService instance
    """
    config = EmbeddingConfig.from_args(
        model_name=model_name,
        embedding_provider=embedding_provider,
        use_openai=use_openai,
        use_gemini=use_gemini,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        batch_size=batch_size
    )
    return config.create_embedding_service()


def simple_embedding_service(model_name: str = "intfloat/multilingual-e5-large-instruct") -> EmbeddingService:
    """Create a simple embedding service with default settings.
    
    This creates an embedding service using the default provider (gemini) 
    and API keys from environment variables.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Ready-to-use EmbeddingService instance
    """
    config = EmbeddingConfig(model_name=model_name)
    return config.create_embedding_service()


if __name__ == "__main__":
    main()