"""
Evaluation module for comparing Azure AI Search and FAISS retrieval performance.

Runs HotpotQA queries against both systems and collects comprehensive metrics.
"""

import json
import time
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from document_processor import DocumentProcessor
from azure_ai_search import AzureAISearchRetriever
from faiss_retriever import FAISSRetriever
from embedding_generator import EmbeddingService, EmbeddingConfig


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a retrieval system."""
    # Relevance metrics  
    precision_at_k: List[float]  # Precision at different k values
    recall_at_k: List[float]     # Recall at different k values
    f1_at_k: List[float]         # F1 score at different k values
    map_score: float             # Mean Average Precision
    ndcg_at_k: List[float]       # NDCG at different k values
    
    # Performance metrics
    avg_query_time: float        # Average query response time
    queries_per_second: float    # Throughput 
    total_queries: int           # Total queries processed
    successful_queries: int      # Queries that returned results
    
    # System-specific metrics
    avg_result_count: float      # Average number of results returned
    system_name: str             # Name of the retrieval system


@dataclass  
class QueryResult:
    """Stores results for a single query."""
    query_id: str
    query_text: str
    retrieved_docs: List[Dict[str, Any]]
    response_time: float
    relevant_docs: List[str]  # Ground truth relevant document titles
    success: bool


class QueryEmbeddingCache:
    """Manages caching of query embeddings to speed up evaluation runs."""
    
    def __init__(self, cache_file: str, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize query embedding cache.
        
        Args:
            cache_file: Path to cache file for storing embeddings
            embedding_service: EmbeddingService for generating new embeddings
        """
        self.cache_file = cache_file
        self.embedding_service = embedding_service
        self.cache = {}  # query_text -> embedding
        self.metadata = {}  # Stores cache metadata
        
        # Try to load existing cache
        self.load_cache()
    
    def load_cache(self) -> bool:
        """Load query embeddings from cache file.
        
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not os.path.exists(self.cache_file):
            print(f"Cache file not found: {self.cache_file}")
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.cache = data.get('embeddings', {})
            self.metadata = data.get('metadata', {})
            
            print(f"Loaded {len(self.cache)} query embeddings from cache")
            if 'model_name' in self.metadata:
                print(f"Cache model: {self.metadata['model_name']}")
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def save_cache(self):
        """Save query embeddings to cache file."""
        try:
            # Update metadata
            if self.embedding_service:
                self.metadata.update({
                    'model_name': self.embedding_service.config.model_name,
                    'embedding_provider': self.embedding_service.config.embedding_provider,
                    'embedding_dim': self.embedding_service.get_embedding_dimension(),
                    'total_queries': len(self.cache),
                    'last_updated': time.time()
                })
            
            data = {
                'embeddings': self.cache,
                'metadata': self.metadata
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.', exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Saved {len(self.cache)} query embeddings to cache")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def get_embedding(self, query_text: str) -> Optional[np.ndarray]:
        """Get embedding for query, either from cache or by generating new one.
        
        Args:
            query_text: Query text to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        # Check cache first
        if query_text in self.cache:
            return self.cache[query_text]
        
        # Generate new embedding if service available
        if self.embedding_service is None:
            print(f"Warning: No embedding service available for query: {query_text[:50]}...")
            return None
        
        try:
            embedding = self.embedding_service.embed_query(query_text)
            self.cache[query_text] = embedding
            return embedding
        except Exception as e:
            print(f"Failed to generate embedding for query: {e}")
            return None
    
    def get_embeddings_batch(self, query_texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Get embeddings for multiple queries efficiently.
        
        Args:
            query_texts: List of query texts
            
        Returns:
            Dictionary mapping query_text -> embedding (or None if failed)
        """
        results = {}
        missing_queries = []
        
        # Check cache for existing embeddings
        for query_text in query_texts:
            if query_text in self.cache:
                results[query_text] = self.cache[query_text]
            else:
                missing_queries.append(query_text)
                results[query_text] = None  # Placeholder
        
        # Generate missing embeddings in batch if service available
        if missing_queries and self.embedding_service:
            try:
                print(f"Generating embeddings for {len(missing_queries)} new queries...")
                embeddings = self.embedding_service.embed_queries(missing_queries)
                
                # Update cache and results
                for i, query_text in enumerate(missing_queries):
                    self.cache[query_text] = embeddings[i]
                    results[query_text] = embeddings[i]
                    
            except Exception as e:
                print(f"Failed to generate batch embeddings: {e}")
        
        return results
    
    def precompute_dataset_embeddings(self, dataset: List[Dict[str, Any]], force_regenerate: bool = False):
        """Precompute embeddings for all queries in dataset.
        
        Args:
            dataset: HotpotQA dataset
            force_regenerate: Whether to regenerate existing embeddings
        """
        if self.embedding_service is None:
            print("Error: No embedding service available for precomputation")
            return
        
        # Extract queries
        queries = [record.get('question', '') for record in dataset]
        queries = [q for q in queries if q]  # Filter empty queries
        
        if force_regenerate:
            # Clear existing cache for these queries
            for query in queries:
                if query in self.cache:
                    del self.cache[query]
        
        # Get embeddings (will generate missing ones)
        print(f"Precomputing embeddings for {len(queries)} queries...")
        start_time = time.time()
        
        self.get_embeddings_batch(queries)
        
        elapsed = time.time() - start_time
        print(f"Precomputation completed in {elapsed:.2f} seconds")
        
        # Save updated cache
        self.save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the query cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_queries': len(self.cache),
            'cache_file': self.cache_file,
            'file_exists': os.path.exists(self.cache_file)
        }
        
        if self.metadata:
            stats.update(self.metadata)
        
        if self.cache:
            # Get sample queries
            sample_queries = list(self.cache.keys())[:3]
            stats['sample_queries'] = sample_queries
            
            # Get embedding dimension from first embedding
            first_embedding = next(iter(self.cache.values()))
            if isinstance(first_embedding, np.ndarray):
                stats['embedding_dimension'] = first_embedding.shape[0]
        
        return stats


class HotpotQAEvaluator:
    """Evaluates retrieval systems on HotpotQA dataset."""
    
    def __init__(self, dataset_path: str, k_values: List[int] = [1, 3, 5, 10], 
                 query_cache_file: Optional[str] = None,
                 embedding_config: Optional[EmbeddingConfig] = None):
        """
        Initialize evaluator.
        
        Args:
            dataset_path: Path to HotpotQA dataset
            k_values: List of k values for precision@k, recall@k metrics
            query_cache_file: Path to query embedding cache file
            embedding_config: Configuration for embedding generation
        """
        self.dataset_path = dataset_path
        self.k_values = k_values
        self.dataset = self._load_dataset()
        
        # Initialize query embedding cache if requested
        self.query_cache = None
        if query_cache_file:
            embedding_service = None
            if embedding_config:
                embedding_service = embedding_config.create_embedding_service()
            self.query_cache = QueryEmbeddingCache(query_cache_file, embedding_service)
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load HotpotQA dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_relevant_docs(self, record: Dict[str, Any]) -> List[str]:
        """Extract relevant document titles from supporting facts."""
        supporting_facts = record.get('supporting_facts', [])
        relevant_titles = [fact[0] for fact in supporting_facts]
        return list(set(relevant_titles))  # Remove duplicates
    
    def _calculate_precision_recall(self, 
                                   retrieved_titles: List[str], 
                                   relevant_titles: List[str],
                                   k: int) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 at k."""
        if k > len(retrieved_titles):
            k = len(retrieved_titles)
        
        if k == 0 or len(relevant_titles) == 0:
            return 0.0, 0.0, 0.0
            
        retrieved_at_k = retrieved_titles[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant_titles))
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(relevant_titles) if len(relevant_titles) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _calculate_dcg(self, retrieved_titles: List[str], relevant_titles: List[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain at k."""
        if k > len(retrieved_titles):
            k = len(retrieved_titles)
            
        dcg = 0.0
        for i, title in enumerate(retrieved_titles[:k]):
            relevance = 1 if title in relevant_titles else 0
            dcg += relevance / (1 + i)  # Using log2(i+1) is more common, but this is simpler
            
        return dcg
    
    def _calculate_ndcg(self, retrieved_titles: List[str], relevant_titles: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        dcg = self._calculate_dcg(retrieved_titles, relevant_titles, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_ranking = relevant_titles + [t for t in retrieved_titles if t not in relevant_titles]
        idcg = self._calculate_dcg(ideal_ranking, relevant_titles, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_average_precision(self, retrieved_titles: List[str], relevant_titles: List[str]) -> float:
        """Calculate Average Precision for a single query."""
        if len(relevant_titles) == 0:
            return 0.0
            
        precisions = []
        relevant_count = 0
        
        for i, title in enumerate(retrieved_titles):
            if title in relevant_titles:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        return statistics.mean(precisions) if precisions else 0.0
    
    def precompute_query_embeddings(self, force_regenerate: bool = False):
        """Precompute embeddings for all queries in the dataset.
        
        Args:
            force_regenerate: Whether to regenerate existing embeddings
        """
        if self.query_cache is None:
            print("Error: Query cache not initialized. Provide query_cache_file and embedding_config.")
            return
        
        print("Precomputing query embeddings for evaluation dataset...")
        self.query_cache.precompute_dataset_embeddings(self.dataset, force_regenerate)
    
    def get_query_embedding(self, query_text: str) -> Optional[np.ndarray]:
        """Get embedding for a query, using cache if available.
        
        Args:
            query_text: Query text to embed
            
        Returns:
            Embedding vector or None if not available
        """
        if self.query_cache:
            return self.query_cache.get_embedding(query_text)
        return None
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the query embedding cache.
        
        Returns:
            Cache statistics or None if no cache
        """
        if self.query_cache:
            return self.query_cache.get_cache_stats()
        return None
    
    def evaluate_system(self, 
                       retriever,
                       system_name: str,
                       max_queries: Optional[int] = None,
                       top_k: int = 10,
                       use_precomputed_embeddings: bool = True) -> EvaluationMetrics:
        """
        Evaluate a retrieval system.
        
        Args:
            retriever: Retrieval system (Azure AI Search or FAISS)
            system_name: Name of the system
            max_queries: Maximum number of queries to evaluate (None for all)
            top_k: Number of documents to retrieve per query
            use_precomputed_embeddings: Whether to use precomputed query embeddings
            
        Returns:
            EvaluationMetrics object
        """
        print(f"Evaluating {system_name}...")
        
        # Check if we should pass query embeddings to retriever
        pass_embeddings = (use_precomputed_embeddings and 
                          self.query_cache is not None and 
                          hasattr(retriever, 'search_with_embedding'))
        
        query_results = []
        total_queries = len(self.dataset) if max_queries is None else min(max_queries, len(self.dataset))
        
        # Process queries
        for i, record in enumerate(self.dataset[:total_queries]):
            if i % 100 == 0:
                print(f"Processing query {i+1}/{total_queries}")
                
            query_id = record.get('_id', str(i))
            query_text = record.get('question', '')
            relevant_docs = self._extract_relevant_docs(record)
            
            # Time the query
            start_time = time.time()
            try:
                # Use precomputed embeddings if available and supported
                if pass_embeddings:
                    query_embedding = self.get_query_embedding(query_text)
                    if query_embedding is not None:
                        search_results = retriever.search_with_embedding(query_text, query_embedding, top_k=top_k)
                    else:
                        # Fallback to regular search
                        search_results = retriever.search(query_text, top_k=top_k)
                else:
                    search_results = retriever.search(query_text, top_k=top_k)
                    
                response_time = time.time() - start_time
                success = True
                
                # Extract retrieved document info
                retrieved_docs = []
                for result in search_results:
                    retrieved_docs.append({
                        'id': result.doc_id,
                        'title': result.title,
                        'score': result.score
                    })
                    
            except Exception as e:
                print(f"Query {i+1} failed: {e}")
                response_time = time.time() - start_time
                retrieved_docs = []
                success = False
            
            query_result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                retrieved_docs=retrieved_docs,
                response_time=response_time,
                relevant_docs=relevant_docs,
                success=success
            )
            query_results.append(query_result)
        
        # Calculate metrics
        return self._calculate_metrics(query_results, system_name)
    
    def _calculate_metrics(self, query_results: List[QueryResult], system_name: str) -> EvaluationMetrics:
        """Calculate evaluation metrics from query results."""
        
        successful_results = [r for r in query_results if r.success]
        total_queries = len(query_results)
        successful_queries = len(successful_results)
        
        if successful_queries == 0:
            print("No successful queries to evaluate!")
            return EvaluationMetrics(
                precision_at_k=[0.0] * len(self.k_values),
                recall_at_k=[0.0] * len(self.k_values),
                f1_at_k=[0.0] * len(self.k_values),
                map_score=0.0,
                ndcg_at_k=[0.0] * len(self.k_values),
                avg_query_time=0.0,
                queries_per_second=0.0,
                total_queries=total_queries,
                successful_queries=successful_queries,
                avg_result_count=0.0,
                system_name=system_name
            )
        
        # Calculate metrics for each k value
        precision_at_k = {k: [] for k in self.k_values}
        recall_at_k = {k: [] for k in self.k_values}
        f1_at_k = {k: [] for k in self.k_values}
        ndcg_at_k = {k: [] for k in self.k_values}
        ap_scores = []
        response_times = []
        result_counts = []
        
        for result in successful_results:
            retrieved_titles = [doc['title'] for doc in result.retrieved_docs]
            relevant_titles = result.relevant_docs
            
            # Calculate metrics at different k values
            for k in self.k_values:
                prec, rec, f1 = self._calculate_precision_recall(retrieved_titles, relevant_titles, k)
                ndcg = self._calculate_ndcg(retrieved_titles, relevant_titles, k)
                
                precision_at_k[k].append(prec)
                recall_at_k[k].append(rec)
                f1_at_k[k].append(f1)
                ndcg_at_k[k].append(ndcg)
            
            # Average precision
            ap = self._calculate_average_precision(retrieved_titles, relevant_titles)
            ap_scores.append(ap)
            
            response_times.append(result.response_time)
            result_counts.append(len(result.retrieved_docs))
        
        # Calculate final metrics
        avg_response_time = statistics.mean(response_times)
        qps = successful_queries / sum(response_times) if sum(response_times) > 0 else 0
        
        return EvaluationMetrics(
            precision_at_k=[statistics.mean(precision_at_k[k]) for k in self.k_values],
            recall_at_k=[statistics.mean(recall_at_k[k]) for k in self.k_values],
            f1_at_k=[statistics.mean(f1_at_k[k]) for k in self.k_values],
            map_score=statistics.mean(ap_scores),
            ndcg_at_k=[statistics.mean(ndcg_at_k[k]) for k in self.k_values],
            avg_query_time=avg_response_time,
            queries_per_second=qps,
            total_queries=total_queries,
            successful_queries=successful_queries,
            avg_result_count=statistics.mean(result_counts),
            system_name=system_name
        )
    
    def compare_systems(self, 
                       azure_metrics: EvaluationMetrics, 
                       faiss_metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Generate comparison report between systems."""
        
        comparison = {
            'systems': {
                'azure_ai_search': asdict(azure_metrics),
                'faiss': asdict(faiss_metrics)
            },
            'comparison': {
                'k_values': self.k_values,
                'precision_comparison': {},
                'recall_comparison': {},
                'f1_comparison': {},
                'ndcg_comparison': {},
                'performance_comparison': {}
            }
        }
        
        # Compare metrics at each k value
        for i, k in enumerate(self.k_values):
            comparison['comparison']['precision_comparison'][f'precision@{k}'] = {
                'azure': azure_metrics.precision_at_k[i],
                'faiss': faiss_metrics.precision_at_k[i],
                'winner': 'azure' if azure_metrics.precision_at_k[i] > faiss_metrics.precision_at_k[i] else 'faiss'
            }
            
            comparison['comparison']['recall_comparison'][f'recall@{k}'] = {
                'azure': azure_metrics.recall_at_k[i],
                'faiss': faiss_metrics.recall_at_k[i], 
                'winner': 'azure' if azure_metrics.recall_at_k[i] > faiss_metrics.recall_at_k[i] else 'faiss'
            }
            
            comparison['comparison']['f1_comparison'][f'f1@{k}'] = {
                'azure': azure_metrics.f1_at_k[i],
                'faiss': faiss_metrics.f1_at_k[i],
                'winner': 'azure' if azure_metrics.f1_at_k[i] > faiss_metrics.f1_at_k[i] else 'faiss'
            }
            
            comparison['comparison']['ndcg_comparison'][f'ndcg@{k}'] = {
                'azure': azure_metrics.ndcg_at_k[i],
                'faiss': faiss_metrics.ndcg_at_k[i],
                'winner': 'azure' if azure_metrics.ndcg_at_k[i] > faiss_metrics.ndcg_at_k[i] else 'faiss'
            }
        
        # Performance comparison
        comparison['comparison']['performance_comparison'] = {
            'avg_query_time': {
                'azure': azure_metrics.avg_query_time,
                'faiss': faiss_metrics.avg_query_time,
                'winner': 'azure' if azure_metrics.avg_query_time < faiss_metrics.avg_query_time else 'faiss'
            },
            'queries_per_second': {
                'azure': azure_metrics.queries_per_second,
                'faiss': faiss_metrics.queries_per_second,
                'winner': 'azure' if azure_metrics.queries_per_second > faiss_metrics.queries_per_second else 'faiss'
            },
            'map_score': {
                'azure': azure_metrics.map_score,
                'faiss': faiss_metrics.map_score,
                'winner': 'azure' if azure_metrics.map_score > faiss_metrics.map_score else 'faiss'
            }
        }
        
        return comparison
    
    def save_results(self, comparison_results: Dict[str, Any], output_path: str):
        """Save comparison results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Azure AI Search vs FAISS on HotpotQA")
    parser.add_argument("--dataset", default="hotpot_dev_fullwiki_v1.json", 
                       help="Path to HotpotQA dataset")
    parser.add_argument("--documents", default="hotpot_documents.json",
                       help="Path to processed documents")
    parser.add_argument("--max_queries", type=int, default=None,
                       help="Maximum number of queries to evaluate")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--skip_azure", action="store_true",
                       help="Skip Azure AI Search evaluation")
    parser.add_argument("--skip_faiss", action="store_true", 
                       help="Skip FAISS evaluation")
    
    # Query embedding cache arguments
    parser.add_argument("--query_cache", default="query_embeddings.pkl",
                       help="Path to query embedding cache file")
    parser.add_argument("--precompute_embeddings", action="store_true",
                       help="Precompute query embeddings before evaluation")
    parser.add_argument("--force_regenerate", action="store_true",
                       help="Force regeneration of existing query embeddings")
    # Embedding configuration is handled internally by EmbeddingConfig
    # No need for command line arguments - uses environment variables and defaults
    parser.add_argument("--disable_precomputed", action="store_true",
                       help="Disable use of precomputed embeddings during evaluation")
    parser.add_argument("--cache_stats", action="store_true",
                       help="Show query embedding cache statistics and exit")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        return
    
    if not os.path.exists(args.documents):
        print(f"Processed documents not found: {args.documents}")
        print("Run document_processor.py first to generate documents")
        return
    
    # Initialize embedding configuration for query cache
    embedding_config = None
    if args.precompute_embeddings or os.path.exists(args.query_cache):
        # Create embedding config with default settings
        # Provider, API keys, and model selection handled by EmbeddingConfig internally
        embedding_config = EmbeddingConfig()
    
    # Initialize evaluator with query cache support
    evaluator = HotpotQAEvaluator(
        args.dataset, 
        query_cache_file=args.query_cache if not args.disable_precomputed else None,
        embedding_config=embedding_config
    )
    
    # Show cache statistics if requested
    if args.cache_stats:
        cache_stats = evaluator.get_cache_stats()
        if cache_stats:
            print("\n=== QUERY EMBEDDING CACHE STATISTICS ===")
            for key, value in cache_stats.items():
                print(f"{key}: {value}")
        else:
            print("No query embedding cache available")
        return
    
    # Precompute query embeddings if requested
    if args.precompute_embeddings:
        print("\n=== PRECOMPUTING QUERY EMBEDDINGS ===")
        if embedding_config is None:
            print("Error: No embedding configuration available for precomputation")
            return
        
        evaluator.precompute_query_embeddings(force_regenerate=args.force_regenerate)
        
        # Show updated cache statistics
        cache_stats = evaluator.get_cache_stats()
        if cache_stats:
            print("\n=== UPDATED CACHE STATISTICS ===")
            for key, value in cache_stats.items():
                print(f"{key}: {value}")
        
        print("\nQuery embedding precomputation completed!")
        return
    
    azure_metrics = None
    faiss_metrics = None
    
    # Evaluate Azure AI Search
    if not args.skip_azure:
        # Check environment variables for Azure (OpenAI not required - uses SentenceTransformers by default)
        search_service = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        search_key = os.getenv('AZURE_SEARCH_API_KEY')
        
        if all([search_service, search_key]):
            azure_retriever = AzureAISearchRetriever(
                search_service_name=search_service,
                search_api_key=search_key
            )
            azure_metrics = evaluator.evaluate_system(
                azure_retriever, "Azure AI Search", args.max_queries,
                use_precomputed_embeddings=not args.disable_precomputed
            )
        else:
            print("Skipping Azure AI Search - missing environment variables:")
            print("  Required: AZURE_SEARCH_SERVICE_NAME, AZURE_SEARCH_API_KEY")
    
    # Evaluate FAISS
    if not args.skip_faiss:
        # Load documents for FAISS
        with open(args.documents, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        faiss_retriever = FAISSRetriever()
        faiss_retriever.build_index(documents)
        
        faiss_metrics = evaluator.evaluate_system(
            faiss_retriever, "FAISS", args.max_queries,
            use_precomputed_embeddings=not args.disable_precomputed
        )
    
    # Generate comparison if both metrics available
    if azure_metrics and faiss_metrics:
        comparison = evaluator.compare_systems(azure_metrics, faiss_metrics)
        evaluator.save_results(comparison, args.output)
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        for metric_type in ['precision_comparison', 'recall_comparison', 'f1_comparison']:
            print(f"\n{metric_type.replace('_', ' ').title()}:")
            for k, values in comparison['comparison'][metric_type].items():
                print(f"  {k}: Azure={values['azure']:.4f}, FAISS={values['faiss']:.4f}, Winner={values['winner']}")
    
    elif azure_metrics:
        print("\n=== AZURE AI SEARCH RESULTS ===")
        print(f"MAP Score: {azure_metrics.map_score:.4f}")
        print(f"Avg Query Time: {azure_metrics.avg_query_time:.4f}s")
        print(f"Queries/Second: {azure_metrics.queries_per_second:.2f}")
        
    elif faiss_metrics:
        print("\n=== FAISS RESULTS ===")
        print(f"MAP Score: {faiss_metrics.map_score:.4f}")
        print(f"Avg Query Time: {faiss_metrics.avg_query_time:.4f}s")
        print(f"Queries/Second: {faiss_metrics.queries_per_second:.2f}")


if __name__ == "__main__":
    main()