#!/usr/bin/env python3
"""
Main script to run the complete HotpotQA evaluation pipeline.

This script orchestrates the entire evaluation process:
1. Process HotpotQA dataset into documents
2. Set up both Azure AI Search and FAISS retrievers
3. Run evaluation on both systems
4. Generate comparison report

Usage:
    python run_evaluation.py --quick     # Quick test with 100 queries
    python run_evaluation.py --full      # Full evaluation (may take hours)
    python run_evaluation.py --setup-only # Only process documents, no evaluation
"""

import os
import json
import argparse
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("python-dotenv not found. Environment variables will be loaded from system only.")
    print("Install with: pip install python-dotenv")

from document_processor import DocumentProcessor
from azure_ai_search import AzureAISearchRetriever
from faiss_retriever import FAISSRetriever
from evaluation import HotpotQAEvaluator
from embedding_generator import EmbeddingConfig


def check_requirements():
    """Check if all required packages are installed."""
    missing_packages = []
    
    try:
        import azure.search.documents
    except ImportError:
        missing_packages.append("azure-search-documents")
    
    try:
        import faiss
    except ImportError:
        missing_packages.append("faiss-cpu")
        
    try:
        import sentence_transformers
    except ImportError:
        missing_packages.append("sentence-transformers")
        
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_packages.append("python-dotenv")
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


def check_environment():
    """Check if required environment variables are set."""
    azure_vars = [
        'AZURE_SEARCH_SERVICE_NAME',
        'AZURE_SEARCH_API_KEY'
    ]
    
    azure_available = all(os.getenv(var) for var in azure_vars)
    
    if not azure_available:
        print("Azure AI Search environment variables not set:")
        for var in azure_vars:
            if not os.getenv(var):
                print(f"  - {var}")
        print("Azure AI Search evaluation will be skipped.")
        print("\nTo use Azure AI Search, set these environment variables or create a .env file:")
        print("AZURE_SEARCH_SERVICE_NAME=your-service-name")
        print("AZURE_SEARCH_API_KEY=your-api-key")
    
    return True, azure_available


def process_documents(dataset_path: str, output_path: str, force: bool = False):
    """Process HotpotQA dataset into documents."""
    
    if os.path.exists(output_path) and not force:
        print(f"Documents already exist at {output_path}. Use --force to reprocess.")
        return True  # Return True so we can proceed to check indices
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the HotpotQA dataset is downloaded.")
        return False
    
    print("Processing HotpotQA dataset...")
    processor = DocumentProcessor(dataset_path)
    documents = processor.process_all()
    processor.save_documents(documents, output_path)
    
    return True


def setup_azure_search(documents_path: str, force_reindex: bool = False):
    """Set up Azure AI Search index."""
    
    # Check environment
    search_service = os.getenv('AZURE_SEARCH_SERVICE_NAME')
    search_key = os.getenv('AZURE_SEARCH_API_KEY')
    
    if not all([search_service, search_key]):
        print("Azure AI Search environment variables not available")
        return None
    
    # Load documents
    if not os.path.exists(documents_path):
        print(f"Documents not found: {documents_path}")
        return None
        
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize retriever
    retriever = AzureAISearchRetriever(
        search_service_name=search_service,
        search_api_key=search_key
    )
    
    # Check if index exists and has documents
    index_exists = retriever.index_exists()
    doc_count = 0
    
    if index_exists:
        doc_count = retriever.get_document_count()
        print(f"Azure index '{retriever.index_name}' exists with {doc_count} documents")
    else:
        print(f"Azure index '{retriever.index_name}' does not exist")
    
    # Create index if it doesn't exist
    if not index_exists:
        print("Creating Azure AI Search index...")
        retriever.create_index()
        index_exists = True
    
    # Index documents if forcing reindex, index is empty, or index was just created
    if force_reindex or doc_count == 0 or not index_exists:
        if force_reindex:
            print("Force reindexing documents...")
        else:
            print("Indexing documents (this may take a while)...")
        retriever.index_documents(documents)
    else:
        print("Index already contains documents. Use --force to reindex.")
    
    return retriever


def setup_faiss_index(documents_path: str, 
                      index_path: str = "hotpot_faiss_index",
                      force_rebuild: bool = False,
                      embedding_provider: str = "gemini",
                      use_openai: bool = False):  # Backward compatibility
    """Set up FAISS index."""
    
    # Handle backward compatibility
    if use_openai:
        embedding_provider = 'openai'
    
    openai_key = os.getenv('OPENAI_API_KEY') if embedding_provider == 'openai' else None
    gemini_key = os.getenv('GEMINI_API_KEY') if embedding_provider == 'gemini' else None
    
    # Try to load existing index
    if os.path.exists(f"{index_path}.faiss") and not force_rebuild:
        print("Loading existing FAISS index...")
        print(f"Using {embedding_provider} embeddings")
        # Create embedding config for backward compatibility
        config = EmbeddingConfig.from_args(
            embedding_provider=embedding_provider,
            openai_api_key=openai_key,
            gemini_api_key=gemini_key
        )
        retriever = FAISSRetriever(embedding_config=config)
        retriever.load_index(index_path)
        return retriever
    
    # Build new index
    if not os.path.exists(documents_path):
        print(f"Documents not found: {documents_path}")
        return None
        
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print("Building FAISS index...")
    print(f"Using {embedding_provider} embeddings")
    # Create embedding config for backward compatibility
    config = EmbeddingConfig.from_args(
        embedding_provider=embedding_provider,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key
    )
    retriever = FAISSRetriever(embedding_config=config)
    retriever.build_index(documents, save_path=index_path)
    
    return retriever


def run_evaluation(dataset_path: str, 
                   azure_retriever: Optional[AzureAISearchRetriever],
                   faiss_retriever: Optional[FAISSRetriever],
                   max_queries: Optional[int] = None,
                   output_path: str = "evaluation_results.json"):
    """Run the evaluation on both systems."""
    
    evaluator = HotpotQAEvaluator(dataset_path)
    
    azure_metrics = None
    faiss_metrics = None
    
    # Evaluate Azure AI Search
    if azure_retriever:
        print("Evaluating Azure AI Search...")
        azure_metrics = evaluator.evaluate_system(
            azure_retriever, "Azure AI Search", max_queries
        )
    
    # Evaluate FAISS
    if faiss_retriever:
        print("Evaluating FAISS...")
        faiss_metrics = evaluator.evaluate_system(
            faiss_retriever, "FAISS", max_queries
        )
    
    # Generate comparison
    if azure_metrics and faiss_metrics:
        print("Generating comparison report...")
        comparison = evaluator.compare_systems(azure_metrics, faiss_metrics)
        evaluator.save_results(comparison, output_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nQueries evaluated: {azure_metrics.total_queries}")
        print(f"Successful queries: Azure={azure_metrics.successful_queries}, FAISS={faiss_metrics.successful_queries}")
        
        print(f"\nMAP Score:")
        print(f"  Azure AI Search: {azure_metrics.map_score:.4f}")
        print(f"  FAISS:           {faiss_metrics.map_score:.4f}")
        print(f"  Winner:          {'Azure' if azure_metrics.map_score > faiss_metrics.map_score else 'FAISS'}")
        
        print(f"\nAverage Query Time:")
        print(f"  Azure AI Search: {azure_metrics.avg_query_time:.4f}s")
        print(f"  FAISS:           {faiss_metrics.avg_query_time:.4f}s")
        print(f"  Winner:          {'Azure' if azure_metrics.avg_query_time < faiss_metrics.avg_query_time else 'FAISS'}")
        
        print(f"\nQueries Per Second:")
        print(f"  Azure AI Search: {azure_metrics.queries_per_second:.2f}")
        print(f"  FAISS:           {faiss_metrics.queries_per_second:.2f}")
        print(f"  Winner:          {'Azure' if azure_metrics.queries_per_second > faiss_metrics.queries_per_second else 'FAISS'}")
        
        return comparison
    
    elif azure_metrics:
        print("\nAzure AI Search evaluation completed")
        print(f"MAP Score: {azure_metrics.map_score:.4f}")
        return {"azure_only": azure_metrics}
        
    elif faiss_metrics:
        print("\nFAISS evaluation completed") 
        print(f"MAP Score: {faiss_metrics.map_score:.4f}")
        return {"faiss_only": faiss_metrics}
    
    else:
        print("No evaluation performed - no retrievers available")
        return None


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Run HotpotQA evaluation comparing Azure AI Search vs FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --quick          # Quick test with 100 queries
  python run_evaluation.py --full           # Full evaluation
  python run_evaluation.py --setup-only     # Only process data, no evaluation
  python run_evaluation.py --faiss-only     # Only evaluate FAISS
  python run_evaluation.py --azure-only     # Only evaluate Azure AI Search
        """
    )
    
    # Evaluation modes
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with 100 queries")
    parser.add_argument("--full", action="store_true", 
                       help="Full evaluation with all queries")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only process documents and setup indices")
    
    # System selection
    parser.add_argument("--azure-only", action="store_true",
                       help="Only evaluate Azure AI Search")
    parser.add_argument("--faiss-only", action="store_true",
                       help="Only evaluate FAISS")
    
    # File paths
    parser.add_argument("--dataset", default="hotpot_dev_fullwiki_v1.json",
                       help="Path to HotpotQA dataset")
    parser.add_argument("--documents", default="hotpot_documents.json",
                       help="Path for processed documents")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output path for results")
    
    # Options
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing/reindexing")
    parser.add_argument("--max-queries", type=int,
                       help="Maximum queries to evaluate")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    env_ok, azure_available = check_environment()
    
    # Exit only if Azure is specifically requested but not available
    if args.azure_only and not azure_available:
        print("Azure-only mode requested but Azure environment variables missing")
        sys.exit(1)
    
    # For general setup, proceed if we can at least process documents
    # FAISS can always work with SentenceTransformers (no API key needed)
    
    # Set query limits
    if args.quick and not args.max_queries:
        args.max_queries = 100
    
    # Process documents
    print("Step 1: Processing documents...")
    if not process_documents(args.dataset, args.documents, args.force):
        sys.exit(1)
    
    # Always proceed to setup step, even if documents already exist
    # This allows checking and setting up indices
    
    if args.setup_only:
        print("\nSetting up indices...")
        # Continue to setup both Azure and FAISS indices, then return
        pass
    
    # Initialize retrievers
    azure_retriever = None
    faiss_retriever = None
    
    # Setup Azure AI Search
    if not args.faiss_only and azure_available:
        print("\nStep 2a: Setting up Azure AI Search...")
        try:
            azure_retriever = setup_azure_search(args.documents, args.force)
        except Exception as e:
            print(f"Failed to setup Azure AI Search: {e}")
            if args.azure_only:
                sys.exit(1)
    
    # Setup FAISS (always available with SentenceTransformers)
    if not args.azure_only:
        print("\nStep 2b: Setting up FAISS...")
        try:
            # Use SentenceTransformers by default (no API key required)
            faiss_retriever = setup_faiss_index(args.documents, force_rebuild=args.force, use_openai=False)
        except Exception as e:
            print(f"Failed to setup FAISS: {e}")
            if args.faiss_only:
                sys.exit(1)
    
    # If setup-only mode, exit after setting up indices
    if args.setup_only:
        print("\nSetup completed successfully!")
        if azure_retriever:
            print("✓ Azure AI Search index ready")
        if faiss_retriever:
            print("✓ FAISS index ready")
        return
    
    # Run evaluation
    print("\nStep 3: Running evaluation...")
    results = run_evaluation(
        args.dataset, 
        azure_retriever, 
        faiss_retriever,
        args.max_queries,
        args.output
    )
    
    if results:
        print(f"\nEvaluation completed! Results saved to {args.output}")
    else:
        print("Evaluation failed - no retrievers available")
        sys.exit(1)


if __name__ == "__main__":
    main()