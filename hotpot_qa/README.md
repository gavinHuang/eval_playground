# HotpotQA Benchmark: Azure AI Search vs FAISS

This benchmark evaluates and compares Azure AI Search and FAISS (Facebook AI Similarity Search) retrieval systems on the HotpotQA dataset for multi-hop question answering.

## Overview

The HotpotQA dataset contains complex questions that require reasoning over multiple documents. This benchmark tests how well different retrieval systems can find relevant passages to answer these questions.

**Systems Evaluated:**
- **Azure AI Search**: Cloud-based search service with hybrid vector and keyword search
- **FAISS**: Local vector similarity search with dense embeddings

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

For Azure AI Search:
```bash
export AZURE_SEARCH_SERVICE_NAME="your-search-service-name"
export AZURE_SEARCH_API_KEY="your-search-api-key"
```

For both systems (required):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run Evaluation

**Quick test (100 queries):**
```bash
python run_evaluation.py --quick
```

**Full evaluation (all 7,405 queries):**
```bash
python run_evaluation.py --full
```

**Setup only (process data without evaluation):**
```bash
python run_evaluation.py --setup-only
```

## Dataset

The evaluation uses the HotpotQA development set (`hotpot_dev_fullwiki_v1.json`) which contains:
- 7,405 question-answer pairs
- Multi-hop questions requiring reasoning across multiple documents
- Supporting facts indicating relevant passages
- Context documents with titles and content

Example question: *"Were Scott Derrickson and Ed Wood of the same nationality?"*

## Architecture

### Document Processing (`document_processor.py`)
- Transforms HotpotQA records into indexable documents
- Each context entry becomes a separate document for granular retrieval
- Generates structured documents with ID, title, content, and source

### Azure AI Search (`azure_ai_search.py`)
- Uses hybrid search combining vector similarity and keyword matching
- Leverages OpenAI embeddings (text-embedding-3-small)
- Implements vector search with HNSW algorithm
- Provides cloud-based scalable search

### FAISS Retriever (`faiss_retriever.py`)
- Local vector search using FAISS IndexFlatIP
- Supports both SentenceTransformers and OpenAI embeddings
- Implements cosine similarity for document ranking
- Fast in-memory search with configurable embedding models

### Evaluation Framework (`evaluation.py`)
- Comprehensive metrics including Precision@K, Recall@K, F1@K, MAP, NDCG@K
- Performance metrics: query time, throughput
- Relevance assessment based on HotpotQA supporting facts
- Statistical comparison between systems

## Metrics

### Relevance Metrics
- **Precision@K**: Fraction of retrieved docs that are relevant at rank K
- **Recall@K**: Fraction of relevant docs retrieved at rank K  
- **F1@K**: Harmonic mean of precision and recall at rank K
- **MAP**: Mean Average Precision across all queries
- **NDCG@K**: Normalized Discounted Cumulative Gain at rank K

### Performance Metrics
- **Average Query Time**: Mean response time per query
- **Queries Per Second**: System throughput
- **Success Rate**: Percentage of queries that returned results

## Usage Examples

### Evaluate Both Systems
```bash
python run_evaluation.py --full --output comparison_results.json
```

### Evaluate Only FAISS
```bash
python run_evaluation.py --faiss-only --max-queries 500
```

### Evaluate Only Azure AI Search
```bash
python run_evaluation.py --azure-only --quick
```

### Process Documents Only
```bash
python document_processor.py
```

### Test Individual Components
```bash
# Test Azure AI Search
python azure_ai_search.py

# Test FAISS
python faiss_retriever.py

# Run evaluation module
python evaluation.py --help
```

## File Structure

```
Benchmarks/hotpot_qa/
├── hotpot_qa_for_azure_ai_search.md    # Task specification
├── hotpot_dev_fullwiki_v1.json         # HotpotQA dataset  
├── hotpot_example_format.json          # Example record format
├── document_processor.py               # Dataset preprocessing
├── azure_ai_search.py                  # Azure AI Search implementation
├── faiss_retriever.py                  # FAISS implementation
├── evaluation.py                       # Evaluation framework
├── run_evaluation.py                   # Main orchestration script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Configuration

### Environment Variables
- `AZURE_SEARCH_SERVICE_NAME`: Your Azure Search service name
- `AZURE_SEARCH_API_KEY`: Your Azure Search API key
- `OPENAI_API_KEY`: OpenAI API key for embeddings (required for both systems)

### Customization Options
- **Embedding Models**: Configure different models in FAISS retriever
- **Index Parameters**: Adjust FAISS index type and parameters
- **Search Parameters**: Modify top-K retrieval, hybrid search weights
- **Evaluation Metrics**: Add custom relevance metrics or modify existing ones

## Expected Results

The evaluation generates a comprehensive comparison including:

```json
{
  "systems": {
    "azure_ai_search": { "precision_at_k": [...], "recall_at_k": [...], ... },
    "faiss": { "precision_at_k": [...], "recall_at_k": [...], ... }
  },
  "comparison": {
    "precision_comparison": { "precision@1": {"azure": 0.45, "faiss": 0.42, "winner": "azure"} },
    "performance_comparison": { "avg_query_time": {"azure": 0.85, "faiss": 0.12, "winner": "faiss"} }
  }
}
```

## Troubleshooting

### Common Issues

**Missing API Keys:**
```bash
Error: OPENAI_API_KEY environment variable not found
```
Set the required environment variables as shown above.

**Azure Search Setup:**
```bash
Failed to create index: Authentication failure
```
Verify your Azure Search service name and API key are correct.

**Memory Issues:**
```bash
MemoryError: Unable to allocate array
```
For large datasets, consider using batch processing or reduce the number of queries evaluated.

**Package Installation:**
```bash
ImportError: No module named 'faiss'
```
Install all requirements: `pip install -r requirements.txt`

### Performance Tips

1. **For faster evaluation**: Use `--quick` mode for development and testing
2. **For Azure cost optimization**: Limit queries with `--max-queries`
3. **For memory efficiency**: Use FAISS with SentenceTransformers instead of OpenAI embeddings
4. **For production**: Consider using FAISS GPU version for better performance

## Contributing

To extend this benchmark:

1. Add new retrieval systems by implementing the same interface as existing retrievers
2. Add new evaluation metrics in the `evaluation.py` module  
3. Extend document processing for different datasets
4. Add support for different embedding models

## License

This benchmark is provided as-is for research and evaluation purposes.