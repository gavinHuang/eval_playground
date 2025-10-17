# FAISS Vector Search Update

## Summary

Updated the `_retrieve_relevant_cells` function in `agentar_scale_sql.py` to use FAISS vector search instead of simple keyword matching. This provides more intelligent semantic retrieval of relevant database cell values.

## Changes Made

### 1. Added Dependencies

**File: `agentar_scale_sql.py`**
- Added `numpy` import for array operations
- Added `faiss` import for vector search
- Added `OpenAIEmbeddings` and `AzureOpenAIEmbeddings` from `langchain_openai`

**File: `pyproject.toml`**
- Added `numpy>=1.24.0` to dependencies
- Added `faiss-cpu>=1.7.4` to dependencies

### 2. Updated `__init__` Method

Added initialization of embeddings model:
```python
# Initialize embeddings model
self.embeddings = self._create_embeddings()

# Initialize vector stores
self.cell_store, self.cell_index, self.cell_texts = self._build_cell_store()
```

### 3. New `_create_embeddings` Method

Created a new method to initialize the OpenAI embeddings model (supports both Azure and standard OpenAI):
- Uses `text-embedding-3-small` model by default
- Auto-detects Azure vs standard OpenAI configuration
- Properly handles API keys and endpoints

### 4. Updated `_build_cell_store` Method

**Before:** Simple dictionary-based storage
```python
def _build_cell_store(self) -> Dict[str, List[str]]:
    # Just stored cell values in a dictionary
```

**After:** FAISS vector index creation
```python
def _build_cell_store(self) -> Tuple[Dict[str, List[str]], Any, List[str]]:
    # 1. Extract cell values from database
    # 2. Create text representations for embedding
    # 3. Generate embeddings using OpenAI API
    # 4. Build FAISS index from embeddings
    # Returns: (cell_store dict, FAISS index, cell texts list)
```

The new version:
- Creates embeddings for up to 10 unique values per table.column
- Builds a FAISS `IndexFlatL2` index for L2 distance search
- Returns tuple of (cell_store, index, cell_texts) instead of just dict

### 5. Updated `_retrieve_relevant_cells` Method

**Before:** Simple keyword substring matching
```python
def _retrieve_relevant_cells(self, keywords: List[str]) -> str:
    for keyword in keywords:
        for key, values in self.cell_store.items():
            if keyword_lower in key.lower():
                # Return if match found
```

**After:** Semantic vector search
```python
def _retrieve_relevant_cells(self, keywords: List[str], top_k: int = 20) -> str:
    # 1. Combine keywords into query text
    # 2. Generate embedding for query
    # 3. Search FAISS index for nearest neighbors
    # 4. Return top-k most semantically similar cells
```

The new version:
- Joins keywords into a single query string
- Generates embedding for the query using OpenAI API
- Performs semantic similarity search using FAISS
- Returns top-k most relevant cells (default 20)
- Deduplicates results

## Benefits

1. **Semantic Understanding**: Vector search understands meaning, not just exact keyword matches
   - "currency" query will find "EUR", "CZK" even without exact match
   - "customer payment" finds transaction-related columns

2. **Better Relevance**: Results are ranked by semantic similarity
   - Most relevant database values appear first
   - Reduces noise from irrelevant keyword matches

3. **Scalability**: FAISS is optimized for large-scale vector search
   - Efficient for databases with many tables/columns
   - Can scale to millions of cell values

4. **Multilingual Support**: Embeddings work across languages
   - Can match English queries to non-English data
   - Better for international datasets

## Testing

Created `test_faiss_vector_search.py` to verify functionality:
- ✅ FAISS index creation
- ✅ Embedding generation
- ✅ Vector search retrieval
- ✅ Keyword extraction integration

Test results show the system successfully:
- Built FAISS index with 21 vectors (one per table.column)
- Retrieved semantically relevant cells for currency-related queries
- Properly integrated with existing Agentar-Scale-SQL pipeline

## Installation

Required packages (auto-installed in venv):
```bash
pip install numpy faiss-cpu
```

## Performance Considerations

1. **Initial Overhead**: Building FAISS index requires:
   - Database scan to extract cell values
   - Embedding generation API calls (~21 calls for debit_card DB)
   - One-time cost during initialization

2. **Query Time**: Each `_retrieve_relevant_cells` call:
   - 1 embedding API call for query
   - Fast FAISS vector search (milliseconds)
   - Minimal overhead vs keyword matching

3. **Cost**: OpenAI embedding API calls:
   - ~$0.00002 per 1K tokens
   - Initialization cost for sample DB: ~$0.001
   - Per-query cost: ~$0.0001

## Future Enhancements

Potential improvements:
1. Cache FAISS index to disk to avoid rebuilding
2. Use different FAISS index types (IVF, HNSW) for larger databases
3. Add hybrid search combining keyword + vector similarity
4. Implement embedding caching for common queries
