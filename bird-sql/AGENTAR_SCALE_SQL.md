# Agentar-Scale-SQL Implementation

This document describes the implementation of the Agentar-Scale-SQL framework, a state-of-the-art text-to-SQL solution that uses a unified LLM with multi-stage generation and selection strategies.

## Overview

Agentar-Scale-SQL implements a sophisticated three-stage pipeline:

1. **Task Understanding**: Extracts context and retrieves relevant information
2. **SQL Generation Scaling**: Generates diverse SQL candidates using parallel and sequential scaling
3. **SQL Selection Scaling**: Consolidates and selects the best candidate using tournament selection

## Architecture

### Stage 1: Task Understanding

The task understanding stage establishes context by:

- **Keyword Extraction**: Uses LLM to extract important keywords from the question and evidence
- **Cell Value Retrieval**: Retrieves relevant database cell values based on keywords (simplified vector store)
- **Skeleton Extraction**: Identifies the SQL query structure/pattern implied by the question
- **Few-Shot Example Retrieval**: Retrieves relevant example queries based on the skeleton

### Stage 2: SQL Generation Scaling

This stage combines two scaling strategies:

#### Parallel Scaling (Diverse Synthesis)

Generates multiple SQL candidates using different approaches:

1. **Reasoning Candidates** (default: 4)
   - Uses DDL schema format
   - Emphasizes step-by-step reasoning
   - Low temperature (0.2-0.5) for precision
   - Simulates `M_reasoning` generator

2. **ICL (In-Context Learning) Candidates** (default: 5)
   - Uses lightweight markdown schema
   - Leverages few-shot examples
   - Varies prompt styles: direct, chain-of-thought, decomposition
   - Higher temperature (0.3-0.75) for diversity
   - Simulates `M_ICL` generator

#### Sequential Scaling (Iterative Refinement)

Refines each candidate through:

1. **Syntax Repair**: Fixes SQL syntax errors using LLM feedback
2. **Semantic Revision**: Optionally revises queries with logical issues
3. **Execution Validation**: Tests each query against the database

### Stage 3: SQL Selection Scaling

Selects the best candidate through:

1. **Candidate Consolidation**
   - Groups candidates by execution results
   - Selects one representative per group
   - Reduces redundancy

2. **Tournament Selection**
   - Performs pairwise comparisons using LLM as judge
   - Assigns scores based on comparison outcomes
   - Selects candidate with highest score
   - Considers execution success, query correctness, and efficiency

## Usage

### Basic Example

```python
from agentar_scale_sql import AgentarScaleSQL

# Initialize
agentar = AgentarScaleSQL(
    db_path="debit_card.db",
    model_name="gpt-4o",
    n_reasoning_candidates=4,
    n_icl_candidates=5
)

# Query
result = agentar.query(
    question="What is the ratio of customers who pay in EUR against customers who pay in CZK?",
    evidence=""
)

print(f"SQL: {result['sql']}")
print(f"Answer: {result['answer']}")
```

### Configuration Options

```python
agentar = AgentarScaleSQL(
    db_path="debit_card.db",
    model_name="gpt-4o",           # LLM model to use
    temperature=0.2,                # Default temperature
    n_reasoning_candidates=4,       # Number of reasoning candidates
    n_icl_candidates=5,             # Number of ICL candidates
    use_azure=True,                 # Use Azure OpenAI
    azure_endpoint="...",
    azure_deployment="..."
)
```

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# OR Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Agentar-Scale-SQL Configuration (optional)
AGENTAR_N_REASONING=3    # Number of reasoning candidates (default: 4)
AGENTAR_N_ICL=4          # Number of ICL candidates (default: 5)
```

## Evaluation

Run evaluation using the integrated evaluation framework:

```bash
# Run Agentar-Scale-SQL only
python run_evaluation.py --mode agentar

# Run all solutions including Agentar-Scale-SQL
python run_evaluation.py --mode full

# Run multiple specific solutions
python run_evaluation.py --mode vanilla langchain agentar
```

## Performance Characteristics

### Strengths

1. **High Accuracy**: Multiple candidate generation strategies increase chance of correct SQL
2. **Robustness**: Iterative refinement handles syntax and semantic errors
3. **Diversity**: Parallel scaling with different temperatures and styles explores solution space
4. **Quality Selection**: Tournament selection with LLM judge picks best candidate

### Trade-offs

1. **Cost**: Generates 9+ candidates per question (more LLM calls)
2. **Latency**: Sequential refinement and tournament selection add time
3. **Complexity**: Multi-stage pipeline requires careful coordination

### Optimization Tips

1. **Reduce Candidate Count**: Lower `n_reasoning_candidates` and `n_icl_candidates` for faster evaluation
2. **Skip Semantic Revision**: Comment out semantic revision step if not needed
3. **Limit Tournament**: Tournament selection can be limited to top-k candidates
4. **Cache Results**: Consider caching cell values and few-shot examples

## Comparison with Other Solutions

| Solution | Approach | Candidates | Selection | Cost | Accuracy |
|----------|----------|------------|-----------|------|----------|
| Vanilla Text2SQL | Single prompt | 1 | N/A | Low | Baseline |
| LangChain Agent | Agentic reasoning | 1 | N/A | Medium | Good |
| Cortex Analyst | Snowflake service | 1 | N/A | Medium | Good |
| **Agentar-Scale-SQL** | Multi-candidate + tournament | 9+ | LLM judge | High | Best |

## Implementation Details

### Simplifications from Original Paper

This implementation includes some simplifications:

1. **Vector Stores**: Uses in-memory stores instead of true vector databases
2. **Few-Shot Examples**: Uses generic examples instead of learned example corpus
3. **Tournament Selection**: Limited to top-k candidates for efficiency
4. **Semantic Revision**: Simplified revision logic

### Future Enhancements

1. **True Vector Stores**: Integrate FAISS or similar for cell/example retrieval
2. **Learned Examples**: Build example corpus from training data
3. **Parallel Execution**: Execute candidate generation in parallel
4. **Adaptive Candidate Count**: Dynamically adjust based on question difficulty
5. **Confidence Scoring**: Add confidence metrics for selection

## References

- Original pseudocode: `agentar-scale.md`
- Paper: "Agentar-Scale: Scaling Text-to-SQL with LLM Agents" (conceptual)
- BIRD-SQL Benchmark: [https://bird-bench.github.io/](https://bird-bench.github.io/)

## License

Same as parent project.
