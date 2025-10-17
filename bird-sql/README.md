# Text-to-SQL Evaluation Framework

This framework evaluates four Text-to-SQL solutions on the BIRD-SQL benchmark:
1. **Snowflake Cortex Analyst** - Snowflake's native semantic model-based solution
2. **LangChain DB Agent** - LangChain's OpenAI-powered database agent
3. **Vanilla Text2SQL** - Direct LLM prompting with schema and sample data
4. **Agentar-Scale-SQL** - State-of-the-art multi-candidate generation with tournament selection

## Architecture

### Components

1. **sql_normalizer.py** - Normalizes SQL queries to AST for exact matching
   - Uses `sqlglot` to parse SQL into Abstract Syntax Trees
   - Provides AST comparison for exact match evaluation
   
2. **snowflake_cortex_analyst.py** - Snowflake Cortex Analyst client
   - REST API client for Cortex Analyst
   - Includes `CortexAnalystWrapper` for unified interface
   
3. **langchain_db_agent.py** - LangChain database agent implementation
   - Uses LangChain's SQL agent with OpenAI models
   - Provides unified query interface matching Cortex Analyst
   
3. **vanilla_text2sql.py** - Vanilla text-to-SQL implementation
   - Direct LLM prompting with basic prompt template
   - Auto-extracts schema information (tables, columns)
   - Includes sample data (3 rows per table)
   - See [VANILLA_TEXT2SQL.md](VANILLA_TEXT2SQL.md) for details

4. **agentar_scale_sql.py** - Agentar-Scale-SQL implementation
   - State-of-the-art multi-stage pipeline
   - Task understanding with keyword/skeleton extraction
   - Parallel scaling: generates multiple diverse candidates
   - Sequential scaling: iterative refinement and error fixing
   - Tournament selection: LLM-judged pairwise comparison
   - See [AGENTAR_SCALE_SQL.md](AGENTAR_SCALE_SQL.md) for details
   
5. **evaluator.py** - Main evaluation orchestrator
   - Loads questions from BIRD-SQL dev.json
   - Evaluates all three solutions
   - Calculates metrics (precision, error distribution, difficulty breakdown)
   - Generates comparison reports

### Unified Interface

All four solutions expose a `query(question: str, evidence: Optional[str]) -> Dict` interface:

```python
{
    "sql": "SELECT ...",        # Generated SQL query
    "answer": "...",            # Answer/result
    "error": None               # Error message if any
}
```

Note: Agentar-Scale-SQL also returns a `candidates` list with all generated candidates for debugging.

## Setup

### Prerequisites

1. Python 3.8+
2. Virtual environment (recommended)

### Installation

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `bird-sql` directory:

```env
# Snowflake Cortex Analyst
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_TOKEN=your-token
SNOWFLAKE_TOKEN_TYPE=OAUTH
SNOWFLAKE_SEMANTIC_MODEL_FILE=@DB.SCHEMA.STAGE/model.yaml

# OpenAI for LangChain and Agentar-Scale-SQL
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o  # Model for Agentar-Scale-SQL (gpt-4o recommended)
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Agentar-Scale-SQL Configuration (optional)
AGENTAR_N_REASONING=3    # Number of reasoning candidates (default: 4)
AGENTAR_N_ICL=4          # Number of ICL candidates (default: 5)

# Optional: Key-pair authentication for Snowflake
SNOWFLAKE_ACCOUNT_IDENTIFIER=MYORG-MYACCOUNT
SNOWFLAKE_USERNAME=your-username
SNOWFLAKE_PRIVATE_KEY_PATH=/path/to/private_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=optional-passphrase
```

### Database Setup

For LangChain agent, create SQLite database from CSV files:

```python
from langchain_db_agent import create_debit_card_db

create_debit_card_db(
    csv_dir="debit_card_csv_export",
    db_path="debit_card.db"
)
```

## Usage

### Quick Start with run_evaluation.py

```bash
# Run all solutions
python run_evaluation.py --mode full

# Run specific solution(s)
python run_evaluation.py --mode agentar
python run_evaluation.py --mode vanilla langchain agentar

# Setup database
python run_evaluation.py --mode setup
```

Available modes:
- `full` - Run all solutions (Cortex, LangChain, Vanilla, Agentar)
- `cortex` - Snowflake Cortex Analyst only
- `langchain` - LangChain DB Agent only
- `vanilla` - Vanilla Text2SQL only
- `agentar` - Agentar-Scale-SQL only
- `setup` - Create SQLite database from CSV files

### Run Full Evaluation

```bash
python evaluator.py
```

This will:
1. Load questions for `debit_card_specializing` from dev.json
2. Evaluate Snowflake Cortex Analyst (if configured)
3. Evaluate LangChain DB Agent
4. Evaluate Vanilla Text2SQL
5. Evaluate Agentar-Scale-SQL
6. Save detailed results and metrics
7. Print comparison report
8. Export CSV comparison

### Test Individual Solutions

**Snowflake Cortex Analyst:**
```bash
python snowflake_cortex_analyst.py
```

**LangChain DB Agent:**
```bash
python langchain_db_agent.py
```

**Vanilla Text2SQL:**
```bash
python vanilla_text2sql.py
```

**Agentar-Scale-SQL:**
```bash
python agentar_scale_sql.py
```

**SQL Normalizer:**
```bash
python sql_normalizer.py
```

## Evaluation Metrics

### Per-Solution Metrics

- **Total Questions**: Number of questions evaluated
- **Exact Matches**: Number of queries matching expected SQL (AST comparison)
- **Precision**: Exact matches / Total questions
- **Errors**: Number of failed queries
- **Average Execution Time**: Mean time per successful query
- **Error Distribution**: Breakdown of error types
- **Difficulty Breakdown**: Precision by difficulty level (simple/moderate/challenging)

### Comparison Output

Example output:

```
================================================================================
EVALUATION COMPARISON
================================================================================

Snowflake Cortex Analyst
--------------------------------------------------------------------------------
Total Questions: 21
Exact Matches: 18
Precision: 85.71%
Errors: 2
Average Execution Time: 1.23s

Difficulty Breakdown:
  simple: 15/17 (88.24%)
  moderate: 3/4 (75.00%)

Error Distribution:
  HTTPError: 2

LangChain DB Agent
--------------------------------------------------------------------------------
Total Questions: 21
Exact Matches: 16
Precision: 76.19%
Errors: 3
Average Execution Time: 2.45s

Difficulty Breakdown:
  simple: 14/17 (82.35%)
  moderate: 2/4 (50.00%)

Error Distribution:
  ValueError: 2
  TimeoutError: 1

Vanilla Text2SQL
--------------------------------------------------------------------------------
Total Questions: 21
Exact Matches: 14
Precision: 66.67%
Errors: 1
Average Execution Time: 1.15s

Difficulty Breakdown:
  simple: 13/17 (76.47%)
  moderate: 1/4 (25.00%)

Error Distribution:
  ValueError: 1

Agentar-Scale-SQL
--------------------------------------------------------------------------------
Total Questions: 21
Exact Matches: 19
Precision: 90.48%
Errors: 0
Average Execution Time: 12.34s

Difficulty Breakdown:
  simple: 16/17 (94.12%)
  moderate: 3/4 (75.00%)

Error Distribution:
  (No errors)
```

## Output Files

Results are saved to `evaluation_results/` directory:

- `snowflake_cortex_analyst_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `snowflake_cortex_analyst_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics
- `langchain_db_agent_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `langchain_db_agent_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics
- `vanilla_text2sql_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `vanilla_text2sql_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics
- `agentar_scale_sql_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `agentar_scale_sql_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics
- `comparison_YYYYMMDD_HHMMSS.csv` - Question-by-question comparison (all solutions)

### Result Schema

**Detailed Results:**
```json
[
  {
    "question_id": 1470,
    "question": "How many gas stations in CZE has Premium gas?",
    "evidence": "",
    "expected_sql": "SELECT COUNT(GasStationID) FROM gasstations WHERE Country = 'CZE' AND Segment = 'Premium'",
    "generated_sql": "SELECT COUNT(GasStationID) FROM gasstations WHERE Country = 'CZE' AND Segment = 'Premium'",
    "exact_match": true,
    "error": null,
    "execution_time": 1.23,
    "difficulty": "simple"
  }
]
```

**Metrics:**
```json
{
  "solution_name": "Snowflake Cortex Analyst",
  "total_questions": 21,
  "exact_matches": 18,
  "errors": 2,
  "precision": 0.8571,
  "avg_execution_time": 1.23,
  "error_distribution": {
    "HTTPError": 2
  },
  "difficulty_breakdown": {
    "simple": {
      "total": 17,
      "matches": 15,
      "errors": 1
    },
    "moderate": {
      "total": 4,
      "matches": 3,
      "errors": 1
    }
  }
}
```

## Customization

### Evaluate Different Database

Modify `evaluator.py`:

```python
evaluator = TextToSQLEvaluator(
    dev_json_path="data/dev_20240627/dev.json",
    db_id="your_database_id",  # Change this
    sql_dialect="sqlite"       # Or postgres, mysql, etc.
)
```

### Use Different Models

For LangChain agent:

```python
langchain_results, langchain_metrics = evaluator.evaluate_langchain_agent(
    db_path="your_database.db",
    model_name="gpt-4",  # Or gpt-3.5-turbo, etc.
    api_key=os.environ.get("OPENAI_API_KEY")
)
```

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Snowflake Authentication

Verify `.env` file has correct credentials:
- `SNOWFLAKE_ACCOUNT_URL`
- `SNOWFLAKE_TOKEN`
- `SNOWFLAKE_TOKEN_TYPE`

### LangChain Agent Issues

- Ensure OpenAI API key is set
- Check database file exists
- Verify CSV files are properly formatted

### SQL Parsing Errors

The normalizer uses `sqlglot` which supports multiple SQL dialects. If queries fail to parse:
1. Check the SQL dialect setting
2. Verify SQL syntax is valid
3. Review sqlglot documentation for dialect-specific issues

## Dependencies

- `requests` - HTTP client for Snowflake API
- `python-dotenv` - Environment variable management
- `PyJWT[crypto]` - JWT token generation
- `cryptography` - Key-pair authentication
- `sqlglot` - SQL parsing and normalization
- `langchain` - LangChain framework
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI integration
- `openai` - OpenAI Python client
- `pandas` - Data manipulation for CSV loading

## References

- [Snowflake Cortex Analyst REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api)
- [LangChain SQL Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/)
- [BIRD-SQL Benchmark](https://bird-bench.github.io/)
- [sqlglot Documentation](https://github.com/tobymao/sqlglot)
