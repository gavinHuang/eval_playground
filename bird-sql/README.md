# Text-to-SQL Evaluation Framework

This framework evaluates two Text-to-SQL solutions on the BIRD-SQL benchmark:
1. **Snowflake Cortex Analyst** - Snowflake's native semantic model-based solution
2. **LangChain DB Agent** - LangChain's OpenAI-powered database agent

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
   
4. **evaluator.py** - Main evaluation orchestrator
   - Loads questions from BIRD-SQL dev.json
   - Evaluates both solutions
   - Calculates metrics (precision, error distribution, difficulty breakdown)
   - Generates comparison reports

### Unified Interface

Both solutions expose a `query(question: str, evidence: Optional[str]) -> Dict` interface:

```python
{
    "sql": "SELECT ...",        # Generated SQL query
    "answer": "...",            # Answer/result
    "error": None               # Error message if any
}
```

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

# OpenAI for LangChain
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

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

### Run Full Evaluation

```bash
python evaluator.py
```

This will:
1. Load questions for `debit_card_specializing` from dev.json
2. Evaluate Snowflake Cortex Analyst
3. Evaluate LangChain DB Agent
4. Save detailed results and metrics
5. Print comparison report

### Test Individual Solutions

**Snowflake Cortex Analyst:**
```bash
python snowflake_cortex_analyst.py
```

**LangChain DB Agent:**
```bash
python langchain_db_agent.py
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
```

## Output Files

Results are saved to `evaluation_results/` directory:

- `snowflake_cortex_analyst_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `snowflake_cortex_analyst_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics
- `langchain_db_agent_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `langchain_db_agent_metrics_YYYYMMDD_HHMMSS.json` - Summary metrics

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
