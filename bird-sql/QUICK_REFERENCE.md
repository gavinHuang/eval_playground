# Quick Reference Guide

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Snowflake
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_TOKEN=your-token
SNOWFLAKE_TOKEN_TYPE=OAUTH
SNOWFLAKE_SEMANTIC_MODEL_FILE=@DB.SCHEMA.STAGE/model.yaml

# OpenAI
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
```

## Quick Start

```bash
# 1. Setup database for LangChain
python run_evaluation.py --mode setup

# 2. Run full evaluation
python run_evaluation.py --mode full

# 3. Check results
ls evaluation_results/
```

## Running Evaluations

### Evaluate Both Solutions
```bash
python run_evaluation.py --mode full
```

### Evaluate Cortex Analyst Only
```bash
python run_evaluation.py --mode cortex
```

### Evaluate LangChain Only
```bash
python run_evaluation.py --mode langchain
```

## Testing Individual Components

### Test SQL Normalizer
```bash
python sql_normalizer.py
```

### Test Cortex Analyst
```bash
python snowflake_cortex_analyst.py
```

### Test LangChain Agent
```bash
python langchain_db_agent.py
```

### Test Full Evaluator
```bash
python evaluator.py
```

## Common Commands

### Create Database from CSVs
```python
from langchain_db_agent import create_debit_card_db

create_debit_card_db(
    csv_dir="debit_card_csv_export",
    db_path="debit_card.db"
)
```

### Run Single Query (Cortex Analyst)
```python
from snowflake_cortex_analyst import CortexAnalystWrapper
import os

client = CortexAnalystWrapper(
    account_url=os.environ.get("SNOWFLAKE_ACCOUNT_URL"),
    token=os.environ.get("SNOWFLAKE_TOKEN"),
    semantic_model_file=os.environ.get("SNOWFLAKE_SEMANTIC_MODEL_FILE")
)

result = client.query("How many gas stations in CZE has Premium gas?")
print(result['sql'])
```

### Run Single Query (LangChain)
```python
from langchain_db_agent import LangChainDBAgent
import os

agent = LangChainDBAgent(
    db_path="debit_card.db",
    api_key=os.environ.get("OPENAI_API_KEY")
)

result = agent.query("How many gas stations in CZE has Premium gas?")
print(result['sql'])
```

### Compare Two SQL Queries
```python
from sql_normalizer import SQLNormalizer

normalizer = SQLNormalizer(dialect="sqlite")

sql1 = "SELECT COUNT(*) FROM gasstations WHERE Country='CZE'"
sql2 = "select count(*) from gasstations where Country = 'CZE'"

match = normalizer.compare_ast(sql1, sql2)
print(f"Match: {match}")  # True
```

## Output Files

### Result Files Location
```
evaluation_results/
├── snowflake_cortex_analyst_results_YYYYMMDD_HHMMSS.json
├── snowflake_cortex_analyst_metrics_YYYYMMDD_HHMMSS.json
├── langchain_db_agent_results_YYYYMMDD_HHMMSS.json
└── langchain_db_agent_metrics_YYYYMMDD_HHMMSS.json
```

### Viewing Results
```bash
# Pretty print JSON results
python -m json.tool evaluation_results/snowflake_cortex_analyst_metrics_*.json

# Or use jq (if installed)
jq '.' evaluation_results/snowflake_cortex_analyst_metrics_*.json
```

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables Not Set
```bash
# Check if .env exists
ls .env

# If not, copy from example
cp .env.example .env

# Edit .env with your credentials
notepad .env  # Windows
nano .env     # Linux/Mac
```

### Database Not Found (LangChain)
```bash
python run_evaluation.py --mode setup
```

### Import Errors
```bash
# Verify setup
python test_setup.py

# Ensure you're in the correct directory
cd bird-sql
```

### Snowflake Authentication Errors
1. Check `SNOWFLAKE_ACCOUNT_URL` is correct
2. Verify `SNOWFLAKE_TOKEN` is valid and not expired
3. Ensure `SNOWFLAKE_TOKEN_TYPE` matches your authentication method
4. For key-pair auth, check private key file path and passphrase

### OpenAI API Errors
1. Verify `OPENAI_API_KEY` is set and valid
2. Check API quota/rate limits
3. Try different model with `OPENAI_MODEL` env var

## File Structure

```
bird-sql/
├── snowflake_cortex_analyst.py    # Cortex Analyst client
├── langchain_db_agent.py          # LangChain agent
├── sql_normalizer.py              # SQL AST comparison
├── evaluator.py                   # Evaluation engine
├── run_evaluation.py              # CLI runner
├── test_setup.py                  # Setup verification
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── IMPLEMENTATION.md              # Implementation details
├── QUICK_REFERENCE.md             # This file
├── .env.example                   # Config template
├── .env                           # Your config (create this)
├── data/
│   └── dev_20240627/
│       └── dev.json               # BIRD-SQL questions
├── debit_card_csv_export/         # CSV data
│   ├── customers.csv
│   ├── gasstations.csv
│   ├── products.csv
│   ├── transactions_1k.csv
│   └── yearmonth.csv
├── debit_card.db                  # SQLite DB (created)
└── evaluation_results/            # Output (created)
    └── *.json
```

## Key Concepts

### Exact Match
SQL queries are compared using AST (Abstract Syntax Tree) comparison, which ignores:
- Whitespace differences
- Capitalization (for keywords)
- Alias ordering
- Comment differences

### Evaluation Metrics
- **Precision**: Exact matches / Total questions
- **Error Rate**: Errors / Total questions
- **Execution Time**: Average time per successful query
- **Difficulty Breakdown**: Precision by question difficulty

### Difficulty Levels
- **simple**: Basic queries (single table, simple conditions)
- **moderate**: Multi-table joins, aggregations
- **challenging**: Complex subqueries, nested logic

## Best Practices

1. **Always verify setup** before running evaluations:
   ```bash
   python test_setup.py
   ```

2. **Start with small tests** before full evaluation:
   ```bash
   # Test individual components first
   python sql_normalizer.py
   python snowflake_cortex_analyst.py
   python langchain_db_agent.py
   ```

3. **Monitor execution** - evaluations can take time:
   ```bash
   # Use mode flags to test one solution at a time
   python run_evaluation.py --mode cortex
   ```

4. **Review results systematically**:
   - Check console output for high-level metrics
   - Examine JSON files for detailed per-question results
   - Compare error distributions to identify issues

5. **Keep credentials secure**:
   - Never commit `.env` file to version control
   - Use `.env.example` for sharing configuration templates
   - Rotate API keys regularly

## Resources

- [BIRD-SQL Benchmark](https://bird-bench.github.io/)
- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api)
- [LangChain SQL Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/)
- [sqlglot Documentation](https://github.com/tobymao/sqlglot)
