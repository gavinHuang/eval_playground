# Goal
you are evaluating two solutions which converting natural languange into sql query. 

# Context
The first solution is using snowflake cortext analyst. the majority of logic is implemented in `snowflake_cortex_analyst.py` (based on https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api)
The second solution is using langchain's database agent (https://python.langchain.com/docs/tutorials/sql_qa/), the code logic is implemented in `langchain_db_agent.py`

# Requirements
- [x] Each solution exposes an interface for the evaluator module to pass a question and `evidence` and return an answer in SQL
- [x] The evaluator loops all questions in `dev.json`, only using records with "db_id": "debit_card_specializing"
- [x] The result (SQL query) is normalized to an AST which is compared with the `SQL` field (also converted to AST) in `dev.json` file
- [x] Compare the result using Exact Match, record scores for each solution, and do proper comparison in terms of precision, error distribution, difficulty breakdown, etc.

# Implementation Status: ✅ COMPLETE

## Implemented Components

1. **sql_normalizer.py** - SQL to AST converter and exact match comparator
   - Uses sqlglot for parsing
   - Handles multiple SQL dialects
   - Provides exact match comparison

2. **snowflake_cortex_analyst.py** - Snowflake Cortex Analyst client
   - REST API client implementation
   - CortexAnalystWrapper with unified interface
   - Supports OAuth and key-pair authentication

3. **langchain_db_agent.py** - LangChain database agent
   - Uses LangChain SQL agent with OpenAI
   - Unified interface matching Cortex Analyst
   - Database creation utility from CSV files

4. **evaluator.py** - Main evaluation engine
   - Loads and filters BIRD-SQL questions
   - Evaluates both solutions
   - Calculates comprehensive metrics
   - Generates detailed reports

5. **run_evaluation.py** - CLI runner script
   - Modes: full, cortex, langchain, setup
   - Automatic environment validation
   - User-friendly output

6. **test_setup.py** - Setup verification script
   - Checks dependencies
   - Validates files and configuration
   - Tests core functionality

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env file
cp .env.example .env
# Edit .env with your credentials

# 3. Verify setup
python test_setup.py

# 4. Setup database for LangChain
python run_evaluation.py --mode setup

# 5. Run evaluation
python run_evaluation.py --mode full
```

## Metrics Calculated

- **Exact Match Count & Precision**
- **Error Count & Distribution**
- **Average Execution Time**
- **Difficulty Breakdown** (simple/moderate/challenging)
- **Per-question Results**

## Output

- Console: Comparison table with key metrics
- JSON Files: Detailed results and metrics for each solution
- Location: `evaluation_results/` directory

See README.md and IMPLEMENTATION.md for complete documentation.
  
