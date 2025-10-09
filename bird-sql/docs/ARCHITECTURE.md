# Architecture Diagram

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EVALUATION FRAMEWORK                          │
└──────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │  run_evaluation  │
                        │      .py         │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   evaluator.py   │
                        │                  │
                        │ - Load questions │
                        │ - Run solutions  │
                        │ - Compare AST    │
                        │ - Calculate      │
                        │   metrics        │
                        └────────┬─────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │ CortexAnalystWrapper   │      │  LangChainDBAgent      │
    │                        │      │                        │
    │ ┌────────────────────┐ │      │ ┌────────────────────┐ │
    │ │ Snowflake Cortex   │ │      │ │   LangChain         │ │
    │ │ Analyst REST API   │ │      │ │   SQL Agent         │ │
    │ │                    │ │      │ │   + OpenAI          │ │
    │ └────────────────────┘ │      │ └────────────────────┘ │
    │                        │      │                        │
    │ query(question,        │      │ query(question,        │
    │       evidence)        │      │       evidence)        │
    │    ↓                   │      │    ↓                   │
    │ Returns:               │      │ Returns:               │
    │ {                      │      │ {                      │
    │   sql: "SELECT...",    │      │   sql: "SELECT...",    │
    │   answer: "...",       │      │   answer: "...",       │
    │   error: null          │      │   error: null          │
    │ }                      │      │ }                      │
    └────────────────────────┘      └────────────────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ sql_normalizer   │
                        │      .py         │
                        │                  │
                        │ - Parse SQL      │
                        │ - Build AST      │
                        │ - Compare AST    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   Exact Match    │
                        │   Comparison     │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Results & Stats │
                        │                  │
                        │ - Precision      │
                        │ - Errors         │
                        │ - Execution Time │
                        │ - Difficulty     │
                        └────────┬─────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │   JSON Files           │      │  Console Output        │
    │                        │      │                        │
    │ - *_results.json       │      │ ┌────────────────────┐ │
    │ - *_metrics.json       │      │ │ Comparison Table   │ │
    │                        │      │ │                    │ │
    │ Saved to:              │      │ │ Cortex: 85.71%     │ │
    │ evaluation_results/    │      │ │ LangChain: 76.19%  │ │
    │                        │      │ └────────────────────┘ │
    └────────────────────────┘      └────────────────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Load Data                                                   │
└─────────────────────────────────────────────────────────────────────┘

        dev.json (BIRD-SQL benchmark)
               │
               ▼
        Filter: db_id = "debit_card_specializing"
               │
               ▼
        21 Questions with Expected SQL

┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: For Each Question                                           │
└─────────────────────────────────────────────────────────────────────┘

Question: "How many gas stations in CZE has Premium gas?"
Evidence: ""
Expected SQL: "SELECT COUNT(GasStationID) FROM gasstations 
               WHERE Country = 'CZE' AND Segment = 'Premium'"
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼                         ▼
    Solution 1             Solution 2               Expected SQL
    (Cortex)              (LangChain)
        │                      │                         │
        ▼                      ▼                         ▼
    Generated SQL          Generated SQL              Parse to AST
        │                      │                         │
        ▼                      ▼                         ▼
    Parse to AST           Parse to AST                AST_expected
        │                      │                         │
        ▼                      ▼                         ▼
    Compare with ──────────────┴─────────────────────────┘
    AST_expected
        │
        ▼
    Exact Match: True/False

┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Aggregate Results                                           │
└─────────────────────────────────────────────────────────────────────┘

    For each solution:
        ├─ Total Questions: 21
        ├─ Exact Matches: Count
        ├─ Precision: Matches / Total
        ├─ Errors: Count
        ├─ Avg Execution Time: Mean
        ├─ Error Distribution: {ErrorType: Count}
        └─ Difficulty Breakdown:
            ├─ simple: X/Y (Z%)
            └─ moderate: A/B (C%)

┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Output                                                      │
└─────────────────────────────────────────────────────────────────────┘

    JSON Files:
        ├─ snowflake_cortex_analyst_results_*.json
        │   └─ [Detailed per-question results]
        ├─ snowflake_cortex_analyst_metrics_*.json
        │   └─ {Aggregate metrics}
        ├─ langchain_db_agent_results_*.json
        │   └─ [Detailed per-question results]
        └─ langchain_db_agent_metrics_*.json
            └─ {Aggregate metrics}

    Console:
        └─ Comparison Table (both solutions side-by-side)
```

## Component Interaction

```
┌──────────────────────────────────────────────────────────────────┐
│                        Unified Interface                          │
│                                                                   │
│  query(question: str, evidence: str) -> Dict                     │
│                                                                   │
│  Returns:                                                         │
│  {                                                                │
│    "sql": "SELECT ...",        # Generated SQL query             │
│    "answer": "...",            # Answer/result                   │
│    "error": None               # Error message if any            │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
                             ▲
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        │                                         │
┌───────┴──────────┐                    ┌────────┴─────────┐
│ Cortex Analyst   │                    │ LangChain Agent  │
│    Wrapper       │                    │                  │
├──────────────────┤                    ├──────────────────┤
│                  │                    │                  │
│ - Snowflake      │                    │ - SQLDatabase    │
│   REST API       │                    │ - ChatOpenAI     │
│ - Semantic       │                    │ - SQL Agent      │
│   Model          │                    │ - SQLite DB      │
│                  │                    │                  │
└──────────────────┘                    └──────────────────┘
```

## SQL Normalization Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL Normalization                            │
└─────────────────────────────────────────────────────────────────┘

Input SQL 1:                    Input SQL 2:
"SELECT COUNT(*)                "select count(*)
 FROM gasstations                from gasstations
 WHERE Country='CZE'"            where Country = 'CZE'"

        │                               │
        └───────────┬───────────────────┘
                    ▼
            ┌─────────────────┐
            │  sqlglot Parse  │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │   Build AST     │
            └────────┬────────┘
                     ▼
            {
              "type": "Select",
              "expressions": [
                {
                  "type": "Count",
                  "this": {"type": "Star"}
                }
              ],
              "from": {
                "type": "From",
                "this": {
                  "type": "Table",
                  "this": "GASSTATIONS"
                }
              },
              "where": {
                "type": "Where",
                "this": {
                  "type": "EQ",
                  "this": "COUNTRY",
                  "expression": "CZE"
                }
              }
            }
                     │
                     ▼
            ┌─────────────────┐
            │  Compare ASTs   │
            └────────┬────────┘
                     ▼
              Exact Match: TRUE
```

## File Dependencies

```
run_evaluation.py
    │
    ├─→ evaluator.py
    │       │
    │       ├─→ sql_normalizer.py
    │       │       └─→ sqlglot (package)
    │       │
    │       ├─→ snowflake_cortex_analyst.py
    │       │       ├─→ requests (package)
    │       │       ├─→ PyJWT (package)
    │       │       └─→ cryptography (package)
    │       │
    │       └─→ langchain_db_agent.py
    │               ├─→ langchain (package)
    │               ├─→ langchain-community (package)
    │               ├─→ langchain-openai (package)
    │               ├─→ openai (package)
    │               └─→ pandas (package)
    │
    └─→ .env (configuration)
            ├─ SNOWFLAKE_ACCOUNT_URL
            ├─ SNOWFLAKE_TOKEN
            ├─ SNOWFLAKE_SEMANTIC_MODEL_FILE
            └─ OPENAI_API_KEY

test_setup.py
    │
    └─→ [Validates all above dependencies]

data/dev_20240627/dev.json
    │
    └─→ [BIRD-SQL benchmark questions]

debit_card_csv_export/*.csv
    │
    └─→ langchain_db_agent.create_debit_card_db()
            │
            └─→ debit_card.db
```

## Evaluation Metrics Calculation

```
┌─────────────────────────────────────────────────────────────────┐
│                   Metrics Calculation                           │
└─────────────────────────────────────────────────────────────────┘

For each solution:

Total Questions = 21

Exact Matches = Σ(result.exact_match == True)

Precision = Exact Matches / Total Questions
          = 18 / 21
          = 0.8571
          = 85.71%

Errors = Σ(result.error != None)
       = 2

Avg Execution Time = Mean(result.execution_time for successful)
                   = (1.2 + 1.3 + 1.1 + ...) / 19
                   = 1.23s

Error Distribution = {
  "HTTPError": 2
}

Difficulty Breakdown = {
  "simple": {
    "total": 17,
    "matches": 15,
    "errors": 1,
    "precision": 15/17 = 88.24%
  },
  "moderate": {
    "total": 4,
    "matches": 3,
    "errors": 1,
    "precision": 3/4 = 75.00%
  }
}
```
