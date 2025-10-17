# Quick Start Guide: Agentar-Scale-SQL

## Prerequisites

1. Activate virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

2. Set environment variables (create `.env` file or export):
```bash
# Required
OPENAI_API_KEY=sk-...
# OR
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Optional
OPENAI_MODEL=gpt-4o
AGENTAR_N_REASONING=3
AGENTAR_N_ICL=4
```

3. Create database (if not exists):
```bash
python run_evaluation.py --mode setup
```

## Quick Test

Test with a single question (fast):
```bash
python test_agentar_scale_sql.py
```

Expected time: ~30-60 seconds
Expected output: Final SQL, answer, and candidate summary

## Run Evaluation

### Option 1: Run Agentar-Scale-SQL Only
```bash
python run_evaluation.py --mode agentar
```

Expected time: ~5-10 minutes (21 questions)
Expected output: 
- JSON results in `evaluation_results/`
- Metrics summary
- Question-by-question status

### Option 2: Compare All Solutions
```bash
python run_evaluation.py --mode full
```

Expected time: ~15-30 minutes (all 4 solutions)
Expected output:
- JSON results for each solution
- CSV comparison file
- Comparative metrics

### Option 3: Compare Specific Solutions
```bash
python run_evaluation.py --mode vanilla agentar
```

Expected time: ~8-15 minutes (2 solutions)

## Understanding Output

### Console Output

```
[Agentar-Scale-SQL] Processing question: How many gas stations...
[Step 1] Task Understanding...
[Step 2] SQL Generation Scaling...
  Generating 3 reasoning-based candidates...
  Generating 4 ICL-based candidates...
  Refining 7 candidates...
[Step 3] SQL Selection Scaling...
  Consolidated 7 candidates to 4 unique results
  Running tournament selection on 4 candidates...
[Complete] Selected best candidate from 7 total candidates
Processed question 1470: ✓
```

Status icons:
- `✓` = Exact SQL match
- `≈` = Result match (different SQL, same result)
- `✗` = No match

### Files Generated

```
evaluation_results/
├── agentar_scale_sql_results_20250117_143022.json    # Detailed results
├── agentar_scale_sql_metrics_20250117_143022.json    # Summary metrics
└── comparison_20250117_143022.csv                     # CSV comparison (if multiple solutions)
```

### Metrics Summary

```
Agentar-Scale-SQL
--------------------------------------------------------------------------------
Total Questions: 21
Exact Matches: 19           # SQL matches expected
Result Matches: 20          # Results match (more important!)
Result Accuracy: 95.24%     # Key metric
Errors: 0
Average Execution Time: 12.34s
```

## Troubleshooting

### Issue: "Database not found"
```bash
python run_evaluation.py --mode setup
```

### Issue: "Missing API key"
Set environment variable:
```powershell
$env:OPENAI_API_KEY="sk-..."
```

### Issue: "Too slow"
Reduce candidate counts:
```powershell
$env:AGENTAR_N_REASONING="2"
$env:AGENTAR_N_ICL="3"
```

### Issue: "Rate limit errors"
Use slower model or add delays:
```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

## Cost Estimation

Approximate costs per question (GPT-4o):

| Candidates | LLM Calls | Est. Tokens | Est. Cost |
|------------|-----------|-------------|-----------|
| 7 (3+4)    | ~15-20    | ~15,000     | $0.15-0.30 |
| 9 (4+5)    | ~20-25    | ~20,000     | $0.20-0.40 |

Full evaluation (21 questions):
- With 7 candidates: ~$3-6
- With 9 candidates: ~$4-8

Using GPT-4o-mini reduces cost by ~90%.

## Next Steps

1. **Review Results**: Check JSON files for detailed candidate information
2. **Compare Solutions**: Look at CSV comparison to see differences
3. **Analyze Failures**: Identify patterns in failed questions
4. **Tune Parameters**: Adjust candidate counts based on accuracy/cost trade-off
5. **Try Different Models**: Test with GPT-4, GPT-4-turbo, etc.

## Advanced Usage

### Custom Evaluation
```python
from evaluator import TextToSQLEvaluator

evaluator = TextToSQLEvaluator(
    dev_json_path="data/dev_20240627/dev.json",
    db_id="debit_card_specializing",
    db_path="debit_card.db"
)

results, metrics = evaluator.evaluate_agentar_scale_sql(
    db_path="debit_card.db",
    model_name="gpt-4o",
    n_reasoning_candidates=4,
    n_icl_candidates=5
)

evaluator.save_results(results, metrics, "evaluation_results")
evaluator.print_comparison([metrics])
```

### Single Question Test
```python
from agentar_scale_sql import AgentarScaleSQL

agentar = AgentarScaleSQL(
    db_path="debit_card.db",
    n_reasoning_candidates=2,
    n_icl_candidates=3
)

result = agentar.query(
    question="How many gas stations in CZE has Premium gas?",
    evidence=""
)

print(result['sql'])
print(result['answer'])
```

## Support

- Main documentation: `AGENTAR_SCALE_SQL.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- General framework: `README.md`
- Issue: Check error messages and traceback
