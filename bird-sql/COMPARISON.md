# Text-to-SQL Implementation Comparison

This document compares the three text-to-SQL implementations in this evaluation framework.

## Quick Overview

| Aspect | Vanilla Text2SQL | LangChain DB Agent | Snowflake Cortex Analyst |
|--------|------------------|-------------------|-------------------------|
| **Approach** | Direct prompting | Agent with tools | Managed API |
| **Complexity** | Low | Medium | Low |
| **Setup** | Database only | Database only | Snowflake + Semantic Model |
| **LLM Calls** | 1 | Multiple (agent loop) | N/A (managed) |
| **Transparency** | High | Medium | Low |
| **Customization** | High | High | Limited |
| **Token Usage** | High (schema + samples) | Variable | N/A |
| **Speed** | Fast (1 call) | Slower (multiple calls) | Depends on service |

## 1. Vanilla Text2SQL

### Architecture
```
User Question → [Schema + Sample Data + Prompt] → LLM → SQL Query
```

### How It Works
1. Extract full schema (tables, columns, types, constraints)
2. Sample 3 rows from each table
3. Build comprehensive prompt with schema, samples, and question
4. Single LLM call to generate SQL
5. Parse and validate the response

### Strengths
✅ **Simplicity**: Single file, straightforward implementation
✅ **Transparency**: Clear prompt, no hidden logic
✅ **Rich Context**: Schema + sample data helps LLM understand data
✅ **Fast**: One LLM call, no iterations
✅ **Easy to Debug**: Can inspect exact prompt sent to LLM
✅ **Customizable**: Easy to modify prompt template

### Weaknesses
⚠️ **No Self-Correction**: Can't retry or fix errors
⚠️ **Token Heavy**: Sends full schema + samples every time
⚠️ **Static Context**: Can't dynamically fetch additional info
⚠️ **No Validation Loop**: Generates query once, doesn't verify

### Best For
- Simple to moderate complexity queries
- When you want full control over prompting
- Baseline comparisons
- Quick prototyping
- Educational purposes

### Code Example
```python
text2sql = VanillaText2SQL(db_path="debit_card.db")
result = text2sql.query("How many EUR customers?")
# Single LLM call with full context
```

## 2. LangChain DB Agent

### Architecture
```
User Question → Agent → [Plan] → Tools (query/schema/list) → [Observe] → SQL
                  ↑                                                    ↓
                  └────────────────── Iterate ──────────────────────────┘
```

### How It Works
1. Agent receives question
2. Plans which tools to use
3. Calls tools (sql_db_schema, sql_db_list_tables, sql_db_query)
4. Observes results
5. Iterates until final answer
6. Returns SQL and answer

### Strengths
✅ **Self-Correcting**: Can retry queries if errors occur
✅ **Dynamic**: Fetches only needed schema information
✅ **Flexible**: Can explore database, check results, refine
✅ **Token Efficient**: Only loads relevant schema
✅ **Robust**: Handles complex multi-step reasoning
✅ **Built-in Tools**: Schema inspection, query execution, validation

### Weaknesses
⚠️ **Complexity**: Multiple LLM calls, harder to debug
⚠️ **Slower**: Agent loop adds latency
⚠️ **Non-Deterministic**: Can take different paths
⚠️ **Higher Cost**: Multiple LLM calls per query
⚠️ **Opaque**: Hard to predict exact behavior

### Best For
- Complex queries requiring exploration
- When schema is large and token efficiency matters
- Production systems needing robustness
- Queries that might need refinement

### Code Example
```python
agent = LangChainDBAgent(db_path="debit_card.db")
result = agent.query("How many EUR customers?")
# Multiple LLM calls with tool iterations
```

## 3. Snowflake Cortex Analyst

### Architecture
```
User Question → REST API → Snowflake Managed Service → SQL + Answer
                              (Semantic Model)
```

### How It Works
1. Send question + semantic model reference to API
2. Snowflake's managed service processes request
3. Uses semantic model (business definitions, relationships)
4. Returns SQL and natural language answer
5. Provides confidence scores

### Strengths
✅ **Semantic Understanding**: Business logic in semantic model
✅ **Managed**: No LLM infrastructure needed
✅ **Optimized**: Snowflake-specific optimizations
✅ **Confidence Scores**: Quality indicators
✅ **Business Context**: Non-technical users can define semantics
✅ **No Token Costs**: Managed service pricing

### Weaknesses
⚠️ **Snowflake Only**: Requires Snowflake account
⚠️ **Semantic Model Required**: Must create and maintain YAML
⚠️ **Less Control**: Can't modify underlying logic
⚠️ **API Dependency**: Network latency, availability
⚠️ **Limited Transparency**: Black box processing

### Best For
- Snowflake users
- Business intelligence applications
- When semantic layer already exists
- Non-technical user queries

### Code Example
```python
client = CortexAnalystWrapper(
    account_url="https://account.snowflake.com",
    token=token,
    semantic_model_file="@stage/model.yaml"
)
result = client.query("How many EUR customers?")
# Managed service handles everything
```

## Detailed Comparison

### Schema Handling

**Vanilla Text2SQL:**
- Extracts full schema at initialization
- Includes all tables, columns, types, constraints
- Sends everything in every prompt
- Pro: Complete context
- Con: Token intensive

**LangChain DB Agent:**
- Dynamically queries schema as needed
- Uses `sql_db_schema` tool for specific tables
- Only loads relevant portions
- Pro: Token efficient
- Con: Might miss relationships

**Cortex Analyst:**
- Uses pre-defined semantic model
- Includes business logic, relationships, definitions
- Must be maintained separately
- Pro: Business-friendly
- Con: Extra maintenance

### Sample Data

**Vanilla Text2SQL:**
- ✅ Automatically includes 3 rows per table
- Helps LLM understand data format and values
- Increases context quality

**LangChain DB Agent:**
- ⚠️ Can query sample data via tools
- Not included by default
- Agent must decide to fetch samples

**Cortex Analyst:**
- ❌ No sample data in semantic model
- Relies on column definitions and types

### Error Handling

**Vanilla Text2SQL:**
```python
# Single attempt, no retry
try:
    sql = llm.generate(prompt)
    result = execute(sql)
except:
    return error
```

**LangChain DB Agent:**
```python
# Agent can retry and self-correct
try:
    sql = llm.generate()
    result = execute(sql)
except SQLError:
    # Agent observes error
    # Generates new SQL
    sql_v2 = llm.generate_with_error_context()
    result = execute(sql_v2)
```

**Cortex Analyst:**
```python
# Managed service handles errors internally
# Returns confidence score
result = api.query(question)
if result.confidence < threshold:
    # Handle low confidence
```

### Prompt Engineering

**Vanilla Text2SQL:**
- Full control over prompt template
- Easy to modify instructions
- Can add domain-specific guidance
- Example:
```python
template = """
DATABASE SCHEMA:
{schema}

SAMPLE DATA:
{samples}

Generate SQLite query for: {question}
"""
```

**LangChain DB Agent:**
- System prompt + tool descriptions
- Less direct control
- Can customize system message
- Agent decides tool usage

**Cortex Analyst:**
- No direct prompt control
- Guidance through semantic model
- Business definitions replace technical prompts

## Performance Considerations

### Latency

**Vanilla Text2SQL:**
- ⚡ Fastest: Single LLM call
- ~1-2 seconds typical
- Depends on model and context size

**LangChain DB Agent:**
- 🐢 Slowest: Multiple LLM calls
- ~3-5+ seconds typical
- Variable based on agent iterations

**Cortex Analyst:**
- ⚡ Fast: Optimized managed service
- ~1-3 seconds typical
- Depends on network and service load

### Cost

**Vanilla Text2SQL:**
- 💰 Moderate: High tokens per call (schema + samples)
- Single call keeps cost predictable
- Example: ~2000 tokens input, ~100 tokens output

**LangChain DB Agent:**
- 💰💰 Higher: Multiple smaller calls
- Variable cost based on iterations
- Example: 3-5 calls × ~500 tokens each

**Cortex Analyst:**
- 💰 Different model: Service pricing, not token-based
- Predictable cost per query
- No token counting needed

### Accuracy

Depends heavily on:
- Query complexity
- Schema clarity
- Model capability
- Sample data quality

**General trends** (from evaluations):
1. **Cortex Analyst**: Highest (semantic model helps)
2. **LangChain Agent**: Medium-High (self-correction helps)
3. **Vanilla Text2SQL**: Medium (single-shot limits)

## When to Use Each

### Choose Vanilla Text2SQL When:
- ✅ Building a baseline or prototype
- ✅ You need full control over prompting
- ✅ Schema is small (fits in context easily)
- ✅ Queries are simple to moderate complexity
- ✅ You want transparency and debuggability
- ✅ Single-shot generation is sufficient

### Choose LangChain DB Agent When:
- ✅ Queries are complex and exploratory
- ✅ Schema is large (need selective loading)
- ✅ Self-correction and robustness are important
- ✅ You can tolerate higher latency
- ✅ You want built-in error recovery
- ✅ Working with any SQL database (SQLite, Postgres, MySQL, etc.)

### Choose Cortex Analyst When:
- ✅ You're already using Snowflake
- ✅ You have or need a semantic layer
- ✅ Business users define query semantics
- ✅ You want managed infrastructure
- ✅ Confidence scores are valuable
- ✅ You prefer API-based solutions

## Migration Path

### From Vanilla → LangChain Agent
```python
# Vanilla
text2sql = VanillaText2SQL(db_path="db.db")

# To LangChain
agent = LangChainDBAgent(db_path="db.db")
# Interface is the same!
```

### From LangChain → Cortex Analyst
1. Create semantic model YAML
2. Upload to Snowflake stage
3. Switch to API client
```python
# LangChain
agent = LangChainDBAgent(db_path="db.db")

# To Cortex
client = CortexAnalystWrapper(
    account_url=url,
    token=token,
    semantic_model_file="@stage/model.yaml"
)
# Interface is the same!
```

## Hybrid Approaches

You can combine approaches:

### Vanilla + Validation
```python
# Generate with vanilla
result = vanilla.query(question)
sql = result['sql']

# Validate with agent
validation = agent.validate_sql(sql)
if not validation['valid']:
    sql = agent.query(question)['sql']
```

### Agent + Semantic Context
```python
# Load semantic definitions
semantics = load_semantic_model()

# Inject into agent prompt
agent = LangChainDBAgent(
    db_path="db.db",
    system_prompt=f"Use these definitions: {semantics}"
)
```

## Conclusion

Each approach has its place:

- **Vanilla Text2SQL**: Best for learning, prototyping, and simple use cases
- **LangChain DB Agent**: Best for production, complex queries, and robustness
- **Cortex Analyst**: Best for Snowflake users wanting managed solutions

The unified interface in this framework makes it easy to evaluate and compare all three!
