# Azure OpenAI Support

The evaluation framework now supports both **standard OpenAI** and **Azure OpenAI** for the LangChain DB Agent.

## Configuration

### Option 1: Standard OpenAI (Default)

Configure in `.env`:
```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
```

Optional custom endpoint:
```env
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Option 2: Azure OpenAI

Configure in `.env`:
```env
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

## Auto-Detection

The framework automatically detects which provider to use:
- If `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_API_KEY` is set → **Azure OpenAI**
- Otherwise → **Standard OpenAI**

You can also explicitly specify the provider when creating the agent:

```python
from langchain_db_agent import LangChainDBAgent

# Explicit Azure OpenAI
agent = LangChainDBAgent(
    db_path="debit_card.db",
    use_azure=True,
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4",
    api_key="your-azure-api-key",
    api_version="2024-08-01-preview"
)

# Explicit Standard OpenAI
agent = LangChainDBAgent(
    db_path="debit_card.db",
    use_azure=False,
    model_name="gpt-4o-mini",
    api_key="your-openai-api-key"
)
```

## Environment Variables

### Standard OpenAI
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model name (default: gpt-4o-mini)
- `OPENAI_BASE_URL` - Custom API endpoint (optional)

### Azure OpenAI
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key (required)
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL (required)
- `AZURE_OPENAI_DEPLOYMENT` - Deployment name (optional, defaults to model_name)
- `AZURE_OPENAI_API_VERSION` - API version (optional, defaults to "2024-08-01-preview")

## Usage Examples

### Running Evaluation with Azure OpenAI

1. Configure `.env` with Azure credentials:
```env
AZURE_OPENAI_API_KEY=abc123...
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

2. Run evaluation:
```bash
python run_evaluation.py --mode langchain
```

The framework will automatically detect and use Azure OpenAI.

### Programmatic Usage

```python
from evaluator import TextToSQLEvaluator

evaluator = TextToSQLEvaluator(
    dev_json_path="data/dev_20240627/dev.json",
    db_id="debit_card_specializing"
)

# Azure OpenAI
results, metrics = evaluator.evaluate_langchain_agent(
    db_path="debit_card.db",
    use_azure=True,
    azure_endpoint="https://myresource.openai.azure.com/",
    azure_deployment="gpt-4",
    api_key="your-azure-api-key"
)
```

## Deployment Names vs Model Names

### Azure OpenAI
- Uses **deployment names** (e.g., "gpt-4", "my-gpt-35-turbo")
- The deployment name is what you configured in Azure Portal
- Pass via `azure_deployment` parameter or `AZURE_OPENAI_DEPLOYMENT` env var

### Standard OpenAI
- Uses **model names** (e.g., "gpt-4o-mini", "gpt-4-turbo")
- These are the official OpenAI model identifiers
- Pass via `model_name` parameter or `OPENAI_MODEL` env var

## API Version

Azure OpenAI requires an API version. The framework defaults to `2024-08-01-preview`.

To use a different version:
```env
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Or programmatically:
```python
agent = LangChainDBAgent(
    db_path="debit_card.db",
    use_azure=True,
    api_version="2024-02-15-preview"
)
```

## Troubleshooting

### Error: "Resource not found"
- Check `AZURE_OPENAI_ENDPOINT` is correct
- Ensure endpoint ends with `/` (e.g., `https://myresource.openai.azure.com/`)

### Error: "Deployment not found"
- Verify `AZURE_OPENAI_DEPLOYMENT` matches your Azure deployment name
- Check deployment is active in Azure Portal

### Error: "Invalid API version"
- Update `AZURE_OPENAI_API_VERSION` to a supported version
- See [Azure OpenAI API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

### Using wrong provider
- Check which env vars are set (Azure vs OpenAI)
- Explicitly set `use_azure=True` or `use_azure=False` to override auto-detection

## Migration from Standard OpenAI to Azure OpenAI

1. **Update `.env` file**:
   ```env
   # Comment out or remove OpenAI config
   # OPENAI_API_KEY=...
   
   # Add Azure config
   AZURE_OPENAI_API_KEY=your-azure-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name
   ```

2. **No code changes required** - auto-detection handles the switch

3. **Update model/deployment name** if needed:
   ```env
   OPENAI_MODEL=your-azure-deployment-name
   ```

## Complete Example

### .env file for Azure OpenAI
```env
# Snowflake configuration
SNOWFLAKE_ACCOUNT_URL=https://your-account.snowflakecomputing.com
SNOWFLAKE_TOKEN=your-token
SNOWFLAKE_TOKEN_TYPE=OAUTH
SNOWFLAKE_SEMANTIC_MODEL_FILE=@DB.SCHEMA.STAGE/model.yaml

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY=1234567890abcdef...
AZURE_OPENAI_ENDPOINT=https://mycompany-openai.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4-deployment
AZURE_OPENAI_API_VERSION=2024-08-01-preview
OPENAI_MODEL=gpt-4-deployment
```

### Running evaluation
```bash
# Verify setup
python test_setup.py

# Setup database
python run_evaluation.py --mode setup

# Run evaluation with Azure OpenAI
python run_evaluation.py --mode langchain
```

## References

- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [LangChain Azure OpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai)
- [Azure OpenAI API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
