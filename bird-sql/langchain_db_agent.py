"""
LangChain Database Agent for Text-to-SQL

This module implements a SQL generation solution using LangChain's database agent.
Reference: https://python.langchain.com/docs/tutorials/sql_qa/
"""

import os
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Any
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.prebuilt import create_react_agent
from pydantic import ConfigDict


class SQLCaptureCallback(BaseCallbackHandler):
    """Callback to capture SQL queries from agent tools"""
    
    def __init__(self):
        self.sql_queries = []
        self.debug = False  # Set to True for debugging
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Capture SQL when sql_db_query tool is used"""
        tool_name = serialized.get('name', '')
        
        if self.debug:
            print(f"\n[DEBUG] Tool: {tool_name}, Input type: {type(input_str)}, Input: {input_str}")
        
        if tool_name == 'sql_db_query':
            # Handle both string and dict inputs
            if isinstance(input_str, dict):
                query = input_str.get('query', input_str.get('sql', ''))
                if query:
                    self.sql_queries.append(query)
            elif isinstance(input_str, str):
                # The input_str might be a string representation of a dict like "{'query': 'SELECT...'}"
                # First try to parse it as JSON
                try:
                    import json
                    # Try to convert Python dict string to JSON-parseable format
                    # Replace single quotes with double quotes for JSON compatibility
                    json_str = input_str.replace("'", '"')
                    input_dict = json.loads(json_str)
                    query = input_dict.get('query', input_dict.get('sql', ''))
                    if query:
                        self.sql_queries.append(query)
                except (json.JSONDecodeError, AttributeError, ValueError):
                    # If that fails, try to evaluate it as Python literal
                    try:
                        import ast
                        input_dict = ast.literal_eval(input_str)
                        if isinstance(input_dict, dict):
                            query = input_dict.get('query', input_dict.get('sql', ''))
                            if query:
                                self.sql_queries.append(query)
                        else:
                            # Not a dict, treat as plain string
                            self.sql_queries.append(input_str)
                    except (ValueError, SyntaxError):
                        # Not a dict representation, treat as plain SQL string
                        self.sql_queries.append(input_str)


class ReasoningModelWrapper(BaseChatModel):
    """Wrapper for reasoning models that don't support certain parameters"""
    
    _llm: BaseChatModel
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, llm: BaseChatModel):
        super().__init__()
        object.__setattr__(self, '_llm', llm)
    
    def _generate(self, messages: List[BaseMessage], stop: Any = None, **kwargs) -> ChatResult:
        """Override to remove unsupported parameters"""
        # Remove 'stop' parameter for reasoning models
        kwargs.pop('stop', None)
        return self._llm._generate(messages, **kwargs)
    
    async def _agenerate(self, messages: List[BaseMessage], stop: Any = None, **kwargs) -> ChatResult:
        """Async override to remove unsupported parameters"""
        kwargs.pop('stop', None)
        return await self._llm._agenerate(messages, **kwargs)
    
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ReasoningModelWrapper":
        """Bind tools to the wrapped model"""
        bound_llm = self._llm.bind_tools(tools, **kwargs)
        return ReasoningModelWrapper(bound_llm)
    
    @property
    def _llm_type(self) -> str:
        return self._llm._llm_type
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return getattr(self._llm, '_identifying_params', {})


class LangChainDBAgent:
    """LangChain-based database agent for text-to-SQL"""
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "gpt-4.1",
        temperature: float = 0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_azure: Optional[bool] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize the LangChain database agent
        
        Args:
            db_path: Path to SQLite database
            model_name: OpenAI model name (or Azure deployment name if use_azure=True)
            temperature: Model temperature (0 for deterministic)
            api_key: OpenAI/Azure API key (defaults to OPENAI_API_KEY or AZURE_OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (optional, for OpenAI only)
            use_azure: Whether to use Azure OpenAI (auto-detected from env vars if not specified)
            azure_endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            azure_deployment: Azure deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env var)
            api_version: Azure API version (defaults to AZURE_OPENAI_API_VERSION env var or "2024-08-01-preview")
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize database connection
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Auto-detect Azure OpenAI if not specified
        if use_azure is None:
            use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_KEY"))
        
        # Initialize LLM based on provider
        if use_azure:
            # Azure OpenAI configuration
            azure_kwargs = {
                "azure_deployment": azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model_name,
                "api_version": api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            }
            
            # Only set temperature if it's not 0 (some models don't support 0)
            if temperature != 0:
                azure_kwargs["temperature"] = temperature
            
            # API key
            if api_key:
                azure_kwargs["api_key"] = api_key
            elif os.environ.get("AZURE_OPENAI_API_KEY"):
                azure_kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
            
            # Endpoint
            if azure_endpoint:
                azure_kwargs["azure_endpoint"] = azure_endpoint
            elif os.environ.get("AZURE_OPENAI_ENDPOINT"):
                azure_kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            self.llm = AzureChatOpenAI(**azure_kwargs)
        else:
            # Standard OpenAI configuration
            llm_kwargs = {
                "model": model_name,
            }
            
            # Only set temperature if it's not 0 (some models don't support 0)
            if temperature != 0:
                llm_kwargs["temperature"] = temperature
            
            if api_key:
                llm_kwargs["api_key"] = api_key
            elif os.environ.get("OPENAI_API_KEY"):
                llm_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
            
            if base_url:
                llm_kwargs["base_url"] = base_url
            elif os.environ.get("OPENAI_BASE_URL"):
                llm_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
            
            self.llm = ChatOpenAI(**llm_kwargs)
        
        # Detect if using reasoning models that don't support certain parameters
        reasoning_models = ["o1-preview", "o1-mini", "o1", "gpt-5"]
        model_to_check = azure_deployment if use_azure and azure_deployment else model_name
        self.is_reasoning_model = any(reasoning_model in model_to_check.lower() for reasoning_model in reasoning_models)
        
        # Only wrap LLM for models that need parameter filtering
        # Azure models (especially newer ones) may not support 'stop' parameter
        # But wrapping can interfere with tool binding, so be selective
        if self.is_reasoning_model:
            self.llm = ReasoningModelWrapper(self.llm)
        
        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create system message/prompt
        system_prompt = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {self.db.dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""
        
        # Create LangGraph agent
        self.agent_executor = create_react_agent(
            self.llm, 
            self.toolkit.get_tools(), 
            prompt=system_prompt
        )
    
    def query(self, question: str, evidence: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            evidence: Additional context/evidence for the question
            verbose: If True, print agent reasoning steps
            
        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - answer: Query result (if executed)
                - intermediate_steps: Agent's reasoning steps
                - error: Error message if any
        """
        # Create SQL capture callback
        sql_callback = SQLCaptureCallback()
        
        try:
            # Combine question with evidence if provided
            full_question = question
            if evidence:
                full_question = f"{question}\n\nContext: {evidence}"
            
            # Run agent with callback - LangGraph uses stream
            # We need to collect all events and wait for completion
            all_messages = []
            final_response = None
            event_count = 0
            
            # Configure with recursion limit to allow multi-step reasoning
            config = {
                "callbacks": [sql_callback],
                "recursion_limit": 50  # Allow up to 50 reasoning steps
            }
            
            for event in self.agent_executor.stream(
                {"messages": [("user", full_question)]},
                stream_mode="values",
                config=config
            ):
                event_count += 1
                all_messages = event.get("messages", [])
                # Keep updating with latest state
                final_response = event
                
                # Optional verbose output
                if verbose:
                    print(f"\n--- Event {event_count} ---")
                    if all_messages:
                        latest_message = all_messages[-1]
                        msg_type = getattr(latest_message, 'type', 'unknown')
                        content = getattr(latest_message, 'content', str(latest_message))
                        
                        # Check for tool calls
                        tool_calls = getattr(latest_message, 'tool_calls', [])
                        if tool_calls:
                            print(f"[{msg_type}] Tool calls: {len(tool_calls)}")
                            for tc in tool_calls:
                                print(f"  - {tc.get('name', 'unknown')}: {str(tc.get('args', {}))[:100]}")
                        else:
                            print(f"[{msg_type}] {content[:300]}...")
            
            if verbose:
                print(f"\nTotal events processed: {event_count}")
                print(f"Total messages: {len(all_messages)}")
                print(f"SQL queries captured: {len(sql_callback.sql_queries)}")
            
            # Extract answer from last message
            answer = None
            if all_messages:
                last_message = all_messages[-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                elif isinstance(last_message, dict):
                    answer = last_message.get('content')
                
                # For AIMessage, extract just the text content
                if hasattr(last_message, 'type') and last_message.type == 'ai':
                    answer = last_message.content
            
            # Extract SQL from callback first
            sql_query = None
            if sql_callback.sql_queries:
                sql_query = sql_callback.sql_queries[-1]
            
            # If callback didn't capture SQL, try extracting from messages
            if not sql_query:
                sql_query = self._extract_sql_from_messages(all_messages)
            
            return {
                "sql": sql_query,
                "answer": answer,
                "intermediate_steps": all_messages,
                "error": None
            }
        
        except Exception as e:
            error_msg = str(e)
            
            # Special handling for output parsing errors with reasoning models
            # These models output entire reasoning chain at once, including final answer
            if "output parsing error" in error_msg.lower() or "parsing llm output" in error_msg.lower():
                # Try to extract SQL and answer from the error message itself
                sql_query = self._extract_sql_from_error(error_msg)
                answer = self._extract_answer_from_error(error_msg)
                
                if sql_query or answer:
                    return {
                        "sql": sql_query,
                        "answer": answer,
                        "intermediate_steps": [],
                        "error": None  # We successfully recovered from the error
                    }
            
            return {
                "sql": None,
                "answer": None,
                "intermediate_steps": [],
                "error": error_msg
            }
    
    def _extract_sql_from_messages(self, messages: List[Any]) -> Optional[str]:
        """
        Extract SQL query from agent messages
        
        Args:
            messages: List of messages from agent execution
            
        Returns:
            SQL query string or None
        """
        import re
        
        # Look through messages for SQL queries
        for message in messages:
            content = None
            if hasattr(message, 'content'):
                content = message.content
            elif isinstance(message, dict):
                content = message.get('content', '')
            
            if content and isinstance(content, str):
                # Look for tool calls with SQL
                if 'sql_db_query' in content.lower():
                    # Try to extract SQL query
                    match = re.search(r'SELECT\s+.*?(?:;|\n|$)', content, re.IGNORECASE | re.DOTALL)
                    if match:
                        sql = match.group(0).strip()
                        if not sql.endswith(';'):
                            sql += ';'
                        return sql
        
        return None
    
    def _extract_sql_from_error(self, error_msg: str) -> Optional[str]:
        """
        Extract SQL query from parsing error message
        
        Args:
            error_msg: Error message string
            
        Returns:
            SQL query string or None
        """
        import re
        
        # Look for SQL query in the error message after "sql_db_query" or "Action Input:"
        # The pattern typically is: Action Input: SELECT ...
        patterns = [
            r'Action Input:\s*(SELECT\s+.*?)(?:\n(?:Observation|Thought|Action|Final Answer):|\Z)',
            r'sql_db_query.*?:\s*(SELECT\s+.*?)(?:\n(?:Observation|Thought|Action|Final Answer):|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                # Clean up the SQL - remove any trailing text after semicolon
                if ';' in sql:
                    sql = sql.split(';')[0] + ';'
                return sql
        
        return None
    
    def _extract_answer_from_error(self, error_msg: str) -> Optional[str]:
        """
        Extract final answer from parsing error message
        
        Args:
            error_msg: Error message string
            
        Returns:
            Answer string or None
        """
        import re
        
        # Look for "Final Answer:" in the error message
        pattern = r'Final Answer:\s*(.+?)(?:\nFor troubleshooting|\Z)'
        match = re.search(pattern, error_msg, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None


def create_debit_card_db(csv_dir: str, db_path: str) -> None:
    """
    Create SQLite database from debit card CSV files
    
    Args:
        csv_dir: Directory containing CSV files
        db_path: Output database path
    """
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    # Map CSV files to table names
    csv_files = {
        "customers": "customers.csv",
        "gasstations": "gasstations.csv",
        "products": "products.csv",
        "transactions_1k": "transactions_1k.csv",
        "yearmonth": "yearmonth.csv"
    }
    
    for table_name, csv_file in csv_files.items():
        csv_path = os.path.join(csv_dir, csv_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Loaded {table_name} from {csv_file}")
        else:
            print(f"Warning: {csv_file} not found")
    
    conn.close()
    print(f"Database created at {db_path}")


def main():
    """Example usage"""
    # Configuration
    csv_dir = "debit_card_csv_export"
    db_path = "debit_card.db"
    
    # Create database if it doesn't exist
    if not os.path.exists(db_path):
        print("Creating database from CSV files...")
        create_debit_card_db(csv_dir, db_path)
    
    # Initialize agent
    print("\nInitializing LangChain DB Agent...")
    agent = LangChainDBAgent(db_path=db_path)
    
    # Test query
    question = "What is the ratio of customers who pay in EUR against customers who pay in CZK?"
    print(f"\nQuestion: {question}")
    
    result = agent.query(question, verbose=False)
    
    print(f"\n{'='*80}")
    print(f"Generated SQL:\n{result['sql']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    if result['error']:
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    main()
