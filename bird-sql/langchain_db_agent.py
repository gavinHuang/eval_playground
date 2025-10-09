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
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.language_models import BaseChatModel
from typing import List, Any
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class ReasoningModelWrapper(BaseChatModel):
    """Wrapper for reasoning models that don't support certain parameters"""
    
    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self._llm = llm
    
    def _generate(self, messages: List[BaseMessage], stop: Any = None, **kwargs) -> ChatResult:
        """Override to remove unsupported parameters"""
        # Remove 'stop' parameter for reasoning models
        kwargs.pop('stop', None)
        return self._llm._generate(messages, **kwargs)
    
    async def _agenerate(self, messages: List[BaseMessage], stop: Any = None, **kwargs) -> ChatResult:
        """Async override to remove unsupported parameters"""
        kwargs.pop('stop', None)
        return await self._llm._agenerate(messages, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return self._llm._llm_type
    
    def __getattr__(self, name: str):
        """Delegate all other attributes to wrapped LLM"""
        return getattr(self._llm, name)


class LangChainDBAgent:
    """LangChain-based database agent for text-to-SQL"""
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "gpt-5",
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
        
        # Wrap LLM for reasoning models to filter out unsupported parameters
        if self.is_reasoning_model:
            self.llm = ReasoningModelWrapper(self.llm)
        
        # Create SQL agent
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def query(self, question: str, evidence: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            evidence: Additional context/evidence for the question
            
        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - answer: Query result (if executed)
                - intermediate_steps: Agent's reasoning steps
                - error: Error message if any
        """
        try:
            # Combine question with evidence if provided
            full_question = question
            if evidence:
                full_question = f"{question}\n\nContext: {evidence}"
            
            # Run agent
            result = self.agent_executor.invoke({"input": full_question})
            
            # Extract SQL from intermediate steps
            sql_query = self._extract_sql_from_result(result)
            
            return {
                "sql": sql_query,
                "answer": result.get("output"),
                "intermediate_steps": result.get("intermediate_steps", []),
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
    
    def _extract_sql_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract SQL query from agent result
        
        Args:
            result: Agent execution result
            
        Returns:
            SQL query string or None
        """
        # Try to extract from intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action = step[0]
                # Check if this is a SQL query action
                if hasattr(action, 'tool') and action.tool == 'sql_db_query':
                    if hasattr(action, 'tool_input'):
                        return action.tool_input
        
        # Fallback: try to extract from output
        output = result.get("output", "")
        if "SELECT" in output.upper():
            # Simple extraction - find SELECT statement
            lines = output.split('\n')
            for line in lines:
                if line.strip().upper().startswith("SELECT"):
                    return line.strip()
        
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
    question = "How many gas stations in CZE has Premium gas?"
    print(f"\nQuestion: {question}")
    
    result = agent.query(question)
    
    print(f"\nGenerated SQL: {result['sql']}")
    print(f"Answer: {result['answer']}")
    
    if result['error']:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
