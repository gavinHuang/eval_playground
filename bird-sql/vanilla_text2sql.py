"""
Vanilla Text-to-SQL Module

A simple text-to-SQL implementation that uses direct LLM prompting with:
1. Basic prompt template
2. Schema information (tables and columns)
3. Sample data (3 rows per table)
"""

import os
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class VanillaText2SQL:
    """Vanilla Text-to-SQL using direct LLM prompting"""
    
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
        Initialize the Vanilla Text-to-SQL module
        
        Args:
            db_path: Path to SQLite database
            model_name: OpenAI model name (or Azure deployment name if use_azure=True)
            temperature: Model temperature (0 for deterministic)
            api_key: OpenAI/Azure API key (defaults to env var)
            base_url: OpenAI API base URL (optional, for OpenAI only)
            use_azure: Whether to use Azure OpenAI (auto-detected from env vars if not specified)
            azure_endpoint: Azure OpenAI endpoint (defaults to env var)
            azure_deployment: Azure deployment name (defaults to env var)
            api_version: Azure API version (defaults to env var)
        """
        self.db_path = db_path
        self.model_name = model_name
        
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
            
            if temperature != 0:
                azure_kwargs["temperature"] = temperature
            
            if api_key:
                azure_kwargs["api_key"] = api_key
            elif os.environ.get("AZURE_OPENAI_API_KEY"):
                azure_kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
            
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
        
        # Get database schema and sample data
        self.schema_info = self._get_schema_info()
        self.sample_data = self._get_sample_data()
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _get_schema_info(self) -> str:
        """Extract schema information from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        schema_lines = ["DATABASE SCHEMA:"]
        schema_lines.append("=" * 80)
        
        for (table_name,) in tables:
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            schema_lines.append(f"\nTable: {table_name}")
            schema_lines.append("-" * 40)
            schema_lines.append("Columns:")
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                pk_marker = " (PRIMARY KEY)" if is_pk else ""
                null_marker = " NOT NULL" if not_null else ""
                schema_lines.append(f"  - {col_name}: {col_type}{pk_marker}{null_marker}")
        
        conn.close()
        
        return "\n".join(schema_lines)
    
    def _get_sample_data(self, num_rows: int = 3) -> str:
        """Extract sample data from each table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        sample_lines = ["\nSAMPLE DATA (first 3 rows from each table):"]
        sample_lines.append("=" * 80)
        
        for (table_name,) in tables:
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            
            # Get sample rows
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {num_rows};")
            rows = cursor.fetchall()
            
            sample_lines.append(f"\nTable: {table_name}")
            sample_lines.append("-" * 40)
            
            if rows:
                # Format as simple table
                sample_lines.append("  " + " | ".join(col_names))
                sample_lines.append("  " + "-" * (len(" | ".join(col_names))))
                for row in rows:
                    sample_lines.append("  " + " | ".join(str(val) for val in row))
            else:
                sample_lines.append("  (No data)")
        
        conn.close()
        
        return "\n".join(sample_lines)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for text-to-SQL"""
        template = """You are a SQL expert. Given a database schema, sample data, and a user question, generate a valid SQLite query to answer the question.

{schema_info}

{sample_data}

INSTRUCTIONS:
1. Generate ONLY the SQL query, nothing else
2. Do not include any explanations or markdown formatting
3. The query should be valid SQLite syntax
4. Use proper JOIN conditions when combining tables
5. Pay attention to the column names and data types
6. If additional context is provided, use it to better understand the question
7. Limit results to at most 5 rows unless a specific number is requested
8. Only query for relevant columns, not SELECT *

{context}

USER QUESTION: {question}

SQL QUERY:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def query(self, question: str, evidence: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            evidence: Additional context/evidence for the question
            
        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - answer: None (not executed in vanilla version)
                - error: Error message if any
        """
        try:
            # Prepare context
            context = ""
            if evidence:
                context = f"ADDITIONAL CONTEXT: {evidence}"
            
            # Generate SQL
            sql_query = self.chain.invoke({
                "schema_info": self.schema_info,
                "sample_data": self.sample_data,
                "question": question,
                "context": context
            })
            
            # Clean up the generated SQL
            sql_query = sql_query.strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith("```"):
                lines = sql_query.split("\n")
                sql_query = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                sql_query = sql_query.strip()
            
            # Remove any "sql" language identifier
            if sql_query.lower().startswith("sql"):
                sql_query = sql_query[3:].strip()
            
            # Ensure it ends with semicolon
            if not sql_query.endswith(";"):
                sql_query += ";"
            
            # Try to execute the query to validate it
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                conn.close()
                
                # Format the result as answer
                answer = str(result) if result else "No results"
                
            except sqlite3.Error as e:
                conn.close()
                return {
                    "sql": sql_query,
                    "answer": None,
                    "error": f"SQL execution error: {str(e)}"
                }
            
            return {
                "sql": sql_query,
                "answer": answer,
                "error": None
            }
        
        except Exception as e:
            return {
                "sql": None,
                "answer": None,
                "error": str(e)
            }


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
    
    # Initialize vanilla text2sql
    print("\nInitializing Vanilla Text2SQL...")
    text2sql = VanillaText2SQL(db_path=db_path)
    
    # Test query
    question = "What is the ratio of customers who pay in EUR against customers who pay in CZK?"
    print(f"\nQuestion: {question}")
    
    result = text2sql.query(question)
    
    print(f"\nGenerated SQL: {result['sql']}")
    print(f"Answer: {result['answer']}")
    
    if result['error']:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
