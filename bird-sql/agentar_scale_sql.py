"""
Agentar-Scale-SQL Implementation

A state-of-the-art text-to-SQL solution implementing the Agentar-Scale-SQL framework.
This solution uses:
1. Task Understanding: Keyword extraction, cell retrieval, skeleton extraction, and few-shot example retrieval
2. SQL Generation Scaling: Parallel scaling (diverse synthesis) and sequential scaling (iterative refinement)
3. SQL Selection Scaling: Candidate consolidation and tournament selection

Reference: agentar-scale.md pseudocode
"""

import os
import sqlite3
import hashlib
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss: pip install faiss-cpu")

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class SQLCandidate:
    """SQL candidate with metadata"""
    sql: str
    source: str  # 'reasoning' or 'icl'
    temperature: float
    execution_result: Optional[Any] = None
    execution_error: Optional[str] = None
    score: int = 0


class AgentarScaleSQL:
    """
    Agentar-Scale-SQL implementation using a unified LLM for all tasks
    """
    
    def __init__(
        self,
        db_path: str,
        model_name: str = "gpt-4.1",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_azure: Optional[bool] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        n_reasoning_candidates: int = 4,
        n_icl_candidates: int = 5
    ):
        """
        Initialize Agentar-Scale-SQL
        
        Args:
            db_path: Path to SQLite database
            model_name: OpenAI model name (or Azure deployment name if use_azure=True)
            temperature: Default temperature for LLM
            api_key: OpenAI/Azure API key
            base_url: OpenAI API base URL
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name
            api_version: Azure API version
            n_reasoning_candidates: Number of reasoning-based candidates
            n_icl_candidates: Number of ICL-based candidates
        """
        self.db_path = db_path
        self.model_name = model_name
        self.n_reasoning_candidates = n_reasoning_candidates
        self.n_icl_candidates = n_icl_candidates
        
        # Auto-detect Azure OpenAI
        if use_azure is None:
            use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_KEY"))
        
        # Create LLM factory function
        self.use_azure = use_azure
        self.llm_config = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "api_version": api_version,
            "temperature": temperature
        }
        
        # Initialize unified LLM
        self.llm = self._create_llm(temperature)
        
        # Initialize embeddings model
        self.embeddings = self._create_embeddings()
        
        # Get database schema
        self.schema_ddl = self._get_schema_ddl()
        self.schema_light = self._get_schema_light()
        
        # Initialize vector stores
        self.cell_store, self.cell_index, self.cell_texts = self._build_cell_store()
        self.example_store = self._build_example_store()
    
    def _create_llm(self, temperature: float) -> Any:
        """Create LLM instance with specified temperature"""
        if self.use_azure:
            azure_kwargs = {
                "azure_deployment": self.llm_config["azure_deployment"] or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or self.llm_config["model_name"],
                "api_version": self.llm_config["api_version"] or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            }
            
            if temperature != 0:
                azure_kwargs["temperature"] = temperature
            
            if self.llm_config["api_key"]:
                azure_kwargs["api_key"] = self.llm_config["api_key"]
            elif os.environ.get("AZURE_OPENAI_API_KEY"):
                azure_kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
            
            if self.llm_config["azure_endpoint"]:
                azure_kwargs["azure_endpoint"] = self.llm_config["azure_endpoint"]
            elif os.environ.get("AZURE_OPENAI_ENDPOINT"):
                azure_kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            return AzureChatOpenAI(**azure_kwargs)
        else:
            llm_kwargs = {
                "model": self.llm_config["model_name"],
            }
            
            if temperature != 0:
                llm_kwargs["temperature"] = temperature
            
            if self.llm_config["api_key"]:
                llm_kwargs["api_key"] = self.llm_config["api_key"]
            elif os.environ.get("OPENAI_API_KEY"):
                llm_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
            
            if self.llm_config["base_url"]:
                llm_kwargs["base_url"] = self.llm_config["base_url"]
            elif os.environ.get("OPENAI_BASE_URL"):
                llm_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
            
            return ChatOpenAI(**llm_kwargs)
    
    def _create_embeddings(self) -> Any:
        """Create embeddings model"""
        if self.use_azure:
            azure_kwargs = {
                "model": os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
                "api_version": self.llm_config["api_version"] or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            }
            
            if self.llm_config["api_key"]:
                azure_kwargs["api_key"] = self.llm_config["api_key"]
            elif os.environ.get("AZURE_OPENAI_API_KEY"):
                azure_kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
            
            if self.llm_config["azure_endpoint"]:
                azure_kwargs["azure_endpoint"] = self.llm_config["azure_endpoint"]
            elif os.environ.get("AZURE_OPENAI_ENDPOINT"):
                azure_kwargs["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            return AzureOpenAIEmbeddings(**azure_kwargs)
        else:
            embeddings_kwargs = {
                "model": "text-embedding-3-small",
            }
            
            if self.llm_config["api_key"]:
                embeddings_kwargs["api_key"] = self.llm_config["api_key"]
            elif os.environ.get("OPENAI_API_KEY"):
                embeddings_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
            
            if self.llm_config["base_url"]:
                embeddings_kwargs["base_url"] = self.llm_config["base_url"]
            elif os.environ.get("OPENAI_BASE_URL"):
                embeddings_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
            
            return OpenAIEmbeddings(**embeddings_kwargs)
    
    def _get_schema_ddl(self) -> str:
        """Extract database schema in DDL format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        ddl_lines = ["-- DATABASE SCHEMA (DDL FORMAT)"]
        for table_name, create_sql in tables:
            if create_sql:
                ddl_lines.append(f"\n{create_sql};")
        
        conn.close()
        return "\n".join(ddl_lines)
    
    def _get_schema_light(self) -> str:
        """Extract database schema in lightweight markdown format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        light_lines = ["# Database Schema (Light Format)\n"]
        
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            light_lines.append(f"## Table: `{table_name}`")
            light_lines.append("| Column | Type | Key |")
            light_lines.append("|--------|------|-----|")
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                key_marker = "PK" if is_pk else ""
                light_lines.append(f"| {col_name} | {col_type} | {key_marker} |")
            
            light_lines.append("")
        
        conn.close()
        return "\n".join(light_lines)
    
    def _build_cell_store(self) -> Tuple[Dict[str, List[str]], Any, List[str]]:
        """Build FAISS vector store for database cell values"""
        print("Building FAISS cell store...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cell_store = defaultdict(list)
        cell_texts = []  # Text representations for FAISS
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000;")
            rows = cursor.fetchall()
            
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            
            for row in rows:
                for col_name, value in zip(col_names, row):
                    if value is not None:
                        key = f"{table_name}.{col_name}"
                        cell_store[key].append(str(value))
        
        # Create text representations for each table.column with sample values
        for key, values in cell_store.items():
            unique_values = list(set(values))[:10]  # Get up to 10 unique values
            text = f"{key}: {', '.join(unique_values)}"
            cell_texts.append(text)
        
        conn.close()
        
        # Build FAISS index
        if len(cell_texts) > 0:
            print(f"Creating embeddings for {len(cell_texts)} cell entries...")
            embeddings_vectors = self.embeddings.embed_documents(cell_texts)
            embeddings_array = np.array(embeddings_vectors, dtype='float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            print(f"FAISS index created with {index.ntotal} vectors")
        else:
            # Empty index
            index = faiss.IndexFlatL2(1536)  # Default dimension for text-embedding-3-small
            print("Warning: No cell data found, created empty index")
        
        return dict(cell_store), index, cell_texts
    
    def _build_example_store(self) -> List[Dict[str, str]]:
        """Build few-shot example store"""
        # Simplified: return generic examples
        return [
            {
                "skeleton": "SELECT column FROM table WHERE condition",
                "question": "Find records matching a condition",
                "sql": "SELECT * FROM table WHERE column = value;"
            },
            {
                "skeleton": "SELECT COUNT(*) FROM table",
                "question": "Count records",
                "sql": "SELECT COUNT(*) FROM table;"
            },
            {
                "skeleton": "SELECT column1, column2 FROM table1 JOIN table2",
                "question": "Join tables",
                "sql": "SELECT t1.col1, t2.col2 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id;"
            }
        ]
    
    def _extract_keywords(self, question: str, evidence: str) -> List[str]:
        """
        Step 1.1: Extract keywords from question and evidence using LLM
        """
        prompt = f"""Extract the most important keywords and phrases from this database query question and evidence.
Focus on: table names, column names, values, conditions, and key concepts.

Question: {question}
Evidence: {evidence}

Return ONLY a comma-separated list of keywords, no explanations."""
        
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        keywords_str = chain.invoke({"question": question, "evidence": evidence})
        
        # Parse keywords
        keywords = [k.strip() for k in keywords_str.split(',')]
        return keywords
    
    def _retrieve_relevant_cells(self, keywords: List[str], top_k: int = 20) -> str:
        """
        Step 1.2: Retrieve relevant database cell values using FAISS vector search
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of top results to return
            
        Returns:
            Formatted string of relevant cell values
        """
        if len(self.cell_texts) == 0:
            return "No specific cell values found."
        
        # Create query from keywords
        query_text = " ".join(keywords)
        
        # Get embedding for query
        query_embedding = self.embeddings.embed_query(query_text)
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Search FAISS index
        k = min(top_k, len(self.cell_texts))
        distances, indices = self.cell_index.search(query_vector, k)
        
        # Collect relevant cells
        relevant_cells = []
        seen = set()
        
        for idx in indices[0]:
            if idx < len(self.cell_texts):
                cell_text = self.cell_texts[idx]
                # Avoid duplicates
                if cell_text not in seen:
                    relevant_cells.append(cell_text)
                    seen.add(cell_text)
        
        if not relevant_cells:
            return "No specific cell values found."
        
        return "\n".join(relevant_cells)
    
    def _extract_skeleton(self, question: str) -> str:
        """
        Step 1.3: Extract SQL skeleton/structure from question using LLM
        """
        prompt = f"""Analyze this database question and extract the SQL query skeleton/structure.
Focus on the query pattern, not specific values.

Question: {question}

Return ONLY the SQL skeleton using generic placeholders like <table>, <column>, <condition>.
Example: SELECT <column> FROM <table> WHERE <condition>"""
        
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        skeleton = chain.invoke({"question": question})
        return skeleton.strip()
    
    def _retrieve_few_shot_examples(self, skeleton: str) -> List[Dict[str, str]]:
        """
        Step 1.4: Retrieve few-shot examples based on SQL skeleton
        """
        # Simplified: return examples that match skeleton pattern
        matching_examples = []
        
        skeleton_lower = skeleton.lower()
        for example in self.example_store:
            if any(pattern in skeleton_lower for pattern in ['select', 'join', 'count', 'where']):
                matching_examples.append(example)
        
        return matching_examples[:3]  # Return top 3
    
    def _task_understanding(
        self,
        question: str,
        evidence: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Step 1: Task Understanding
        Returns: (relevant_cells, few_shot_examples)
        """
        # Extract keywords
        keywords = self._extract_keywords(question, evidence)
        
        # Retrieve relevant cells
        relevant_cells = self._retrieve_relevant_cells(keywords)
        
        # Extract skeleton
        skeleton = self._extract_skeleton(question)
        
        # Retrieve few-shot examples
        few_shot_examples = self._retrieve_few_shot_examples(skeleton)
        
        return relevant_cells, few_shot_examples
    
    def _generate_reasoning_candidate(
        self,
        question: str,
        evidence: str,
        relevant_cells: str,
        temperature: float = 0.2
    ) -> str:
        """
        Generate SQL using reasoning approach (DDL schema + step-by-step reasoning)
        """
        prompt = f"""You are an expert SQL developer. Generate a SQL query using deep, step-by-step reasoning.

{self.schema_ddl}

Relevant database values:
{relevant_cells}

Question: {question}
Context: {evidence}

Think step-by-step:
1. Identify which tables are needed
2. Determine the columns to select
3. Identify join conditions if multiple tables
4. Determine WHERE conditions
5. Consider aggregations or grouping if needed
6. Consider ordering or limiting results

Generate ONLY the SQL query (no explanations or markdown):"""
        
        llm = self._create_llm(temperature)
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        sql = chain.invoke({
            "question": question,
            "evidence": evidence,
            "schema": self.schema_ddl,
            "cells": relevant_cells
        })
        
        return self._clean_sql(sql)
    
    def _generate_icl_candidate(
        self,
        question: str,
        evidence: str,
        relevant_cells: str,
        few_shot_examples: List[Dict[str, str]],
        style: str = "direct",
        temperature: float = 0.5
    ) -> str:
        """
        Generate SQL using ICL approach (Light schema + few-shot examples)
        """
        # Format few-shot examples
        examples_text = "\n\n".join([
            f"Example {i+1}:\nQuestion: {ex['question']}\nSQL: {ex['sql']}"
            for i, ex in enumerate(few_shot_examples)
        ])
        
        if style == "cot":
            reasoning_instruction = "\nThink step-by-step before generating the query."
        elif style == "decomposition":
            reasoning_instruction = "\nDecompose the question into sub-questions, then generate the query."
        else:  # direct
            reasoning_instruction = ""
        
        prompt = f"""Generate a SQL query to answer the question.

{self.schema_light}

Relevant database values:
{relevant_cells}

Here are some example queries:
{examples_text}

Question: {question}
Context: {evidence}{reasoning_instruction}

Generate ONLY the SQL query (no explanations or markdown):"""
        
        llm = self._create_llm(temperature)
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        sql = chain.invoke({
            "question": question,
            "evidence": evidence,
            "schema": self.schema_light,
            "cells": relevant_cells,
            "examples": examples_text
        })
        
        return self._clean_sql(sql)
    
    def _clean_sql(self, sql: str) -> str:
        """Clean generated SQL"""
        sql = sql.strip()
        
        # Remove markdown code blocks
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            sql = sql.strip()
        
        # Remove sql language identifier
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
        
        # Ensure semicolon
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _fix_syntax_error(self, sql: str, error: str) -> str:
        """
        Step 2b: Fix syntax errors in SQL
        """
        prompt = f"""The following SQL query has a syntax error. Fix it.

SQL Query:
{sql}

Error:
{error}

Return ONLY the corrected SQL query (no explanations or markdown):"""
        
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        fixed_sql = chain.invoke({"sql": sql, "error": error})
        
        return self._clean_sql(fixed_sql)
    
    def _revise_semantics(self, sql: str, question: str, evidence: str, relevant_cells: str) -> str:
        """
        Step 2c: Revise SQL for semantic issues
        """
        prompt = f"""The following SQL query executed but may have logical/semantic issues. Revise it to better answer the question.

SQL Query:
{sql}

Question: {question}
Context: {evidence}

Relevant database values:
{relevant_cells}

Return ONLY the revised SQL query (no explanations or markdown):"""
        
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        revised_sql = chain.invoke({
            "sql": sql,
            "question": question,
            "evidence": evidence,
            "cells": relevant_cells
        })
        
        return self._clean_sql(revised_sql)
    
    def _execute_sql(self, sql: str) -> Tuple[Optional[Any], Optional[str]]:
        """Execute SQL and return result or error"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return result, None
        except Exception as e:
            return None, str(e)
    
    def _sql_generation_scaling(
        self,
        question: str,
        evidence: str,
        relevant_cells: str,
        few_shot_examples: List[Dict[str, str]]
    ) -> List[SQLCandidate]:
        """
        Step 2: SQL Generation Scaling
        Combines Parallel Scaling (Diverse Synthesis) and Sequential Scaling (Iterative Refinement)
        """
        candidates = []
        
        # Parallel Scaling: Reasoning Candidates
        print(f"  Generating {self.n_reasoning_candidates} reasoning-based candidates...")
        for i in range(self.n_reasoning_candidates):
            temp = 0.2 + (i * 0.1)  # Vary temperature slightly
            sql = self._generate_reasoning_candidate(question, evidence, relevant_cells, temp)
            candidates.append(SQLCandidate(
                sql=sql,
                source="reasoning",
                temperature=temp
            ))
        
        # Parallel Scaling: ICL Candidates
        print(f"  Generating {self.n_icl_candidates} ICL-based candidates...")
        styles = ["direct", "cot", "decomposition"]
        for i in range(self.n_icl_candidates):
            style = styles[i % len(styles)]
            temp = 0.3 + (i * 0.15)
            sql = self._generate_icl_candidate(
                question, evidence, relevant_cells, few_shot_examples, style, temp
            )
            candidates.append(SQLCandidate(
                sql=sql,
                source="icl",
                temperature=temp
            ))
        
        # Sequential Scaling: Iterative Refinement
        print(f"  Refining {len(candidates)} candidates...")
        refined_candidates = []
        
        for candidate in candidates:
            # Execute SQL
            result, error = self._execute_sql(candidate.sql)
            candidate.execution_result = result
            candidate.execution_error = error
            
            # Syntax repair
            if error and ("syntax" in error.lower() or "near" in error.lower()):
                fixed_sql = self._fix_syntax_error(candidate.sql, error)
                result, error = self._execute_sql(fixed_sql)
                candidate.sql = fixed_sql
                candidate.execution_result = result
                candidate.execution_error = error
            
            # Semantic revision (optional - can be expensive)
            # Only revise if execution succeeded but result seems suspicious
            # if error is None and result is not None and len(result) == 0:
            #     revised_sql = self._revise_semantics(candidate.sql, question, evidence, relevant_cells)
            #     result, error = self._execute_sql(revised_sql)
            #     candidate.sql = revised_sql
            #     candidate.execution_result = result
            #     candidate.execution_error = error
            
            refined_candidates.append(candidate)
        
        return refined_candidates
    
    def _consolidate_candidates(self, candidates: List[SQLCandidate]) -> List[SQLCandidate]:
        """
        Step 3a: Consolidate candidates by grouping by execution results
        """
        # Group by result hash
        result_groups = defaultdict(list)
        
        for candidate in candidates:
            if candidate.execution_result is not None:
                # Create hash of result
                result_str = str(sorted(candidate.execution_result))
                result_hash = hashlib.md5(result_str.encode()).hexdigest()
                result_groups[result_hash].append(candidate)
            else:
                # Failed queries get individual groups
                result_groups[id(candidate)].append(candidate)
        
        # Select one representative from each group
        representatives = []
        for group in result_groups.values():
            # Prefer reasoning candidates, then shortest SQL
            group_sorted = sorted(
                group,
                key=lambda c: (
                    c.source != "reasoning",
                    len(c.sql)
                )
            )
            representatives.append(group_sorted[0])
        
        print(f"  Consolidated {len(candidates)} candidates to {len(representatives)} unique results")
        return representatives
    
    def _tournament_selection(
        self,
        candidates: List[SQLCandidate],
        question: str,
        evidence: str
    ) -> SQLCandidate:
        """
        Step 3b: Tournament Selection using pairwise comparison
        """
        if len(candidates) == 0:
            raise ValueError("No candidates to select from")
        
        if len(candidates) == 1:
            return candidates[0]
        
        print(f"  Running tournament selection on {len(candidates)} candidates...")
        
        # Initialize scores
        for candidate in candidates:
            candidate.score = 0
        
        # Pairwise comparison (simplified: compare top candidates)
        # Full O(n^2) comparison can be expensive
        top_k = min(5, len(candidates))
        top_candidates = candidates[:top_k]
        
        for i in range(len(top_candidates)):
            for j in range(i + 1, len(top_candidates)):
                c_i = top_candidates[i]
                c_j = top_candidates[j]
                
                # Compare using LLM
                winner = self._compare_candidates(c_i, c_j, question, evidence)
                
                if winner == "first":
                    c_i.score += 1
                elif winner == "second":
                    c_j.score += 1
        
        # Select candidate with highest score
        best_candidate = max(candidates, key=lambda c: (
            c.score,
            c.execution_error is None,  # Prefer no errors
            c.source == "reasoning",  # Prefer reasoning
            -len(c.sql)  # Prefer shorter SQL
        ))
        
        return best_candidate
    
    def _compare_candidates(
        self,
        c1: SQLCandidate,
        c2: SQLCandidate,
        question: str,
        evidence: str
    ) -> str:
        """
        Compare two SQL candidates using LLM as reasoning selector
        Returns: "first", "second", or "tie"
        """
        prompt = f"""You are a SQL expert evaluating two SQL queries that attempt to answer the same question.

Question: {question}
Context: {evidence}

SQL Query 1:
{c1.sql}
Result: {c1.execution_result if c1.execution_error is None else f"Error: {c1.execution_error}"}

SQL Query 2:
{c2.sql}
Result: {c2.execution_result if c2.execution_error is None else f"Error: {c2.execution_error}"}

Which query better answers the question? Consider:
1. Correctness of logic
2. Execution success
3. Query efficiency
4. Result relevance

Respond with ONLY: "first", "second", or "tie"."""
        
        chain = ChatPromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "evidence": evidence,
            "sql1": c1.sql,
            "sql2": c2.sql
        })
        
        response_lower = response.strip().lower()
        if "first" in response_lower:
            return "first"
        elif "second" in response_lower:
            return "second"
        else:
            return "tie"
    
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
                - error: Error message if any
                - candidates: All generated candidates (for debugging)
        """
        try:
            evidence = evidence or ""
            
            print(f"[Agentar-Scale-SQL] Processing question: {question[:100]}...")
            
            # Step 1: Task Understanding
            print("[Step 1] Task Understanding...")
            relevant_cells, few_shot_examples = self._task_understanding(question, evidence)
            
            # Step 2: SQL Generation Scaling
            print("[Step 2] SQL Generation Scaling...")
            candidates = self._sql_generation_scaling(
                question, evidence, relevant_cells, few_shot_examples
            )
            
            # Step 3: SQL Selection Scaling
            print("[Step 3] SQL Selection Scaling...")
            
            # Consolidate candidates
            consolidated = self._consolidate_candidates(candidates)
            
            # Tournament selection
            best_candidate = self._tournament_selection(consolidated, question, evidence)
            
            print(f"[Complete] Selected best candidate from {len(candidates)} total candidates")
            
            return {
                "sql": best_candidate.sql,
                "answer": str(best_candidate.execution_result) if best_candidate.execution_result else None,
                "error": best_candidate.execution_error,
                "candidates": [
                    {
                        "sql": c.sql,
                        "source": c.source,
                        "temperature": c.temperature,
                        "execution_error": c.execution_error,
                        "score": c.score
                    }
                    for c in candidates
                ]
            }
        
        except Exception as e:
            return {
                "sql": None,
                "answer": None,
                "error": str(e),
                "candidates": []
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
    csv_dir = "debit_card_csv_export"
    db_path = "debit_card.db"
    
    # Create database if it doesn't exist
    if not os.path.exists(db_path):
        print("Creating database from CSV files...")
        create_debit_card_db(csv_dir, db_path)
    
    # Initialize Agentar-Scale-SQL
    print("\nInitializing Agentar-Scale-SQL...")
    agentar = AgentarScaleSQL(
        db_path=db_path,
        n_reasoning_candidates=2,  # Reduced for demo
        n_icl_candidates=3
    )
    
    # Test query
    question = "What is the ratio of customers who pay in EUR against customers who pay in CZK?"
    print(f"\nQuestion: {question}")
    
    result = agentar.query(question)
    
    print(f"\n{'='*80}")
    print(f"Final SQL:\n{result['sql']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    if result['error']:
        print(f"\nError: {result['error']}")
    
    # Print candidate summary
    if result.get('candidates'):
        print(f"\n{'='*80}")
        print(f"Generated {len(result['candidates'])} candidates:")
        for i, c in enumerate(result['candidates'], 1):
            status = "✓" if not c['execution_error'] else "✗"
            print(f"{i}. {status} [{c['source']}] Score: {c['score']}")


if __name__ == "__main__":
    main()
