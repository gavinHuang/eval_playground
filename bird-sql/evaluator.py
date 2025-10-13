"""
Evaluator module for comparing Text-to-SQL solutions

Evaluates Snowflake Cortex Analyst vs LangChain DB Agent on BIRD-SQL benchmark.
"""

import json
import os
import sqlite3
import csv
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

from sql_normalizer import SQLNormalizer
from snowflake_cortex_analyst import CortexAnalystWrapper
from langchain_db_agent import LangChainDBAgent
from vanilla_text2sql import VanillaText2SQL


@dataclass
class EvaluationResult:
    """Result for a single evaluation"""
    question_id: int
    question: str
    evidence: str
    expected_sql: str
    generated_sql: Optional[str]
    exact_match: bool
    result_match: bool
    expected_result: Optional[Any]
    generated_result: Optional[Any]
    error: Optional[str]
    execution_time: float
    difficulty: str
    sql_execution_error: Optional[str] = None


@dataclass
class SolutionMetrics:
    """Metrics for a solution"""
    solution_name: str
    total_questions: int
    exact_matches: int
    result_matches: int
    errors: int
    precision: float
    result_accuracy: float
    avg_execution_time: float
    error_distribution: Dict[str, int]
    difficulty_breakdown: Dict[str, Dict[str, int]]


class TextToSQLEvaluator:
    """Evaluator for Text-to-SQL solutions"""
    
    def __init__(
        self,
        dev_json_path: str,
        db_id: str = "debit_card_specializing",
        sql_dialect: str = "sqlite",
        db_path: Optional[str] = None,
        snowflake_warehouse: Optional[str] = None,
        snowflake_database: Optional[str] = None,
        snowflake_schema: Optional[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            dev_json_path: Path to dev.json file
            db_id: Database ID to filter questions
            sql_dialect: SQL dialect for normalization
            db_path: Path to SQLite database for executing queries (optional)
            snowflake_warehouse: Snowflake warehouse for executing queries via REST API (optional)
            snowflake_database: Snowflake database for executing queries via REST API (optional)
            snowflake_schema: Snowflake schema for executing queries via REST API (optional)
        """
        self.dev_json_path = dev_json_path
        self.db_id = db_id
        self.normalizer = SQLNormalizer(dialect=sql_dialect)
        self.db_path = db_path
        self.snowflake_warehouse = snowflake_warehouse
        self.snowflake_database = snowflake_database
        self.snowflake_schema = snowflake_schema
        
        # Load questions
        self.questions = self._load_questions()
        print(f"Loaded {len(self.questions)} questions for {db_id}")
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from dev.json filtered by db_id"""
        with open(self.dev_json_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        
        # Filter by db_id
        filtered = [q for q in all_questions if q.get('db_id') == self.db_id]
        return filtered
    
    def _execute_sql(self, sql: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
        """
        Execute SQL query and return results
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Tuple of (results, error). Results is a list of tuples or None if error.
        """
        if not self.db_path:
            return None, "No database path configured"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return results, None
        except Exception as e:
            return None, str(e)
    
    def _normalize_value(self, value: Any) -> Any:
        """
        Normalize a value for comparison, treating numeric strings as numbers.
        
        Args:
            value: Value to normalize
            
        Returns:
            Normalized value
        """
        if value is None:
            return None
        
        # If it's a string, try to convert to number
        if isinstance(value, str):
            # Try integer first
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
            
            # Try float
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
            
            # Return as-is if not a number
            return value
        
        # If it's already a number, return as-is
        return value
    
    def _normalize_row(self, row: Tuple) -> Tuple:
        """
        Normalize all values in a row.
        
        Args:
            row: Row tuple to normalize
            
        Returns:
            Normalized row tuple
        """
        return tuple(self._normalize_value(val) for val in row)
    
    def _compare_results(
        self, 
        expected_result: Optional[List[Tuple]], 
        generated_result: Optional[List[Tuple]]
    ) -> bool:
        """
        Compare two query results for equality.
        Generated result can have more columns than expected, as long as it contains
        all the expected columns with correct values.
        Values are normalized so that "123" equals 123.
        
        Args:
            expected_result: Expected query result (reference answer)
            generated_result: Generated query result (may have extra columns)
            
        Returns:
            True if generated result contains all expected data, False otherwise
        """
        if expected_result is None or generated_result is None:
            return False
        
        # If both are empty, they match
        if len(expected_result) == 0 and len(generated_result) == 0:
            return True
        
        # If row counts don't match, results are different
        if len(expected_result) != len(generated_result):
            return False
        
        # Get column counts
        if len(expected_result) > 0:
            expected_col_count = len(expected_result[0])
        else:
            return True  # Empty results match
            
        if len(generated_result) > 0:
            generated_col_count = len(generated_result[0])
        else:
            return False  # Generated is empty but expected is not
        
        # Generated result must have at least as many columns as expected
        if generated_col_count < expected_col_count:
            return False
        
        # Normalize expected results
        normalized_expected = [self._normalize_row(row) for row in expected_result]
        
        try:
            # If generated has more columns, we need to check if expected columns are contained
            if generated_col_count > expected_col_count:
                # Try all possible column combinations to see if any match
                from itertools import combinations
                
                # Extract just the expected number of columns from generated result
                # Try all combinations of columns
                for col_indices in combinations(range(generated_col_count), expected_col_count):
                    # Extract selected columns from generated result
                    projected_result = [
                        tuple(row[i] for i in col_indices) 
                        for row in generated_result
                    ]
                    
                    # Normalize projected result
                    normalized_projected = [self._normalize_row(row) for row in projected_result]
                    
                    # Compare projected result with expected result
                    try:
                        set1 = set(normalized_expected)
                        set2 = set(normalized_projected)
                        if set1 == set2:
                            return True
                    except (TypeError, ValueError):
                        # If conversion to set fails, try sorted comparison
                        try:
                            sorted1 = sorted(normalized_expected)
                            sorted2 = sorted(normalized_projected)
                            if sorted1 == sorted2:
                                return True
                        except TypeError:
                            # If sorting fails, compare directly
                            if normalized_expected == normalized_projected:
                                return True
                
                # No column combination matched
                return False
            else:
                # Same number of columns - do exact comparison
                # Normalize generated results
                normalized_generated = [self._normalize_row(row) for row in generated_result]
                
                try:
                    set1 = set(normalized_expected)
                    set2 = set(normalized_generated)
                    return set1 == set2
                except (TypeError, ValueError):
                    # If conversion to set fails (unhashable types), compare as lists
                    try:
                        sorted1 = sorted(normalized_expected)
                        sorted2 = sorted(normalized_generated)
                        return sorted1 == sorted2
                    except TypeError:
                        # If sorting fails, just compare directly
                        return normalized_expected == normalized_generated
                        
        except Exception:
            # If any error occurs, fall back to exact comparison
            return expected_result == generated_result
    
    def evaluate_cortex_analyst(
        self,
        account_url: str,
        token: str,
        semantic_model_file: str,
        token_type: str = "OAUTH"
    ) -> tuple[List[EvaluationResult], SolutionMetrics]:
        """
        Evaluate Snowflake Cortex Analyst solution
        
        Args:
            account_url: Snowflake account URL
            token: Authorization token
            semantic_model_file: Path to semantic model file on stage
            token_type: Token type
            
        Returns:
            Tuple of (results list, metrics)
        """
        client = CortexAnalystWrapper(
            account_url=account_url,
            token=token,
            semantic_model_file=semantic_model_file,
            token_type=token_type,
            warehouse=self.snowflake_warehouse,
            database=self.snowflake_database,
            schema=self.snowflake_schema
        )
        
        results = []
        
        for question_data in self.questions:
            start_time = time.time()
            
            try:
                # Query the client
                response = client.query(
                    question=question_data['question'],
                    evidence=question_data.get('evidence', '')
                )
                
                # Extract SQL from message.content list
                generated_sql = None
                if 'message' in response and 'content' in response['message']:
                    for content_item in response['message']['content']:
                        if content_item.get('type') == 'sql':
                            generated_sql = content_item.get('statement')
                            break
                
                execution_time = time.time() - start_time
                
                # Compare with expected SQL (AST-based)
                # DISABLED: Skipping exact match for now - LLM hallucination issue
                exact_match = False
                # if generated_sql:
                #     exact_match = self.normalizer.compare_ast(
                #         generated_sql,
                #         question_data['SQL']
                #     )
                
                # Execute queries and compare results
                result_match = False
                expected_result = None
                generated_result = None
                sql_execution_error = None
                
                # Get expected result by executing expected SQL against SQLite
                if self.db_path:
                    expected_result, expected_error = self._execute_sql(question_data['SQL'])
                
                # Execute generated SQL against Snowflake
                if generated_sql:
                    generated_result, generated_error = client.execute_snowflake_sql(generated_sql)
                    if generated_error:
                        sql_execution_error = generated_error
                    # Compare results if we have both
                    if expected_result is not None and generated_result is not None:
                        result_match = self._compare_results(expected_result, generated_result)
                
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=generated_sql,
                    exact_match=exact_match,
                    result_match=result_match,
                    expected_result=expected_result,
                    generated_result=generated_result,
                    error=response['error'],
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=sql_execution_error
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"ERROR on question {question_data['question_id']}: {str(e)}")
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=None,
                    exact_match=False,
                    result_match=False,
                    expected_result=None,
                    generated_result=None,
                    error=str(e),
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=None
                )
            
            results.append(result)
            status_icon = "✓" if result.exact_match else ("≈" if result.result_match else "✗")
            print(f"Processed question {result.question_id}: {status_icon}")
            if not result.exact_match and result.generated_sql:
                print(f"  Expected SQL: {result.expected_sql}")
                print(f"  Generated SQL: {result.generated_sql}")
                if result.expected_result is not None:
                    print(f"  Expected Result: {result.expected_result}")
                if result.generated_result is not None:
                    print(f"  Generated Result: {result.generated_result}")
                if result.result_match:
                    print(f"  ✓ Results match despite SQL difference")
                elif result.expected_result is not None and result.generated_result is not None:
                    print(f"  ✗ Results differ")
        
        metrics = self._calculate_metrics("Snowflake Cortex Analyst", results)
        return results, metrics
    
    def evaluate_langchain_agent(
        self,
        db_path: str,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        use_azure: Optional[bool] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ) -> tuple[List[EvaluationResult], SolutionMetrics]:
        """
        Evaluate LangChain DB Agent solution
        
        Args:
            db_path: Path to SQLite database
            model_name: OpenAI model name (or Azure deployment name)
            api_key: OpenAI/Azure API key
            use_azure: Whether to use Azure OpenAI (auto-detected if None)
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name
            api_version: Azure API version
            
        Returns:
            Tuple of (results list, metrics)
        """
        agent = LangChainDBAgent(
            db_path=db_path,
            model_name=model_name,
            api_key=api_key,
            use_azure=use_azure,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version
        )
        
        results = []
        
        for question_data in self.questions:
            start_time = time.time()
            
            try:
                # Query agent
                response = agent.query(
                    question=question_data['question'],
                    evidence=question_data.get('evidence', '')
                )
                
                generated_sql = response['sql']
                execution_time = time.time() - start_time
                
                # Compare with expected SQL (AST-based)
                # DISABLED: Skipping exact match for now - LLM hallucination issue
                exact_match = False
                # if generated_sql:
                #     exact_match = self.normalizer.compare_ast(
                #         generated_sql,
                #         question_data['SQL']
                #     )
                
                # Execute queries and compare results
                result_match = False
                expected_result = None
                generated_result = None
                sql_execution_error = None
                
                if self.db_path and generated_sql:
                    expected_result, expected_error = self._execute_sql(question_data['SQL'])
                    generated_result, generated_error = self._execute_sql(generated_sql)
                    
                    if generated_error:
                        sql_execution_error = generated_error
                    
                    if expected_error is None and generated_error is None:
                        result_match = self._compare_results(expected_result, generated_result)
                
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=generated_sql,
                    exact_match=exact_match,
                    result_match=result_match,
                    expected_result=expected_result,
                    generated_result=generated_result,
                    error=response['error'],
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=sql_execution_error
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"ERROR on question {question_data['question_id']}: {str(e)}")
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=None,
                    exact_match=False,
                    result_match=False,
                    expected_result=None,
                    generated_result=None,
                    error=str(e),
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=None
                )
            
            results.append(result)
            status_icon = "✓" if result.exact_match else ("≈" if result.result_match else "✗")
            print(f"Processed question {result.question_id}: {status_icon}")
            if not result.exact_match and result.generated_sql:
                print(f"  Expected SQL: {result.expected_sql}")
                print(f"  Generated SQL: {result.generated_sql}")
                if result.expected_result is not None:
                    print(f"  Expected Result: {result.expected_result}")
                if result.generated_result is not None:
                    print(f"  Generated Result: {result.generated_result}")
                if result.result_match:
                    print(f"  ✓ Results match despite SQL difference")
                elif result.expected_result is not None and result.generated_result is not None:
                    print(f"  ✗ Results differ")
        
        metrics = self._calculate_metrics("LangChain DB Agent", results)
        return results, metrics
    
    def evaluate_vanilla_text2sql(
        self,
        db_path: str,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        use_azure: Optional[bool] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ) -> tuple[List[EvaluationResult], SolutionMetrics]:
        """
        Evaluate Vanilla Text2SQL solution
        
        Args:
            db_path: Path to SQLite database
            model_name: OpenAI model name (or Azure deployment name)
            api_key: OpenAI/Azure API key
            use_azure: Whether to use Azure OpenAI (auto-detected if None)
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name
            api_version: Azure API version
            
        Returns:
            Tuple of (results list, metrics)
        """
        text2sql = VanillaText2SQL(
            db_path=db_path,
            model_name=model_name,
            api_key=api_key,
            use_azure=use_azure,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version
        )
        
        results = []
        
        for question_data in self.questions:
            start_time = time.time()
            
            try:
                # Query vanilla text2sql
                response = text2sql.query(
                    question=question_data['question'],
                    evidence=question_data.get('evidence', '')
                )
                
                generated_sql = response['sql']
                execution_time = time.time() - start_time
                
                # Compare with expected SQL (AST-based)
                # DISABLED: Skipping exact match for now - LLM hallucination issue
                exact_match = False
                # if generated_sql:
                #     exact_match = self.normalizer.compare_ast(
                #         generated_sql,
                #         question_data['SQL']
                #     )
                
                # Execute queries and compare results
                result_match = False
                expected_result = None
                generated_result = None
                sql_execution_error = None
                
                if self.db_path and generated_sql:
                    expected_result, expected_error = self._execute_sql(question_data['SQL'])
                    generated_result, generated_error = self._execute_sql(generated_sql)
                    
                    if generated_error:
                        sql_execution_error = generated_error
                    
                    if expected_error is None and generated_error is None:
                        result_match = self._compare_results(expected_result, generated_result)
                
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=generated_sql,
                    exact_match=exact_match,
                    result_match=result_match,
                    expected_result=expected_result,
                    generated_result=generated_result,
                    error=response['error'],
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=sql_execution_error
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"ERROR on question {question_data['question_id']}: {str(e)}")
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=None,
                    exact_match=False,
                    result_match=False,
                    expected_result=None,
                    generated_result=None,
                    error=str(e),
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown'),
                    sql_execution_error=None
                )
            
            results.append(result)
            status_icon = "✓" if result.exact_match else ("≈" if result.result_match else "✗")
            print(f"Processed question {result.question_id}: {status_icon}")
            if not result.exact_match and result.generated_sql:
                print(f"  Expected SQL: {result.expected_sql}")
                print(f"  Generated SQL: {result.generated_sql}")
                if result.expected_result is not None:
                    print(f"  Expected Result: {result.expected_result}")
                if result.generated_result is not None:
                    print(f"  Generated Result: {result.generated_result}")
                if result.result_match:
                    print(f"  ✓ Results match despite SQL difference")
                elif result.expected_result is not None and result.generated_result is not None:
                    print(f"  ✗ Results differ")
        
        metrics = self._calculate_metrics("Vanilla Text2SQL", results)
        return results, metrics
    
    def _calculate_metrics(
        self,
        solution_name: str,
        results: List[EvaluationResult]
    ) -> SolutionMetrics:
        """Calculate metrics from evaluation results"""
        total = len(results)
        exact_matches = sum(1 for r in results if r.exact_match)
        result_matches = sum(1 for r in results if r.result_match)
        errors = sum(1 for r in results if r.error is not None)
        precision = exact_matches / total if total > 0 else 0.0
        result_accuracy = result_matches / total if total > 0 else 0.0
        
        # Calculate average execution time (excluding errors)
        successful_times = [r.execution_time for r in results if r.error is None]
        avg_execution_time = sum(successful_times) / len(successful_times) if successful_times else 0.0
        
        # Error distribution
        error_distribution = {}
        for r in results:
            if r.error:
                error_type = type(r.error).__name__ if isinstance(r.error, Exception) else "Error"
                error_distribution[error_type] = error_distribution.get(error_type, 0) + 1
        
        # Difficulty breakdown
        difficulty_breakdown = {}
        for r in results:
            diff = r.difficulty
            if diff not in difficulty_breakdown:
                difficulty_breakdown[diff] = {"total": 0, "matches": 0, "result_matches": 0, "errors": 0}
            
            difficulty_breakdown[diff]["total"] += 1
            if r.exact_match:
                difficulty_breakdown[diff]["matches"] += 1
            if r.result_match:
                difficulty_breakdown[diff]["result_matches"] += 1
            if r.error:
                difficulty_breakdown[diff]["errors"] += 1
        
        return SolutionMetrics(
            solution_name=solution_name,
            total_questions=total,
            exact_matches=exact_matches,
            result_matches=result_matches,
            errors=errors,
            precision=precision,
            result_accuracy=result_accuracy,
            avg_execution_time=avg_execution_time,
            error_distribution=error_distribution,
            difficulty_breakdown=difficulty_breakdown
        )
    
    def save_results(
        self,
        results: List[EvaluationResult],
        metrics: SolutionMetrics,
        output_dir: str
    ):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solution_name_safe = metrics.solution_name.replace(" ", "_").lower()
        
        # Save detailed results
        results_file = os.path.join(
            output_dir,
            f"{solution_name_safe}_results_{timestamp}.json"
        )
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        # Save metrics
        metrics_file = os.path.join(
            output_dir,
            f"{solution_name_safe}_metrics_{timestamp}.json"
        )
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(
                asdict(metrics),
                f,
                indent=2,
                ensure_ascii=False
            )
        
        print(f"Results saved to {results_file}")
        print(f"Metrics saved to {metrics_file}")
    
    def print_comparison(
        self,
        metrics_list: List[SolutionMetrics]
    ):
        """Print comparison of solutions"""
        print("\n" + "="*80)
        print("EVALUATION COMPARISON")
        print("="*80)
        
        for metrics in metrics_list:
            print(f"\n{metrics.solution_name}")
            print("-"*80)
            print(f"Total Questions: {metrics.total_questions}")
            print(f"Exact Matches (SQL): {metrics.exact_matches}")
            print(f"Result Matches (Query Results): {metrics.result_matches}")
            print(f"SQL Precision: {metrics.precision:.2%}")
            print(f"Result Accuracy: {metrics.result_accuracy:.2%}")
            print(f"Errors: {metrics.errors}")
            print(f"Average Execution Time: {metrics.avg_execution_time:.2f}s")
            
            print("\nDifficulty Breakdown:")
            for diff, stats in metrics.difficulty_breakdown.items():
                sql_precision = stats['matches'] / stats['total'] if stats['total'] > 0 else 0
                result_precision = stats['result_matches'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {diff}:")
                print(f"    SQL Matches: {stats['matches']}/{stats['total']} ({sql_precision:.2%})")
                print(f"    Result Matches: {stats['result_matches']}/{stats['total']} ({result_precision:.2%})")
            
            if metrics.error_distribution:
                print("\nError Distribution:")
                for error_type, count in metrics.error_distribution.items():
                    print(f"  {error_type}: {count}")
        
        print("\n" + "="*80)
    
    def export_comparison_csv(
        self,
        results_by_solution: Dict[str, List[EvaluationResult]],
        output_path: str
    ):
        """
        Export question-by-question comparison to CSV
        
        Args:
            results_by_solution: Dictionary mapping solution names to their results
            output_path: Path to output CSV file
        """
        if not results_by_solution:
            print("No results to export")
            return
        
        # Get all question IDs (use first solution as reference)
        first_solution = next(iter(results_by_solution.values()))
        question_ids = [r.question_id for r in first_solution]
        
        # Create CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Build header
            header = [
                'question_id',
                'question',
                'evidence',
                'difficulty',
                'expected_sql',
                'expected_result'
            ]
            
            # Add columns for each solution
            solution_names = list(results_by_solution.keys())
            for solution_name in solution_names:
                header.extend([
                    f'{solution_name}_sql',
                    f'{solution_name}_result',
                    f'{solution_name}_match',
                    f'{solution_name}_error',
                    f'{solution_name}_sql_exec_error'
                ])
            
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            
            # Write rows
            for question_id in question_ids:
                row = {}
                
                # Get results for this question from all solutions
                question_results = {}
                for solution_name, results in results_by_solution.items():
                    for r in results:
                        if r.question_id == question_id:
                            question_results[solution_name] = r
                            break
                
                if not question_results:
                    continue
                
                # Use first solution's result for common fields
                first_result = next(iter(question_results.values()))
                
                # Common fields
                row['question_id'] = first_result.question_id
                row['question'] = first_result.question
                row['evidence'] = first_result.evidence
                row['difficulty'] = first_result.difficulty
                row['expected_sql'] = first_result.expected_sql
                row['expected_result'] = str(first_result.expected_result) if first_result.expected_result else ''
                
                # Solution-specific fields
                for solution_name in solution_names:
                    if solution_name in question_results:
                        r = question_results[solution_name]
                        row[f'{solution_name}_sql'] = r.generated_sql or ''
                        row[f'{solution_name}_result'] = str(r.generated_result) if r.generated_result else ''
                        row[f'{solution_name}_match'] = 'YES' if r.result_match else 'NO'
                        row[f'{solution_name}_error'] = r.error or ''
                        row[f'{solution_name}_sql_exec_error'] = r.sql_execution_error or ''
                    else:
                        row[f'{solution_name}_sql'] = ''
                        row[f'{solution_name}_result'] = ''
                        row[f'{solution_name}_match'] = ''
                        row[f'{solution_name}_error'] = 'Not evaluated'
                        row[f'{solution_name}_sql_exec_error'] = ''
                
                writer.writerow(row)
        
        print(f"\nComparison CSV saved to {output_path}")


def main():
    """Run evaluation"""
    # Configuration
    dev_json_path = "data/dev_20240627/dev.json"
    db_id = "debit_card_specializing"
    
    # Get Snowflake configuration for REST API
    snowflake_warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
    snowflake_database = os.environ.get("SNOWFLAKE_DATABASE")
    snowflake_schema = os.environ.get("SNOWFLAKE_SCHEMA")
    
    # Initialize evaluator
    evaluator = TextToSQLEvaluator(
        dev_json_path=dev_json_path,
        db_id=db_id,
        db_path="debit_card.db",
        snowflake_warehouse=snowflake_warehouse,
        snowflake_database=snowflake_database,
        snowflake_schema=snowflake_schema
    )
    
    all_metrics = []
    
    # Evaluate Cortex Analyst (if configured)
    if os.environ.get("SNOWFLAKE_ACCOUNT_URL") and os.environ.get("SNOWFLAKE_TOKEN"):
        print("\n" + "="*80)
        print("Evaluating Snowflake Cortex Analyst")
        print("="*80)
        
        cortex_results, cortex_metrics = evaluator.evaluate_cortex_analyst(
            account_url=os.environ.get("SNOWFLAKE_ACCOUNT_URL"),
            token=os.environ.get("SNOWFLAKE_TOKEN"),
            semantic_model_file=os.environ.get("SNOWFLAKE_SEMANTIC_MODEL_FILE"),
            token_type=os.environ.get("SNOWFLAKE_TOKEN_TYPE", "OAUTH")
        )
        
        evaluator.save_results(cortex_results, cortex_metrics, "evaluation_results")
        all_metrics.append(cortex_metrics)
    
    # Evaluate LangChain Agent
    print("\n" + "="*80)
    print("Evaluating LangChain DB Agent")
    print("="*80)
    
    # Auto-detect Azure OpenAI or standard OpenAI
    use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_KEY"))
    
    langchain_results, langchain_metrics = evaluator.evaluate_langchain_agent(
        db_path="debit_card.db",
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY" if use_azure else "OPENAI_API_KEY"),
        use_azure=use_azure,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    
    evaluator.save_results(langchain_results, langchain_metrics, "evaluation_results")
    all_metrics.append(langchain_metrics)
    
    # Evaluate Vanilla Text2SQL
    print("\n" + "="*80)
    print("Evaluating Vanilla Text2SQL")
    print("="*80)
    
    vanilla_results, vanilla_metrics = evaluator.evaluate_vanilla_text2sql(
        db_path="debit_card.db",
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY" if use_azure else "OPENAI_API_KEY"),
        use_azure=use_azure,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    
    evaluator.save_results(vanilla_results, vanilla_metrics, "evaluation_results")
    all_metrics.append(vanilla_metrics)
    
    # Print comparison
    evaluator.print_comparison(all_metrics)


if __name__ == "__main__":
    main()
