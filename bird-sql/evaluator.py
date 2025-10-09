"""
Evaluator module for comparing Text-to-SQL solutions

Evaluates Snowflake Cortex Analyst vs LangChain DB Agent on BIRD-SQL benchmark.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

from sql_normalizer import SQLNormalizer
from snowflake_cortex_analyst import CortexAnalystWrapper
from langchain_db_agent import LangChainDBAgent


@dataclass
class EvaluationResult:
    """Result for a single evaluation"""
    question_id: int
    question: str
    evidence: str
    expected_sql: str
    generated_sql: Optional[str]
    exact_match: bool
    error: Optional[str]
    execution_time: float
    difficulty: str


@dataclass
class SolutionMetrics:
    """Metrics for a solution"""
    solution_name: str
    total_questions: int
    exact_matches: int
    errors: int
    precision: float
    avg_execution_time: float
    error_distribution: Dict[str, int]
    difficulty_breakdown: Dict[str, Dict[str, int]]


class TextToSQLEvaluator:
    """Evaluator for Text-to-SQL solutions"""
    
    def __init__(
        self,
        dev_json_path: str,
        db_id: str = "debit_card_specializing",
        sql_dialect: str = "sqlite"
    ):
        """
        Initialize evaluator
        
        Args:
            dev_json_path: Path to dev.json file
            db_id: Database ID to filter questions
            sql_dialect: SQL dialect for normalization
        """
        self.dev_json_path = dev_json_path
        self.db_id = db_id
        self.normalizer = SQLNormalizer(dialect=sql_dialect)
        
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
            token_type=token_type
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
                
                generated_sql = response['sql']
                execution_time = time.time() - start_time
                
                # Compare with expected SQL
                exact_match = False
                if generated_sql:
                    exact_match = self.normalizer.compare_ast(
                        generated_sql,
                        question_data['SQL']
                    )
                
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=generated_sql,
                    exact_match=exact_match,
                    error=response['error'],
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown')
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
                    error=str(e),
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown')
                )
            
            results.append(result)
            if result.exact_match:
                print(f"Processed question {result.question_id}: ✓")
            else:
                print(f"Processed question {result.question_id}: ✗")
                if result.generated_sql:
                    print(f"  Expected: {result.expected_sql}")
                    print(f"  Generated: {result.generated_sql}")
        
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
                
                # Compare with expected SQL
                exact_match = False
                if generated_sql:
                    exact_match = self.normalizer.compare_ast(
                        generated_sql,
                        question_data['SQL']
                    )
                
                result = EvaluationResult(
                    question_id=question_data['question_id'],
                    question=question_data['question'],
                    evidence=question_data.get('evidence', ''),
                    expected_sql=question_data['SQL'],
                    generated_sql=generated_sql,
                    exact_match=exact_match,
                    error=response['error'],
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown')
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
                    error=str(e),
                    execution_time=execution_time,
                    difficulty=question_data.get('difficulty', 'unknown')
                )
            
            results.append(result)
            if result.exact_match:
                print(f"Processed question {result.question_id}: ✓")
            else:
                print(f"Processed question {result.question_id}: ✗")
                if result.generated_sql:
                    print(f"  Expected: {result.expected_sql}")
                    print(f"  Generated: {result.generated_sql}")
        
        metrics = self._calculate_metrics("LangChain DB Agent", results)
        return results, metrics
    
    def _calculate_metrics(
        self,
        solution_name: str,
        results: List[EvaluationResult]
    ) -> SolutionMetrics:
        """Calculate metrics from evaluation results"""
        total = len(results)
        exact_matches = sum(1 for r in results if r.exact_match)
        errors = sum(1 for r in results if r.error is not None)
        precision = exact_matches / total if total > 0 else 0.0
        
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
                difficulty_breakdown[diff] = {"total": 0, "matches": 0, "errors": 0}
            
            difficulty_breakdown[diff]["total"] += 1
            if r.exact_match:
                difficulty_breakdown[diff]["matches"] += 1
            if r.error:
                difficulty_breakdown[diff]["errors"] += 1
        
        return SolutionMetrics(
            solution_name=solution_name,
            total_questions=total,
            exact_matches=exact_matches,
            errors=errors,
            precision=precision,
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
            print(f"Exact Matches: {metrics.exact_matches}")
            print(f"Precision: {metrics.precision:.2%}")
            print(f"Errors: {metrics.errors}")
            print(f"Average Execution Time: {metrics.avg_execution_time:.2f}s")
            
            print("\nDifficulty Breakdown:")
            for diff, stats in metrics.difficulty_breakdown.items():
                precision = stats['matches'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {diff}: {stats['matches']}/{stats['total']} ({precision:.2%})")
            
            if metrics.error_distribution:
                print("\nError Distribution:")
                for error_type, count in metrics.error_distribution.items():
                    print(f"  {error_type}: {count}")
        
        print("\n" + "="*80)


def main():
    """Run evaluation"""
    # Configuration
    dev_json_path = "data/dev_20240627/dev.json"
    db_id = "debit_card_specializing"
    
    # Initialize evaluator
    evaluator = TextToSQLEvaluator(
        dev_json_path=dev_json_path,
        db_id=db_id
    )
    
    # Evaluate Cortex Analyst
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
    
    # Print comparison
    evaluator.print_comparison([cortex_metrics, langchain_metrics])


if __name__ == "__main__":
    main()
