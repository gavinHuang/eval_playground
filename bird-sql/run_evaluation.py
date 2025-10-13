"""
Quick runner script for evaluation

Usage:
    python run_evaluation.py --mode full           # Run all solutions
    python run_evaluation.py --mode cortex         # Run Cortex Analyst only
    python run_evaluation.py --mode langchain      # Run LangChain only
    python run_evaluation.py --mode vanilla        # Run Vanilla Text2SQL only
    python run_evaluation.py --mode setup          # Setup database
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import TextToSQLEvaluator
from langchain_db_agent import create_debit_card_db


def setup_database():
    """Create SQLite database from CSV files"""
    print("Setting up database...")
    csv_dir = "debit_card_csv_export"
    db_path = "debit_card.db"
    
    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory '{csv_dir}' not found")
        return False
    
    if os.path.exists(db_path):
        response = input(f"Database '{db_path}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping database creation")
            return True
        os.remove(db_path)
    
    try:
        create_debit_card_db(csv_dir, db_path)
        print(f"✓ Database created successfully at {db_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False


def run_evaluation(mode):
    """Run evaluation based on mode
    
    Args:
        mode: Either a string ('full', 'cortex', etc.) or a list of mode strings
    """
    # Normalize mode to a list
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = mode
    
    # Configuration
    dev_json_path = "data/dev_20240627/dev.json"
    db_id = "debit_card_specializing"
    db_path = "debit_card.db"
    
    if not os.path.exists(dev_json_path):
        print(f"Error: Dev JSON file not found at {dev_json_path}")
        return
    
    # Initialize evaluator with database path for result comparison
    evaluator = TextToSQLEvaluator(
        dev_json_path=dev_json_path,
        db_id=db_id,
        db_path=db_path if os.path.exists(db_path) else None
    )
    
    if not os.path.exists(db_path):
        print(f"Warning: Database not found at {db_path}. Result comparison will be skipped.")
    
    metrics_list = []
    results_by_solution = {}  # Store results for CSV export
    
    # Run Cortex Analyst evaluation
    if 'full' in modes or 'cortex' in modes:
        print("\n" + "="*80)
        print("Evaluating Snowflake Cortex Analyst")
        print("="*80)
        
        # Check environment variables
        required_vars = ["SNOWFLAKE_ACCOUNT_URL", "SNOWFLAKE_TOKEN", "SNOWFLAKE_SEMANTIC_MODEL_FILE"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
            print("Skipping Cortex Analyst evaluation")
        else:
            try:
                cortex_results, cortex_metrics = evaluator.evaluate_cortex_analyst(
                    account_url=os.environ.get("SNOWFLAKE_ACCOUNT_URL"),
                    token=os.environ.get("SNOWFLAKE_TOKEN"),
                    semantic_model_file=os.environ.get("SNOWFLAKE_SEMANTIC_MODEL_FILE"),
                    token_type=os.environ.get("SNOWFLAKE_TOKEN_TYPE", "OAUTH")
                )
                
                evaluator.save_results(cortex_results, cortex_metrics, "evaluation_results")
                metrics_list.append(cortex_metrics)
                results_by_solution["Cortex_Analyst"] = cortex_results
                print(f"✓ Cortex Analyst evaluation complete")
            except Exception as e:
                print(f"✗ Error evaluating Cortex Analyst: {e}")
    
    # Run LangChain evaluation
    if 'full' in modes or 'langchain' in modes:
        print("\n" + "="*80)
        print("Evaluating LangChain DB Agent")
        print("="*80)
        
        db_path = "debit_card.db"
        
        if not os.path.exists(db_path):
            print(f"✗ Database not found at {db_path}")
            print("Run with --mode setup to create the database")
        else:
            # Auto-detect Azure OpenAI or standard OpenAI
            use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_KEY"))
            
            # Check for required API key
            api_key_var = "AZURE_OPENAI_API_KEY" if use_azure else "OPENAI_API_KEY"
            if not os.environ.get(api_key_var):
                print(f"✗ Missing {api_key_var} environment variable")
                print("Skipping LangChain evaluation")
            else:
                try:
                    langchain_results, langchain_metrics = evaluator.evaluate_langchain_agent(
                        db_path=db_path,
                        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        api_key=os.environ.get(api_key_var),
                        use_azure=use_azure,
                        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                        api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
                    )
                    
                    evaluator.save_results(langchain_results, langchain_metrics, "evaluation_results")
                    metrics_list.append(langchain_metrics)
                    results_by_solution["LangChain_Agent"] = langchain_results
                    print(f"✓ LangChain evaluation complete")
                except Exception as e:
                    print(f"✗ Error evaluating LangChain: {e}")
    
    # Run Vanilla Text2SQL evaluation
    if 'full' in modes or 'vanilla' in modes:
        print("\n" + "="*80)
        print("Evaluating Vanilla Text2SQL")
        print("="*80)
        
        db_path = "debit_card.db"
        
        if not os.path.exists(db_path):
            print(f"✗ Database not found at {db_path}")
            print("Run with --mode setup to create the database")
        else:
            # Auto-detect Azure OpenAI or standard OpenAI
            use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_KEY"))
            
            # Check for required API key
            api_key_var = "AZURE_OPENAI_API_KEY" if use_azure else "OPENAI_API_KEY"
            if not os.environ.get(api_key_var):
                print(f"✗ Missing {api_key_var} environment variable")
                print("Skipping Vanilla Text2SQL evaluation")
            else:
                try:
                    vanilla_results, vanilla_metrics = evaluator.evaluate_vanilla_text2sql(
                        db_path=db_path,
                        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        api_key=os.environ.get(api_key_var),
                        use_azure=use_azure,
                        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                        api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
                    )
                    
                    evaluator.save_results(vanilla_results, vanilla_metrics, "evaluation_results")
                    metrics_list.append(vanilla_metrics)
                    results_by_solution["Vanilla_Text2SQL"] = vanilla_results
                    print(f"✓ Vanilla Text2SQL evaluation complete")
                except Exception as e:
                    print(f"✗ Error evaluating Vanilla Text2SQL: {e}")
    
    # Print comparison if we have results
    if metrics_list:
        evaluator.print_comparison(metrics_list)
        
        # Export CSV comparison if we have any results
        if results_by_solution:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"evaluation_results/comparison_{timestamp}.csv"
            evaluator.export_comparison_csv(results_by_solution, csv_path)
    else:
        print("\n✗ No evaluations completed")


def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
    parser.add_argument(
        "--mode",
        nargs='+',
        default=["full"],
        help="Evaluation mode(s): full, cortex, langchain, vanilla, setup (default: full). Can specify multiple modes separated by spaces."
    )
    
    args = parser.parse_args()
    
    # Flatten modes if comma-separated (e.g., "cortex,vanilla")
    modes = []
    for mode in args.mode:
        modes.extend(mode.split(','))
    
    # Validate modes
    valid_modes = ["full", "cortex", "langchain", "vanilla", "setup"]
    invalid_modes = [m for m in modes if m not in valid_modes]
    if invalid_modes:
        print(f"Error: Invalid mode(s): {', '.join(invalid_modes)}")
        print(f"Valid modes: {', '.join(valid_modes)}")
        return
    
    # Handle setup mode separately
    if "setup" in modes:
        setup_database()
        if len(modes) == 1:
            return
        # Remove setup from modes to continue with evaluation
        modes = [m for m in modes if m != "setup"]
    
    # If full is specified, just use that
    if "full" in modes:
        run_evaluation("full")
    elif modes:
        # Convert multiple modes to a combined mode list
        run_evaluation(modes)


if __name__ == "__main__":
    main()
