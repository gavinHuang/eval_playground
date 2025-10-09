"""
Test script to verify the evaluation framework setup

This script checks:
1. All dependencies are installed
2. Configuration is present
3. Data files exist
4. Components can be imported
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    packages = [
        ("requests", "requests"),
        ("dotenv", "python-dotenv"),
        ("jwt", "PyJWT"),
        ("cryptography", "cryptography"),
        ("sqlglot", "sqlglot"),
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_openai", "langchain-openai"),
        ("openai", "openai"),
        ("pandas", "pandas")
    ]
    
    all_ok = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_files():
    """Check if required files exist"""
    print("\nChecking files...")
    
    required_files = [
        "data/dev_20240627/dev.json",
        "debit_card_csv_export/customers.csv",
        "debit_card_csv_export/gasstations.csv",
        "debit_card_csv_export/products.csv",
        "debit_card_csv_export/transactions_1k.csv",
        "debit_card_csv_export/yearmonth.csv"
    ]
    
    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_modules():
    """Check if project modules can be imported"""
    print("\nChecking project modules...")
    
    modules = [
        "sql_normalizer",
        "snowflake_cortex_analyst",
        "langchain_db_agent",
        "evaluator"
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}.py")
        except ImportError as e:
            print(f"  ✗ {module_name}.py - ERROR: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠ {module_name}.py - Warning: {e}")
    
    return all_ok


def check_config():
    """Check if environment configuration exists"""
    print("\nChecking configuration...")
    
    import os
    
    # Check if .env exists
    env_exists = Path(".env").exists()
    env_example_exists = Path(".env.example").exists()
    
    if env_exists:
        print(f"  ✓ .env file found")
    else:
        print(f"  ✗ .env file not found")
        if env_example_exists:
            print(f"    → Copy .env.example to .env and configure")
    
    # Check essential env vars (if .env exists)
    if env_exists:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass
        
        snowflake_vars = [
            "SNOWFLAKE_ACCOUNT_URL",
            "SNOWFLAKE_TOKEN",
            "SNOWFLAKE_SEMANTIC_MODEL_FILE"
        ]
        
        openai_vars = [
            "OPENAI_API_KEY"
        ]
        
        print("\n  Snowflake configuration:")
        for var in snowflake_vars:
            if os.environ.get(var):
                print(f"    ✓ {var} is set")
            else:
                print(f"    ✗ {var} is not set")
        
        print("\n  OpenAI configuration:")
        for var in openai_vars:
            if os.environ.get(var):
                print(f"    ✓ {var} is set")
            else:
                print(f"    ✗ {var} is not set")
    
    return env_exists


def test_sql_normalizer():
    """Test SQL normalizer functionality"""
    print("\nTesting SQL normalizer...")
    
    try:
        from sql_normalizer import SQLNormalizer
        
        normalizer = SQLNormalizer(dialect="sqlite")
        
        sql1 = "SELECT * FROM users WHERE id = 1"
        sql2 = "select * from users where id=1"
        
        match = normalizer.compare_ast(sql1, sql2)
        
        if match:
            print("  ✓ SQL normalizer working correctly")
            return True
        else:
            print("  ✗ SQL normalizer failed to match identical queries")
            return False
    except Exception as e:
        print(f"  ✗ SQL normalizer test failed: {e}")
        return False


def main():
    print("="*80)
    print("EVALUATION FRAMEWORK SETUP VERIFICATION")
    print("="*80)
    
    results = {
        "Dependencies": check_imports(),
        "Files": check_files(),
        "Modules": check_modules(),
        "Configuration": check_config(),
        "SQL Normalizer": test_sql_normalizer()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_ok = True
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_ok = False
    
    print("\n" + "="*80)
    
    if all_ok:
        print("✓ All checks passed! You're ready to run evaluations.")
        print("\nNext steps:")
        print("  1. Ensure .env is configured with your credentials")
        print("  2. Run: python run_evaluation.py --mode setup")
        print("  3. Run: python run_evaluation.py --mode full")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Copy .env.example to .env and configure")
        print("  - Ensure data files are in the correct location")
    
    print("="*80)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
