"""
Test script for Agentar-Scale-SQL implementation

This script tests the Agentar-Scale-SQL implementation with a simple question.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agentar_scale_sql import AgentarScaleSQL


def test_agentar_scale_sql():
    """Test Agentar-Scale-SQL with a simple question"""
    
    print("=" * 80)
    print("Testing Agentar-Scale-SQL Implementation")
    print("=" * 80)
    
    # Check if database exists
    db_path = "debit_card.db"
    if not os.path.exists(db_path):
        print(f"\n✗ Database not found at {db_path}")
        print("Run: python run_evaluation.py --mode setup")
        return False
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("\n✗ No API key found")
        print("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable")
        return False
    
    try:
        # Initialize with reduced candidate counts for faster testing
        print("\nInitializing Agentar-Scale-SQL...")
        print("  - Using reduced candidate counts for faster testing")
        print("  - Reasoning candidates: 2")
        print("  - ICL candidates: 3")
        
        agentar = AgentarScaleSQL(
            db_path=db_path,
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            n_reasoning_candidates=2,
            n_icl_candidates=3
        )
        
        print("\n✓ Initialization successful")
        
        # Test with a simple question
        question = "How many gas stations are there in total?"
        evidence = ""
        
        print(f"\n{'=' * 80}")
        print(f"Test Question: {question}")
        print(f"{'=' * 80}\n")
        
        result = agentar.query(question, evidence)
        
        # Display results
        print(f"\n{'=' * 80}")
        print("Results")
        print(f"{'=' * 80}")
        
        if result['error']:
            print(f"\n✗ Error: {result['error']}")
            return False
        
        print(f"\nFinal SQL:")
        print(f"  {result['sql']}")
        
        print(f"\nAnswer:")
        print(f"  {result['answer']}")
        
        # Show candidate summary
        if result.get('candidates'):
            print(f"\n{'=' * 80}")
            print(f"Candidate Summary ({len(result['candidates'])} total)")
            print(f"{'=' * 80}")
            
            successful = [c for c in result['candidates'] if not c['execution_error']]
            failed = [c for c in result['candidates'] if c['execution_error']]
            
            print(f"\n✓ Successful: {len(successful)}")
            print(f"✗ Failed: {len(failed)}")
            
            # Show top 3 candidates
            print(f"\nTop 3 Candidates:")
            sorted_candidates = sorted(
                result['candidates'],
                key=lambda c: (
                    c['execution_error'] is None,
                    c['score'],
                    -c['temperature']
                ),
                reverse=True
            )
            
            for i, c in enumerate(sorted_candidates[:3], 1):
                status = "✓" if not c['execution_error'] else "✗"
                print(f"\n{i}. {status} [{c['source']}] Score: {c['score']}, Temp: {c['temperature']:.2f}")
                print(f"   SQL: {c['sql'][:100]}...")
                if c['execution_error']:
                    print(f"   Error: {c['execution_error'][:100]}")
        
        print(f"\n{'=' * 80}")
        print("✓ Test completed successfully!")
        print(f"{'=' * 80}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_agentar_scale_sql()
    sys.exit(0 if success else 1)
