"""
SQL Normalizer - Converts SQL queries to AST for comparison

Uses sqlglot for parsing SQL to AST and normalization.
"""

import sqlglot
from sqlglot import parse_one, Expression
from typing import Dict, Any, Optional
import json


class SQLNormalizer:
    """Normalizes SQL queries for comparison"""
    
    def __init__(self, dialect: str = "sqlite"):
        """
        Initialize SQL normalizer
        
        Args:
            dialect: SQL dialect (sqlite, postgres, mysql, etc.)
        """
        self.dialect = dialect
    
    def normalize_sql(self, sql: str) -> Optional[str]:
        """
        Normalize SQL query to a canonical form
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL string or None if parsing fails
        """
        try:
            # Parse and re-generate to normalize
            parsed = parse_one(sql, dialect=self.dialect)
            normalized = parsed.sql(dialect=self.dialect, pretty=False)
            return normalized
        except Exception as e:
            print(f"Error normalizing SQL: {e}")
            return None
    
    def sql_to_ast(self, sql: str) -> Optional[Dict[str, Any]]:
        """
        Convert SQL to AST representation
        
        Args:
            sql: SQL query string
            
        Returns:
            AST as dictionary or None if parsing fails
        """
        try:
            parsed = parse_one(sql, dialect=self.dialect)
            ast_dict = self._expression_to_dict(parsed)
            return ast_dict
        except Exception as e:
            print(f"Error parsing SQL to AST: {e}")
            return None
    
    def _expression_to_dict(self, expr: Expression) -> Dict[str, Any]:
        """
        Convert sqlglot Expression to dictionary recursively
        
        Args:
            expr: sqlglot Expression object
            
        Returns:
            Dictionary representation of the expression
        """
        if expr is None:
            return None
        
        result = {
            "type": expr.__class__.__name__
        }
        
        # Add key-value pairs from the expression
        for key, value in expr.args.items():
            if value is None:
                continue
            elif isinstance(value, list):
                result[key] = [self._expression_to_dict(v) if isinstance(v, Expression) else v 
                              for v in value]
            elif isinstance(value, Expression):
                result[key] = self._expression_to_dict(value)
            else:
                result[key] = str(value) if not isinstance(value, (bool, int, float)) else value
        
        return result
    
    def compare_ast(self, sql1: str, sql2: str) -> bool:
        """
        Compare two SQL queries by their AST
        
        Args:
            sql1: First SQL query
            sql2: Second SQL query
            
        Returns:
            True if ASTs match (exact match), False otherwise
        """
        ast1 = self.sql_to_ast(sql1)
        ast2 = self.sql_to_ast(sql2)
        
        if ast1 is None or ast2 is None:
            return False
        
        return self._compare_dict(ast1, ast2)
    
    def _compare_dict(self, d1: Any, d2: Any) -> bool:
        """
        Recursively compare two dictionaries
        
        Args:
            d1: First dictionary
            d2: Second dictionary
            
        Returns:
            True if dictionaries are equal, False otherwise
        """
        if type(d1) != type(d2):
            return False
        
        if isinstance(d1, dict):
            if set(d1.keys()) != set(d2.keys()):
                return False
            return all(self._compare_dict(d1[k], d2[k]) for k in d1.keys())
        
        elif isinstance(d1, list):
            if len(d1) != len(d2):
                return False
            return all(self._compare_dict(v1, v2) for v1, v2 in zip(d1, d2))
        
        else:
            # For primitive types, compare values
            # Normalize strings (case-insensitive, strip whitespace)
            if isinstance(d1, str) and isinstance(d2, str):
                return d1.strip().upper() == d2.strip().upper()
            return d1 == d2


def test_normalizer():
    """Test the SQL normalizer"""
    normalizer = SQLNormalizer(dialect="sqlite")
    
    # Test cases
    sql1 = "SELECT * FROM users WHERE id = 1"
    sql2 = "select * from users where id=1"
    sql3 = "SELECT * FROM users WHERE id = 2"
    
    print("Test 1: Identical queries with different formatting")
    print(f"SQL1: {sql1}")
    print(f"SQL2: {sql2}")
    print(f"Match: {normalizer.compare_ast(sql1, sql2)}")
    print()
    
    print("Test 2: Different queries")
    print(f"SQL1: {sql1}")
    print(f"SQL3: {sql3}")
    print(f"Match: {normalizer.compare_ast(sql1, sql3)}")
    print()
    
    # Test AST conversion
    print("Test 3: AST representation")
    ast = normalizer.sql_to_ast(sql1)
    print(json.dumps(ast, indent=2))


if __name__ == "__main__":
    test_normalizer()
