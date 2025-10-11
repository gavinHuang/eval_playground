"""
Deep dive into why the evaluation is failing
"""

import sqlite3
from langchain_db_agent import LangChainDBAgent

# Initialize agent
agent = LangChainDBAgent(db_path="debit_card.db", model_name="gpt-4o-mini")

print("="*80)
print("ANALYZING WHY EVALUATION IS FAILING")
print("="*80)

# Test Question 1
print("\n1. QUESTION 1470:")
print("   Question: How many gas stations in CZE has Premium gas?")
print("   Expected SQL: SELECT COUNT(GasStationID) FROM gasstations WHERE Country = 'CZE' AND Segment = 'Premium'")

# Check what the database actually has
conn = sqlite3.connect('debit_card.db')
cursor = conn.cursor()

print("\n   Checking database schema:")
cursor.execute("PRAGMA table_info(gasstations)")
columns = cursor.fetchall()
print(f"   OK gasstations table exists")
print(f"   OK Columns: {[c[1] for c in columns]}")

print("\n   Executing expected SQL:")
cursor.execute("SELECT COUNT(GasStationID) FROM gasstations WHERE Country = 'CZE' AND Segment = 'Premium'")
expected_result = cursor.fetchall()
print(f"   OK Result: {expected_result}")

# Now check what LangChain agent sees
print("\n   What LangChain agent sees when querying the schema:")
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri(f"sqlite:///debit_card.db")
print(f"   Tables: {db.get_usable_table_names()}")
print(f"\n   Schema info:")
print(db.get_table_info())

conn.close()

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

print("""
PROBLEM #1: Agent is hallucinating schema
- The database HAS the correct tables and columns
- Expected SQL queries execute successfully
- But the LangChain agent is NOT using the actual schema
- Instead, it's making up table/column names that don't exist

EVIDENCE:
- Question 1470: Agent queried "gas_stations" (wrong) instead of "gasstations"
- Question 1470: Agent used "station_id, fuel_id" (don't exist)
- Question 1471: Agent queried "energy_data, consumption" (don't exist)

PROBLEM #2: Why is exact match 0%?
- The agent is generating valid SQL for a DIFFERENT database schema
- Even though the actual schema is available to the agent
- The AST comparison correctly identifies these as non-matching

PROBLEM #3: Why is result match only 1.56%?
- Most generated SQL fails to execute (wrong table/column names)
- Only Question 1471 happened to generate SQL that executed
- Even when it executed, it queried the wrong tables

SOLUTION NEEDED:
1. Ensure the agent is actually looking at the database schema
2. The agent should not hallucinate non-existent tables/columns
3. Need to debug why LangChain agent isn't using the schema properly
""")
