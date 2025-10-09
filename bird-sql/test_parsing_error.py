"""
Test script to verify the parsing error handling fix
"""

from langchain_db_agent import LangChainDBAgent

# Test the agent with a query that might trigger parsing errors
db_path = "debit_card.db"

# Initialize agent with GPT-5 (reasoning model)
agent = LangChainDBAgent(
    db_path=db_path,
    model_name="gpt-5",
    temperature=0
)

# Test query
question = "What is the amount spent by customer 38508 at the gas stations? How much had the customer spent in January 2012?"

print(f"Testing query: {question}\n")

result = agent.query(question)

print("\n=== RESULT ===")
print(f"SQL: {result['sql']}")
print(f"\nAnswer: {result['answer']}")
print(f"\nError: {result['error']}")

if result['error']:
    print("\n❌ Query failed with error")
else:
    print("\n✅ Query succeeded")
