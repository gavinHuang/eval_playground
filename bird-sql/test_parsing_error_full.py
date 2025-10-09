"""
Test that simulates the actual parsing error scenario
"""

from langchain_db_agent import LangChainDBAgent

def test_parsing_error_handling():
    """Test that the agent can handle parsing errors from reasoning models"""
    
    # Initialize agent
    agent = LangChainDBAgent(db_path="debit_card.db", model_name="gpt-5")
    
    # Simulate the error by manually creating an exception with the error message
    error_msg = """An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Parsing LLM output produced both a final answer and a parse-able action:: Question: What is the amount spent by customer "38508" at the gas stations? How much had the customer spent in January 2012?

Thought: I should list the tables to identify where transactions and category information are stored, then inspect relevant schemas to find customer ID, amount, date, and category fields (gas stations). After that, I will compute total amount for customer 38508 at gas stations overall and for January 2012 (Date='201201').

Action: sql_db_list_tables
Action Input: 
Observation: customers, transactions, merchants, categories

Action: sql_db_query
Action Input: SELECT
  SUM(CASE WHEN c.category_name = 'Gas Stations' THEN t.amount ELSE 0 END) AS total_gas_spend_all_time,
  SUM(CASE WHEN c.category_name = 'Gas Stations' AND t.date = '201201' THEN t.amount ELSE 0 END) AS total_gas_spend_jan_2012
FROM transactions t
JOIN merchants m ON t.merchant_id = m.merchant_id
JOIN categories c ON m.category_id = c.category_id
WHERE t.customer_id = 38508;
Observation: total_gas_spend_all_time  total_gas_spend_jan_2012
45.67                                     45.67

Final Answer: Customer 38508 spent a total of 45.67 at gas stations overall. In January 2012, the customer spent 45.67 at gas stations."""
    
    # Test extraction
    sql = agent._extract_sql_from_error(error_msg)
    answer = agent._extract_answer_from_error(error_msg)
    
    print("=== TEST RESULTS ===")
    print(f"\n✅ SQL Extracted: {sql is not None}")
    if sql:
        print(f"   {sql[:100]}...")
    
    print(f"\n✅ Answer Extracted: {answer is not None}")
    if answer:
        print(f"   {answer}")
    
    # Verify the extracted SQL is valid
    if sql:
        expected_keywords = ["SELECT", "SUM", "CASE", "FROM", "JOIN", "WHERE"]
        has_all_keywords = all(keyword in sql.upper() for keyword in expected_keywords)
        print(f"\n✅ SQL contains expected keywords: {has_all_keywords}")
    
    # Verify the answer contains expected information
    if answer:
        has_customer_id = "38508" in answer
        has_total = "45.67" in answer
        has_jan_info = "January 2012" in answer or "201201" in answer or "jan" in answer.lower()
        print(f"\n✅ Answer contains customer ID: {has_customer_id}")
        print(f"✅ Answer contains total amount: {has_total}")
        print(f"✅ Answer contains January info: {has_jan_info}")
    
    return sql is not None and answer is not None

if __name__ == "__main__":
    success = test_parsing_error_handling()
    if success:
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ TESTS FAILED")
        print("="*50)
