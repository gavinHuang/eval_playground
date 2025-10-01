import sqlite3
import pandas as pd
import os
from pathlib import Path

def sqlite_to_csv(sqlite_file_path, output_dir):
    """
    Convert SQLite database tables to CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Found {len(tables)} tables in the database:")
    
    for table in tables:
        table_name = table[0]
        print(f"Processing table: {table_name}")
        
        # Read table into pandas DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Save as CSV
        csv_file = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"  - Exported {len(df)} rows to {csv_file}")
        print(f"  - Columns: {list(df.columns)}")
        print()
    
    conn.close()
    print("Conversion completed!")

if __name__ == "__main__":
    sqlite_file = r"c:\Users\ghuang\projects\GenAI\Eval_Playground\bird-sql\data\dev_20240627\dev_databases\debit_card_specializing\debit_card_specializing.sqlite"
    output_directory = r"c:\Users\ghuang\projects\GenAI\Eval_Playground\debit_card_csv_export"
    
    print(f"Converting SQLite database: {sqlite_file}")
    print(f"Output directory: {output_directory}")
    print()
    
    sqlite_to_csv(sqlite_file, output_directory)