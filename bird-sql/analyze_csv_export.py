import pandas as pd
import os

def analyze_csv_files(csv_directory):
    """
    Analyze the exported CSV files and provide summary for Snowflake loading
    """
    print("=== DEBIT CARD DATABASE CSV EXPORT SUMMARY ===\n")
    print("Ready for Snowflake loading!\n")
    
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    
    for csv_file in sorted(csv_files):
        if csv_file == 'sqlite_sequence.csv':
            continue  # Skip this SQLite internal table
            
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_sql_query(f"SELECT * FROM '{file_path.replace('.csv', '')}'", f"sqlite:///{file_path}")
        
        table_name = csv_file.replace('.csv', '').upper()
        
        print(f"TABLE: {table_name}")
        print(f"File: {csv_file}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        # Show column info with data types
        print("Column Details:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            print(f"  - {col}: {dtype} (null values: {null_count})")
        
        print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        print("-" * 50)
        print()

if __name__ == "__main__":
    csv_directory = r"c:\Users\ghuang\projects\GenAI\Eval_Playground\debit_card_csv_export"
    
    print("Reading CSV files directly...")
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    
    for csv_file in sorted(csv_files):
        if csv_file == 'sqlite_sequence.csv':
            continue  # Skip this SQLite internal table
            
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path)
        
        table_name = csv_file.replace('.csv', '').upper()
        
        print(f"TABLE: {table_name}")
        print(f"File: {csv_file}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        # Show column info
        print("Column Details:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            print(f"  - {col}: {dtype} (null values: {null_count}, sample: {sample_value})")
        
        print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        print("-" * 60)
        print()
        
    print("\n=== SNOWFLAKE LOADING RECOMMENDATIONS ===")
    print("1. Use COPY INTO command to load these CSV files")
    print("2. Consider creating a file format object for consistent parsing")
    print("3. Main tables to focus on:")
    print("   - CUSTOMERS: Customer segmentation data")
    print("   - GASSTATIONS: Gas station information")
    print("   - PRODUCTS: Product catalog")
    print("   - TRANSACTIONS_1K: Transaction records (sample of 1000)")
    print("   - YEARMONTH: Monthly consumption data")
    print("4. The sqlite_sequence.csv can be ignored (SQLite internal)")