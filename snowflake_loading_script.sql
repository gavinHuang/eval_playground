-- Snowflake Loading Script for Debit Card Database
-- Run these commands in your Snowflake worksheet

-- 1. Create a database and schema (adjust names as needed)
CREATE DATABASE IF NOT EXISTS DEBIT_CARD_DB;
USE DATABASE DEBIT_CARD_DB;
CREATE SCHEMA IF NOT EXISTS RAW_DATA;
USE SCHEMA RAW_DATA;

-- 2. Create a file format for CSV files
CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('NULL', 'null', '')
    EMPTY_FIELD_AS_NULL = TRUE;

-- 3. Create tables with appropriate data types

-- Customers table
CREATE OR REPLACE TABLE CUSTOMERS (
    CustomerID INT,
    Segment VARCHAR(50),
    Currency VARCHAR(10)
);

-- Gas Stations table
CREATE OR REPLACE TABLE GASSTATIONS (
    GasStationID INT,
    ChainID INT,
    Country VARCHAR(10),
    Segment VARCHAR(50)
);

-- Products table
CREATE OR REPLACE TABLE PRODUCTS (
    ProductID INT,
    Description VARCHAR(255)
);

-- Transactions table
CREATE OR REPLACE TABLE TRANSACTIONS_1K (
    TransactionID INT,
    Date DATE,
    Time TIME,
    CustomerID INT,
    CardID INT,
    GasStationID INT,
    ProductID INT,
    Amount INT,
    Price DECIMAL(10,2)
);

-- Year Month table
CREATE OR REPLACE TABLE YEARMONTH (
    CustomerID INT,
    Date INT,  -- Format: YYYYMM
    Consumption DECIMAL(10,2)
);

-- 4. Create a stage for file uploads (you'll need to upload files here first)
CREATE OR REPLACE STAGE DEBIT_CARD_STAGE;

-- 5. After uploading your CSV files to the stage, use these COPY INTO commands:

-- Load customers data
COPY INTO CUSTOMERS
FROM @DEBIT_CARD_STAGE/customers.csv
FILE_FORMAT = CSV_FORMAT;

-- Load gas stations data
COPY INTO GASSTATIONS
FROM @DEBIT_CARD_STAGE/gasstations.csv
FILE_FORMAT = CSV_FORMAT;

-- Load products data
COPY INTO PRODUCTS
FROM @DEBIT_CARD_STAGE/products.csv
FILE_FORMAT = CSV_FORMAT;

-- Load transactions data
COPY INTO TRANSACTIONS_1K
FROM @DEBIT_CARD_STAGE/transactions_1k.csv
FILE_FORMAT = CSV_FORMAT;

-- Load year month data
COPY INTO YEARMONTH
FROM @DEBIT_CARD_STAGE/yearmonth.csv
FILE_FORMAT = CSV_FORMAT;

-- 6. Verify the data loads
SELECT 'CUSTOMERS' as table_name, COUNT(*) as row_count FROM CUSTOMERS
UNION ALL
SELECT 'GASSTATIONS' as table_name, COUNT(*) as row_count FROM GASSTATIONS
UNION ALL
SELECT 'PRODUCTS' as table_name, COUNT(*) as row_count FROM PRODUCTS
UNION ALL
SELECT 'TRANSACTIONS_1K' as table_name, COUNT(*) as row_count FROM TRANSACTIONS_1K
UNION ALL
SELECT 'YEARMONTH' as table_name, COUNT(*) as row_count FROM YEARMONTH;

-- 7. Sample queries to test the data
SELECT * FROM CUSTOMERS LIMIT 10;
SELECT * FROM TRANSACTIONS_1K LIMIT 10;
SELECT * FROM YEARMONTH LIMIT 10;