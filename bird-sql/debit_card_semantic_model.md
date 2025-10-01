# Debit Card Specializing Database - Semantic Model

## Domain Overview
This semantic model represents a gas station debit card transaction system that tracks customer purchases, product sales, and consumption patterns across different gas station chains and locations.

## Business Context
The system captures debit card transactions at gas stations, allowing analysis of:
- Customer purchasing behavior and segments
- Gas station performance by chain and location
- Product sales and pricing analysis
- Consumption patterns over time
- Geographic and demographic insights

## Core Business Entities

### 1. Customer (customers table)
**Business Definition**: Individual customers who use debit cards at gas stations
- **Primary Key**: CustomerID (integer)
- **Attributes**:
  - `Segment` (text): Customer classification/category (client segment)
  - `Currency` (text): Customer's preferred/base currency
- **Business Rules**:
  - Each customer belongs to exactly one segment
  - Customer segments likely represent loyalty tiers, demographics, or value classifications

### 2. Gas Station (gasstations table)
**Business Definition**: Physical locations where fuel and products are sold
- **Primary Key**: GasStationID (integer)
- **Attributes**:
  - `ChainID` (integer): Identifies the gas station chain/brand
  - `Country` (text): Geographic location country
  - `Segment` (text): Chain segment classification
- **Business Rules**:
  - Each gas station belongs to exactly one chain
  - Chain segments likely represent market positioning (premium, budget, etc.)

### 3. Product (products table)
**Business Definition**: Items available for purchase at gas stations
- **Primary Key**: ProductID (integer)
- **Attributes**:
  - `Description` (text): Product name and details
- **Business Rules**:
  - Products can be sold at multiple gas stations
  - Likely includes fuel types, convenience items, services

### 4. Transaction (transactions_1k table)
**Business Definition**: Individual purchase events using debit cards
- **Primary Key**: TransactionID (integer)
- **Attributes**:
  - `Date` (date): Transaction date
  - `Time` (text): Transaction time
  - `CustomerID` (integer): Customer making purchase
  - `CardID` (integer): Debit card used
  - `GasStationID` (integer): Location of purchase
  - `ProductID` (integer): Product purchased
  - `Amount` (integer): Quantity purchased
  - `Price` (real): Unit price or total price
- **Business Rules**:
  - Each transaction involves one customer, one card, one gas station, one product
  - Amount represents quantity, Price represents monetary value
  - Time tracking allows for temporal analysis

### 5. Customer Consumption (yearmonth table)
**Business Definition**: Aggregated customer consumption data by time period
- **Composite Primary Key**: CustomerID + Date
- **Attributes**:
  - `CustomerID` (integer): Customer identifier
  - `Date` (text): Time period (likely YYYY-MM format)
  - `Consumption` (real): Aggregated consumption amount
- **Business Rules**:
  - Pre-aggregated data for performance
  - Enables time-series analysis of customer behavior

## Relationships & Data Model

### Primary Relationships
1. **Customer ← Transaction** (1:N)
   - One customer can have many transactions
   - `customers.CustomerID` ← `transactions_1k.CustomerID`

2. **Gas Station ← Transaction** (1:N)
   - One gas station can process many transactions
   - `gasstations.GasStationID` ← `transactions_1k.GasStationID`

3. **Product ← Transaction** (1:N)
   - One product can appear in many transactions
   - `products.ProductID` ← `transactions_1k.ProductID`

4. **Customer ← Consumption** (1:N)
   - One customer can have consumption records for multiple periods
   - `customers.CustomerID` ← `yearmonth.CustomerID`

### Dimensional Model Structure

#### Fact Tables
1. **Transaction Fact** (`transactions_1k`)
   - **Measures**: Amount, Price
   - **Dimensions**: Date, Time, Customer, Gas Station, Product, Card

2. **Consumption Fact** (`yearmonth`)
   - **Measures**: Consumption
   - **Dimensions**: Customer, Date (time period)

#### Dimension Tables
1. **Customer Dimension** (`customers`)
   - Attributes: Segment, Currency
   - Hierarchies: Segment → Customer

2. **Gas Station Dimension** (`gasstations`)
   - Attributes: Chain, Country, Segment
   - Hierarchies: Country → Chain → Gas Station

3. **Product Dimension** (`products`)
   - Attributes: Description
   - Hierarchies: Product Category → Product (inferred from description)

4. **Date Dimension** (derived from transaction dates)
   - Hierarchies: Year → Quarter → Month → Day

## Key Performance Indicators (KPIs)

### Financial Metrics
- **Total Revenue**: SUM(transactions.Price)
- **Average Transaction Value**: AVG(transactions.Price)
- **Revenue per Customer**: Total Revenue / COUNT(DISTINCT customers)
- **Revenue per Gas Station**: Total Revenue / COUNT(DISTINCT gas stations)

### Volume Metrics
- **Total Transaction Volume**: SUM(transactions.Amount)
- **Transaction Count**: COUNT(transactions)
- **Average Purchase Quantity**: AVG(transactions.Amount)

### Customer Metrics
- **Active Customers**: COUNT(DISTINCT customers with transactions)
- **Customer Retention Rate**: Returning customers / Total customers
- **Average Consumption per Customer**: AVG(yearmonth.Consumption)

### Operational Metrics
- **Transactions per Gas Station**: COUNT(transactions) / COUNT(DISTINCT gas stations)
- **Peak Transaction Hours**: Transaction count by time
- **Geographic Performance**: Metrics by country

## Business Questions & Use Cases

### Customer Analytics
1. Which customer segments generate the most revenue?
2. How does consumption vary by customer segment over time?
3. What are the purchasing patterns of high-value customers?
4. Which customers are at risk of churning?

### Geographic & Location Analytics
1. Which countries/regions perform best?
2. How do different gas station chains compare?
3. Which locations have the highest transaction volumes?
4. What are the regional preferences for products?

### Product Analytics
1. Which products are most popular?
2. How do product preferences vary by customer segment?
3. What are the seasonal trends in product sales?
4. Which products have the highest profit margins?

### Temporal Analytics
1. What are the peak transaction times and days?
2. How do sales patterns change seasonally?
3. Are there trends in customer consumption over time?
4. What are the growth rates by time period?

## Data Quality Considerations

### Key Data Quality Rules
1. **Referential Integrity**: All foreign keys must reference valid primary keys
2. **Date Consistency**: Transaction dates should be valid and within reasonable ranges
3. **Price Validation**: Prices should be positive values
4. **Amount Validation**: Transaction amounts should be positive
5. **Customer Uniqueness**: Each CustomerID should represent a unique customer

### Potential Data Issues
1. Missing or null values in critical fields
2. Inconsistent date formats between tables
3. Currency conversion needs for multi-currency analysis
4. Duplicate transactions
5. Orphaned records due to referential integrity issues

## Security & Privacy Considerations
- Customer data requires protection under data privacy regulations
- Card information may need encryption or masking
- Access controls should be implemented based on business roles
- Audit trails for data access and modifications
- Compliance with PCI DSS for card transaction data

## Extension Opportunities
1. **Customer Segmentation Enhancement**: ML-based dynamic segmentation
2. **Predictive Analytics**: Customer lifetime value, churn prediction
3. **Real-time Analytics**: Streaming transaction processing
4. **Geographic Intelligence**: Integration with location services
5. **Product Recommendation**: Collaborative filtering based on purchase history