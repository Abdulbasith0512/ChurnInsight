"""
=============================================================
STEP 8: SQL Integration (SQLite)
=============================================================
Telecom Customer Churn Dataset
- Store dataset in SQLite database
- SQL queries: churn rate, churn by contract, avg charges
- Python + sqlite3 integration
=============================================================
"""

import pandas as pd
import sqlite3
import os

# ══════════════════════════════════════════════════════════════
# 1. LOAD RAW DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 8: SQL DATABASE INTEGRATION (SQLite)")
print("=" * 65)

df = pd.read_csv("TelecomCustomerChurn.csv")

# Fix TotalCharges (stored as string in raw CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(f"\n  Loaded CSV: {df.shape[0]} rows x {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# 2. CREATE SQLITE DATABASE & STORE DATA
# ══════════════════════════════════════════════════════════════
DB_NAME = "telecom_churn.db"

# Remove old DB if exists (fresh start)
if os.path.exists(DB_NAME):
    os.remove(DB_NAME)

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

print(f"\n  [OK] Created SQLite database: {DB_NAME}")

# ── Create table with explicit schema ────────────────────────
cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customerID       TEXT PRIMARY KEY,
        Gender           TEXT,
        SeniorCitizen    INTEGER,
        Partner          TEXT,
        Dependents       TEXT,
        Tenure           INTEGER,
        PhoneService     TEXT,
        MultipleLines    TEXT,
        InternetService  TEXT,
        OnlineSecurity   TEXT,
        OnlineBackup     TEXT,
        DeviceProtection TEXT,
        TechSupport      TEXT,
        StreamingTV      TEXT,
        StreamingMovies  TEXT,
        Contract         TEXT,
        PaperlessBilling TEXT,
        PaymentMethod    TEXT,
        MonthlyCharges   REAL,
        TotalCharges     REAL,
        Churn            TEXT
    )
""")
conn.commit()
print("  [OK] Created table: customers")

# ── Insert data using pandas ────────────────────────────────
df.to_sql("customers", conn, if_exists="replace", index=False)
print(f"  [OK] Inserted {len(df)} rows into 'customers' table")

# ── Verify row count ────────────────────────────────────────
count = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
print(f"  [OK] Verified: {count} rows in database")


# ══════════════════════════════════════════════════════════════
# HELPER: Run & display SQL query
# ══════════════════════════════════════════════════════════════
def run_query(conn, title, sql, description=""):
    """Execute SQL query, print results as a formatted table."""
    print(f"\n{'-' * 65}")
    print(f"  [Q] {title}")
    print(f"{'-' * 65}")
    if description:
        print(f"  >> {description}\n")
    print(f"  SQL Query:")
    for line in sql.strip().split("\n"):
        print(f"    {line}")
    print()

    result_df = pd.read_sql_query(sql, conn)
    print(f"  Result:")
    # Indent each line of the result table
    for line in result_df.to_string(index=False).split("\n"):
        print(f"    {line}")
    print()
    return result_df


# ══════════════════════════════════════════════════════════════
# 3. SQL QUERY 1 - Overall Churn Rate
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SQL QUERIES & ANALYSIS")
print("=" * 65)

q1 = """
SELECT
    COUNT(*)                                    AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN Churn = 'No'  THEN 1 ELSE 0 END) AS retained,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                           AS churn_rate_pct
FROM customers;
"""

run_query(conn, "QUERY 1: Overall Churn Rate", q1,
          "What percentage of customers have churned?")


# ══════════════════════════════════════════════════════════════
# 4. SQL QUERY 2 - Churn by Contract Type
# ══════════════════════════════════════════════════════════════
q2 = """
SELECT
    Contract,
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN Churn = 'No'  THEN 1 ELSE 0 END) AS retained,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
"""

run_query(conn, "QUERY 2: Churn Rate by Contract Type", q2,
          "Which contract type has the highest churn?")


# ══════════════════════════════════════════════════════════════
# 5. SQL QUERY 3 - Avg Monthly Charges (Churned vs Retained)
# ══════════════════════════════════════════════════════════════
q3 = """
SELECT
    Churn,
    COUNT(*)                       AS customer_count,
    ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charges,
    ROUND(MIN(MonthlyCharges), 2)  AS min_monthly_charges,
    ROUND(MAX(MonthlyCharges), 2)  AS max_monthly_charges,
    ROUND(AVG(TotalCharges), 2)    AS avg_total_charges
FROM customers
GROUP BY Churn;
"""

run_query(conn, "QUERY 3: Avg Monthly Charges - Churned vs Retained", q3,
          "Do churned customers pay more on average?")


# ══════════════════════════════════════════════════════════════
# 6. SQL QUERY 4 — Churn by Internet Service Type
# ══════════════════════════════════════════════════════════════
q4 = """
SELECT
    InternetService,
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
"""

run_query(conn, "QUERY 4: Churn by Internet Service Type", q4,
          "Does internet service type affect churn?")


# ══════════════════════════════════════════════════════════════
# 7. SQL QUERY 5 — Top 10 Highest-Risk Customers
# ══════════════════════════════════════════════════════════════
q5 = """
SELECT
    customerID,
    Contract,
    Tenure,
    MonthlyCharges,
    TotalCharges,
    InternetService,
    Churn
FROM customers
WHERE Churn = 'Yes'
    AND Contract = 'Month-to-month'
    AND Tenure <= 12
ORDER BY MonthlyCharges DESC
LIMIT 10;
"""

run_query(conn, "QUERY 5: Top 10 High-Risk Churned Customers", q5,
          "Short-tenure, month-to-month churners paying the most.")


# ══════════════════════════════════════════════════════════════
# 8. SQL QUERY 6 — Churn by Payment Method
# ══════════════════════════════════════════════════════════════
q6 = """
SELECT
    PaymentMethod,
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;
"""

run_query(conn, "QUERY 6: Churn by Payment Method", q6,
          "Which payment method is associated with highest churn?")


# ══════════════════════════════════════════════════════════════
# 9. SQL QUERY 7 — Senior Citizens Churn Analysis
# ══════════════════════════════════════════════════════════════
q7 = """
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END
                                                    AS customer_type,
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                               AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly
FROM customers
GROUP BY SeniorCitizen;
"""

run_query(conn, "QUERY 7: Senior Citizen Churn Analysis", q7,
          "Are senior citizens more likely to churn?")


# ══════════════════════════════════════════════════════════════
# 10. SQL QUERY 8 — Tenure Bucket Analysis
# ══════════════════════════════════════════════════════════════
q8 = """
SELECT
    CASE
        WHEN Tenure <= 12 THEN '01. 0-12 months'
        WHEN Tenure <= 24 THEN '02. 13-24 months'
        WHEN Tenure <= 48 THEN '03. 25-48 months'
        ELSE                   '04. 49-72 months'
    END                                             AS tenure_bucket,
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2
    )                                               AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly
FROM customers
GROUP BY tenure_bucket
ORDER BY tenure_bucket;
"""

run_query(conn, "QUERY 8: Churn by Tenure Bucket", q8,
          "How does tenure length relate to churn?")


# ══════════════════════════════════════════════════════════════
# 11. SUMMARY & KEY SQL INSIGHTS
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  SQL INTEGRATION SUMMARY")
print("=" * 65)

# Gather key stats via SQL for the summary
stats = {}
stats["churn_rate"] = pd.read_sql_query(
    "SELECT ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS v FROM customers",
    conn).iloc[0, 0]
stats["highest_churn_contract"] = pd.read_sql_query(
    q2, conn).iloc[0]["Contract"]
stats["avg_charge_churned"] = pd.read_sql_query(
    "SELECT ROUND(AVG(MonthlyCharges),2) AS v FROM customers WHERE Churn='Yes'",
    conn).iloc[0, 0]
stats["avg_charge_retained"] = pd.read_sql_query(
    "SELECT ROUND(AVG(MonthlyCharges),2) AS v FROM customers WHERE Churn='No'",
    conn).iloc[0, 0]

print(f"""
  Database        : {DB_NAME}
  Table           : customers
  Total Records   : {count}
  Queries Run     : 8

  +----------------------------------------------------------+
  |  KEY SQL INSIGHTS                                        |
  +----------------------------------------------------------+

  1. Overall churn rate: {stats['churn_rate']}%

  2. Highest churn contract: {stats['highest_churn_contract']}
     -> Month-to-month contracts have significantly higher churn

  3. Avg monthly charges:
     - Churned  : ${stats['avg_charge_churned']}
     - Retained : ${stats['avg_charge_retained']}
     -> Churned customers pay ~${stats['avg_charge_churned'] - stats['avg_charge_retained']:.2f} MORE

  4. Fiber optic internet users churn more than DSL users

  5. Electronic check payment method is linked to highest churn

  6. Senior citizens have a notably higher churn rate

  7. Customers with tenure <= 12 months are the most at-risk
""")

# ── Close connection ─────────────────────────────────────────
conn.close()
print(f"  [OK] Database connection closed.")
print(f"  [OK] Database file saved: {DB_NAME}")
print(f"       (Open with any SQLite browser to explore further)")
print("\n" + "=" * 65)
print("  STEP 8 COMPLETE - SQL Integration Done!")
print("=" * 65)
