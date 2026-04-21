"""
=============================================================
STEP 2: Data Cleaning & Preprocessing
=============================================================
Telecom Customer Churn Dataset
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("TelecomCustomerChurn.csv")
print(f"Original shape: {df.shape}\n")

# ── 2. Fix TotalCharges (string -> numeric) ──────────────────
# TotalCharges was loaded as string — likely has blank/space values
print("=" * 60)
print("FIX: TotalCharges Column (str -> numeric)")
print("=" * 60)

# Check non-numeric entries
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
null_total = df["TotalCharges"].isnull().sum()
print(f"   Non-numeric / blank entries found: {null_total}")

if null_total > 0:
    # These are typically new customers with Tenure = 0
    print(f"   Tenure values for those rows:")
    print(f"   {df[df['TotalCharges'].isnull()]['Tenure'].unique()}")
    # Fill with 0 (new customers, no charges yet)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print("   -> Filled with 0 (new customers)")

print(f"   TotalCharges dtype now: {df['TotalCharges'].dtype}\n")

# ── 3. Drop customerID (not a feature) ──────────────────────
print("=" * 60)
print("DROP: customerID (identifier, not a feature)")
print("=" * 60)
df.drop(columns=["customerID"], inplace=True)
print(f"   Remaining columns: {df.shape[1]}\n")

# ── 4. Encode Target Variable ───────────────────────────────
print("=" * 60)
print("ENCODE: Target Variable (Churn)")
print("=" * 60)
print(f"   Before: {df['Churn'].unique()}")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"   After:  {df['Churn'].unique()}")
print(f"   Distribution:\n{df['Churn'].value_counts().to_string()}\n")

# ── 5. Encode Binary Categorical Features ───────────────────
print("=" * 60)
print("ENCODE: Binary Categorical Features (Yes/No -> 1/0)")
print("=" * 60)

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"   {col}: encoded")

# Gender encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
print(f"   Gender: encoded (Male=1, Female=0)")
print()

# ── 6. Encode Multi-Category Features (One-Hot) ─────────────
print("=" * 60)
print("ENCODE: Multi-Category Features (One-Hot Encoding)")
print("=" * 60)

multi_cat_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]

print(f"   Columns to encode: {len(multi_cat_cols)}")
for col in multi_cat_cols:
    print(f"   - {col}: {df[col].nunique()} categories -> {df[col].unique()}")

df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)
print(f"\n   Shape after one-hot encoding: {df.shape}\n")

# ── 7. Final Check ──────────────────────────────────────────
print("=" * 60)
print("FINAL DATA CHECK")
print("=" * 60)
print(f"   Shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types:\n{df.dtypes.value_counts().to_string()}")
print()

# ── 8. Show All Final Columns ────────────────────────────────
print("=" * 60)
print("FINAL COLUMNS")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col} ({df[col].dtype})")
print()

# ── 9. Save Cleaned Data ────────────────────────────────────
df.to_csv("TelecomCustomerChurn_cleaned.csv", index=False)
print("Cleaned dataset saved to: TelecomCustomerChurn_cleaned.csv")

# ── 10. Quick Correlation with Target ────────────────────────
print("\n" + "=" * 60)
print("TOP 10 FEATURES CORRELATED WITH CHURN")
print("=" * 60)
corr = df.corr(numeric_only=True)["Churn"].drop("Churn").abs().sort_values(ascending=False)
for feat, val in corr.head(10).items():
    bar = "#" * int(val * 40)
    print(f"   {feat:<35s} {val:.4f}  {bar}")
