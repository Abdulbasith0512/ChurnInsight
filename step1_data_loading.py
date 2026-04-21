"""
=============================================================
STEP 1: Data Loading & Understanding
=============================================================
Telecom Customer Churn Dataset
Source: https://github.com/YBIFoundation/Dataset
=============================================================
"""

import pandas as pd
import numpy as np

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("TelecomCustomerChurn.csv")
print("✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── 2. First 5 Rows ─────────────────────────────────────────
print("=" * 60)
print("📋 FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string())
print()

# ── 3. Column Names & Data Types ────────────────────────────
print("=" * 60)
print("📊 COLUMN NAMES & DATA TYPES")
print("=" * 60)
print(df.dtypes.to_string())
print()

# ── 4. Missing Values ───────────────────────────────────────
print("=" * 60)
print("❓ MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})
print(missing_df[missing_df["Missing Count"] > 0].to_string() 
      if missing.sum() > 0 
      else "   ✅ No missing values found!")
print(f"\n   Total missing values: {missing.sum()}")
print()

# ── 5. Basic Statistics ─────────────────────────────────────
print("=" * 60)
print("📈 BASIC STATISTICS (Numerical Features)")
print("=" * 60)
print(df.describe().round(2).to_string())
print()

print("=" * 60)
print("📈 BASIC STATISTICS (Categorical Features)")
print("=" * 60)
print(df.describe(include="object").to_string())
print()

# ── 6. Identify Target Variable ─────────────────────────────
print("=" * 60)
print("🎯 TARGET VARIABLE")
print("=" * 60)
target = "Churn"
print(f"   Target Column: '{target}'")
print(f"   Unique Values: {df[target].unique()}")
print(f"\n   Value Counts:")
print(df[target].value_counts().to_string())
print()

# ── 7. Identify Numerical & Categorical Features ────────────
print("=" * 60)
print("🔢 NUMERICAL FEATURES")
print("=" * 60)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"   Count: {len(numerical_cols)}")
for col in numerical_cols:
    print(f"   • {col}")
print()

print("=" * 60)
print("🔤 CATEGORICAL FEATURES")
print("=" * 60)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
# Remove customerID and target from feature lists
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")
if target in categorical_cols:
    categorical_cols.remove(target)

print(f"   Count: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"   • {col} → {df[col].nunique()} unique values")
print()

# ── Summary ──────────────────────────────────────────────────
print("=" * 60)
print("📝 SUMMARY")
print("=" * 60)
print(f"   Total Samples     : {df.shape[0]}")
print(f"   Total Features    : {df.shape[1] - 2}  (excluding customerID & target)")
print(f"   Numerical Features: {len(numerical_cols)}")
print(f"   Categorical Features: {len(categorical_cols)}")
print(f"   Target Variable   : '{target}'")
print(f"   Missing Values    : {missing.sum()}")
print("=" * 60)
