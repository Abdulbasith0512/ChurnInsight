"""
=============================================================
STEP 5: Feature Engineering
=============================================================
Telecom Customer Churn Dataset
- Encode categorical variables
- Scale numerical features
- Create new derived features
- Save ML-ready dataset
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Original Dataset ────────────────────────────────
df = pd.read_csv("TelecomCustomerChurn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df.drop(columns=["customerID"], inplace=True)

print(f"Original shape: {df.shape}")
print("=" * 65)
print("  STEP 5: FEATURE ENGINEERING")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# PART 1: Create New Derived Features
# ══════════════════════════════════════════════════════════════
print("\n--- PART 1: Creating New Derived Features ---\n")

# Feature 1: Average Monthly Charge (TotalCharges / Tenure)
df["AvgMonthlyCharge"] = np.where(
    df["Tenure"] > 0,
    df["TotalCharges"] / df["Tenure"],
    df["MonthlyCharges"]
)
print("  1. AvgMonthlyCharge = TotalCharges / Tenure")

# Feature 2: Tenure Groups (binned)
bins = [0, 12, 24, 48, 72]
labels = ["0-12m", "12-24m", "24-48m", "48-72m"]
df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins, labels=labels, include_lowest=True)
print("  2. TenureGroup (0-12m, 12-24m, 24-48m, 48-72m)")

# Feature 3: Number of services subscribed
service_cols = ["PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

def count_services(row):
    count = 0
    for col in service_cols:
        if row[col] == "Yes" or row[col] == "Fiber optic" or row[col] == "DSL":
            count += 1
    return count

df["NumServices"] = df.apply(count_services, axis=1)
print("  3. NumServices (total services subscribed: 0-9)")

# Feature 4: Has Internet Service (binary)
df["HasInternet"] = (df["InternetService"] != "No").astype(int)
print("  4. HasInternet (binary)")

# Feature 5: High Value Customer (above median monthly charges + long tenure)
median_charges = df["MonthlyCharges"].median()
median_tenure = df["Tenure"].median()
df["HighValueCustomer"] = (
    (df["MonthlyCharges"] > median_charges) & (df["Tenure"] > median_tenure)
).astype(int)
print("  5. HighValueCustomer (high charges + long tenure)")

# Feature 6: Charge per Service
df["ChargePerService"] = np.where(
    df["NumServices"] > 0,
    df["MonthlyCharges"] / df["NumServices"],
    0
)
print("  6. ChargePerService = MonthlyCharges / NumServices")

print(f"\n  New features created: 6")
print(f"  Shape after new features: {df.shape}")

# ══════════════════════════════════════════════════════════════
# PART 2: Encode Categorical Variables
# ══════════════════════════════════════════════════════════════
print("\n--- PART 2: Encoding Categorical Variables ---\n")

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("  Target 'Churn': Yes=1, No=0")

# Binary features (Label Encoding)
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
binary_cols = ["Gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = df[col].map(binary_map)
    print(f"  {col}: Label Encoded (1/0)")

# Multi-line: already binary after removing "No internet service" was handled
binary_yes_no = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for col in binary_yes_no:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"  {col}: Label Encoded (1/0)")

# Multi-category features (One-Hot Encoding)
multi_cat_cols = ["InternetService", "Contract", "PaymentMethod", "TenureGroup"]
print(f"\n  One-Hot Encoding: {multi_cat_cols}")
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)

print(f"  Shape after encoding: {df.shape}")

# ══════════════════════════════════════════════════════════════
# PART 3: Scale Numerical Features
# ══════════════════════════════════════════════════════════════
print("\n--- PART 3: Scaling Numerical Features (StandardScaler) ---\n")

num_cols_to_scale = ["Tenure", "MonthlyCharges", "TotalCharges",
                     "AvgMonthlyCharge", "NumServices", "ChargePerService"]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])

print(f"  Scaled columns: {num_cols_to_scale}")
print(f"\n  Before Scaling (sample):")
print(f"  {'Feature':<20s} {'Mean':>10s} {'Std':>10s}")
print(f"  {'-'*40}")
for col in num_cols_to_scale:
    print(f"  {col:<20s} {df[col].mean():>10.2f} {df[col].std():>10.2f}")

print(f"\n  After Scaling (sample):")
print(f"  {'Feature':<20s} {'Mean':>10s} {'Std':>10s}")
print(f"  {'-'*40}")
for col in num_cols_to_scale:
    print(f"  {col:<20s} {df_scaled[col].mean():>10.4f} {df_scaled[col].std():>10.4f}")

# ══════════════════════════════════════════════════════════════
# PART 4: Final Dataset Overview
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL DATASET OVERVIEW")
print("=" * 65)

# Separate features and target
X = df_scaled.drop(columns=["Churn"])
y = df_scaled["Churn"]

print(f"\n  Features (X) shape : {X.shape}")
print(f"  Target   (y) shape : {y.shape}")
print(f"  Missing values     : {df_scaled.isnull().sum().sum()}")
print(f"  All numeric dtypes : {df_scaled.select_dtypes(include=np.number).shape[1] == df_scaled.shape[1]}")

print(f"\n  All Columns ({len(df_scaled.columns)}):")
for i, col in enumerate(df_scaled.columns, 1):
    marker = " [TARGET]" if col == "Churn" else ""
    print(f"    {i:2d}. {col} ({df_scaled[col].dtype}){marker}")

# ── Save Processed Datasets ─────────────────────────────────
df_scaled.to_csv("TelecomChurn_featured.csv", index=False)
print(f"\n  Saved: TelecomChurn_featured.csv")

# Also save unscaled version (useful for interpretability)
df.to_csv("TelecomChurn_encoded.csv", index=False)
print(f"  Saved: TelecomChurn_encoded.csv (unscaled)")

print("\n" + "=" * 65)
print("  Dataset is READY for Machine Learning!")
print("=" * 65)
