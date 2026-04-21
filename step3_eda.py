"""
=============================================================
STEP 3: Exploratory Data Analysis (EDA)
=============================================================
Telecom Customer Churn Dataset
Visualizations using Matplotlib & Seaborn
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style Setup ──────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

# ── Load Original Dataset (for readable labels) ─────────────
df = pd.read_csv("TelecomCustomerChurn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns\n")

# ══════════════════════════════════════════════════════════════
# PLOT 1: Churn Distribution
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Countplot
colors = ["#2ecc71", "#e74c3c"]
ax1 = axes[0]
churn_counts = df["Churn"].value_counts()
bars = ax1.bar(churn_counts.index, churn_counts.values, color=colors,
               edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold", fontsize=11)
ax1.set_title("Churn Distribution (Count)")
ax1.set_xlabel("Churn")
ax1.set_ylabel("Count")
ax1.set_ylim(0, churn_counts.max() * 1.15)

# Pie chart
ax2 = axes[1]
ax2.pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90, explode=(0, 0.05),
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax2.set_title("Churn Distribution (Proportion)")

plt.tight_layout()
plt.savefig("eda_1_churn_distribution.png", bbox_inches="tight")
plt.close()
print("Saved: eda_1_churn_distribution.png")

# ══════════════════════════════════════════════════════════════
# PLOT 2: Churn vs Categorical Features
# ══════════════════════════════════════════════════════════════
cat_features = ["Gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "Contract", "PaperlessBilling", "PaymentMethod"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ax = axes[i]
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Churn vs {col}")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Churn", labels=["No", "Yes"], fontsize=8)
    ax.set_ylim(0, 100)

plt.suptitle("Churn Rate by Categorical Features", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eda_2_churn_vs_categorical.png", bbox_inches="tight")
plt.close()
print("Saved: eda_2_churn_vs_categorical.png")

# ══════════════════════════════════════════════════════════════
# PLOT 3: Churn vs Internet Services
# ══════════════════════════════════════════════════════════════
internet_features = ["InternetService", "OnlineSecurity", "OnlineBackup",
                     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(internet_features):
    ax = axes[i]
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Churn vs {col}")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Churn", labels=["No", "Yes"], fontsize=8)
    ax.set_ylim(0, 100)

# Hide the last unused subplot
axes[7].set_visible(False)

plt.suptitle("Churn Rate by Internet & Add-on Services", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eda_3_churn_vs_internet_services.png", bbox_inches="tight")
plt.close()
print("Saved: eda_3_churn_vs_internet_services.png")

# ══════════════════════════════════════════════════════════════
# PLOT 4: Churn vs Numerical Features (Box + KDE)
# ══════════════════════════════════════════════════════════════
num_features = ["Tenure", "MonthlyCharges", "TotalCharges"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Boxplots
for i, col in enumerate(num_features):
    ax = axes[0][i]
    sns.boxplot(x="Churn", y=col, data=df, ax=ax, palette=colors,
                width=0.4, fliersize=3)
    ax.set_title(f"{col} by Churn (Boxplot)")
    ax.set_xlabel("Churn")

# Row 2: KDE (Density) plots
for i, col in enumerate(num_features):
    ax = axes[1][i]
    for label, color in zip(["No", "Yes"], colors):
        subset = df[df["Churn"] == label][col]
        sns.kdeplot(subset, ax=ax, color=color, label=label, fill=True, alpha=0.3, linewidth=2)
    ax.set_title(f"{col} Distribution by Churn")
    ax.legend(title="Churn")
    ax.set_xlabel(col)

plt.suptitle("Churn vs Numerical Features", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eda_4_churn_vs_numerical.png", bbox_inches="tight")
plt.close()
print("Saved: eda_4_churn_vs_numerical.png")

# ══════════════════════════════════════════════════════════════
# PLOT 5: Correlation Heatmap (Cleaned Data)
# ══════════════════════════════════════════════════════════════
df_clean = pd.read_csv("TelecomCustomerChurn_cleaned.csv")

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = df_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("eda_5_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: eda_5_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# PLOT 6: Tenure Distribution by Contract Type & Churn
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data=df, x="Tenure", hue="Churn", kde=True,
             palette=colors, alpha=0.5, bins=30, ax=ax)
ax.set_title("Tenure Distribution by Churn Status", fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_6_tenure_distribution.png", bbox_inches="tight")
plt.close()
print("Saved: eda_6_tenure_distribution.png")

# ══════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM EDA")
print("=" * 60)

# Insight 1: Churn rate
churn_rate = df["Churn"].value_counts(normalize=True)["Yes"] * 100
print(f"""
1. IMBALANCED TARGET:
   - Churn rate is {churn_rate:.1f}% (1,869 out of 7,043)
   - ~73.5% customers are retained, ~26.5% churned
   - Will need to handle class imbalance during modeling
""")

# Insight 2: Contract type
monthly_churn = df[df["Contract"] == "Monthly"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
twoyear_churn = df[df["Contract"] == "Two year"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
print(f"""2. CONTRACT TYPE IS A STRONG PREDICTOR:
   - Month-to-month customers churn at {monthly_churn:.1f}%
   - Two-year contract customers churn at only {twoyear_churn:.1f}%
   - Longer contracts = significantly lower churn
""")

# Insight 3: Tenure
churned_tenure = df[df["Churn"] == "Yes"]["Tenure"].median()
retained_tenure = df[df["Churn"] == "No"]["Tenure"].median()
print(f"""3. NEW CUSTOMERS CHURN MORE:
   - Churned customers median tenure: {churned_tenure} months
   - Retained customers median tenure: {retained_tenure} months
   - Early months are the highest risk period
""")

# Insight 4: Fiber optic
fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
dsl_churn = df[df["InternetService"] == "DSL"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
print(f"""4. FIBER OPTIC USERS CHURN MORE:
   - Fiber optic churn rate: {fiber_churn:.1f}%
   - DSL churn rate: {dsl_churn:.1f}%
   - May indicate service quality issues or pricing concerns
""")

# Insight 5: Add-on services
security_no = df[df["OnlineSecurity"] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
security_yes = df[df["OnlineSecurity"] == "Yes"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
print(f"""5. ADD-ON SERVICES REDUCE CHURN:
   - Without Online Security: {security_no:.1f}% churn
   - With Online Security: {security_yes:.1f}% churn
   - TechSupport, OnlineBackup show similar patterns
   - Customers with more services are more invested & less likely to leave
""")

print("=" * 60)
print("All 6 EDA plots saved successfully!")
print("=" * 60)
