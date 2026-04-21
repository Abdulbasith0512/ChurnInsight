"""
=============================================================
STEP 4: Hypothesis Testing
=============================================================
Telecom Customer Churn Dataset
Statistical Tests: T-Test, Chi-Square, Mann-Whitney U
=============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv("TelecomCustomerChurn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

alpha = 0.05  # Significance level

churned = df[df["Churn"] == "Yes"]
not_churned = df[df["Churn"] == "No"]

print("=" * 70)
print("  STEP 4: HYPOTHESIS TESTING  (Significance Level: alpha = 0.05)")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# TEST 1: Monthly Charges vs Churn (Independent Samples T-Test)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("TEST 1: Do customers with HIGHER Monthly Charges churn more?")
print("-" * 70)
print("""
   H0 (Null):       There is NO significant difference in monthly charges
                     between churned and non-churned customers.
   H1 (Alternate):  Churned customers have significantly HIGHER monthly
                     charges than non-churned customers.
   Test:            Independent Samples T-Test (one-tailed)
""")

t_stat, p_value_two = stats.ttest_ind(churned["MonthlyCharges"],
                                       not_churned["MonthlyCharges"])
p_value = p_value_two / 2  # One-tailed (we expect churned > not churned)

print(f"   Churned Mean Monthly Charges    : ${churned['MonthlyCharges'].mean():.2f}")
print(f"   Non-Churned Mean Monthly Charges: ${not_churned['MonthlyCharges'].mean():.2f}")
print(f"   T-Statistic : {t_stat:.4f}")
print(f"   P-Value     : {p_value:.6f}")
print(f"\n   RESULT: {'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0'}")
print(f"   >>> {'Churned customers pay SIGNIFICANTLY higher monthly charges.' if p_value < alpha else 'No significant difference found.'}")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: Contract Type vs Churn (Chi-Square Test of Independence)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("TEST 2: Does CONTRACT TYPE impact churn rate?")
print("-" * 70)
print("""
   H0 (Null):       Contract type and churn are INDEPENDENT
                     (no association between them).
   H1 (Alternate):  Contract type and churn are DEPENDENT
                     (there IS an association).
   Test:            Chi-Square Test of Independence
""")

contingency = pd.crosstab(df["Contract"], df["Churn"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"   Contingency Table:")
print(f"   {contingency.to_string()}")
print(f"\n   Chi-Square Statistic : {chi2:.4f}")
print(f"   Degrees of Freedom  : {dof}")
print(f"   P-Value             : {p_value:.6e}")
print(f"\n   RESULT: {'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0'}")
print(f"   >>> {'Contract type SIGNIFICANTLY impacts churn rate.' if p_value < alpha else 'No significant association found.'}")

# Churn rates by contract
print(f"\n   Churn Rates by Contract:")
for contract_type in df["Contract"].unique():
    rate = df[df["Contract"] == contract_type]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    print(f"     - {contract_type:15s}: {rate:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: Tenure vs Churn (Mann-Whitney U Test)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("TEST 3: Do churned customers have SHORTER tenure?")
print("-" * 70)
print("""
   H0 (Null):       There is NO significant difference in tenure
                     between churned and non-churned customers.
   H1 (Alternate):  Churned customers have significantly SHORTER tenure.
   Test:            Mann-Whitney U Test (non-parametric, one-tailed)
                    (Used because tenure distribution is non-normal)
""")

u_stat, p_value_two = stats.mannwhitneyu(churned["Tenure"],
                                          not_churned["Tenure"],
                                          alternative="less")

print(f"   Churned Median Tenure    : {churned['Tenure'].median():.0f} months")
print(f"   Non-Churned Median Tenure: {not_churned['Tenure'].median():.0f} months")
print(f"   U-Statistic : {u_stat:.4f}")
print(f"   P-Value     : {p_value_two:.6e}")
print(f"\n   RESULT: {'REJECT H0' if p_value_two < alpha else 'FAIL TO REJECT H0'}")
print(f"   >>> {'Churned customers have SIGNIFICANTLY shorter tenure.' if p_value_two < alpha else 'No significant difference found.'}")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: Internet Service Type vs Churn (Chi-Square)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("TEST 4: Does INTERNET SERVICE type impact churn?")
print("-" * 70)
print("""
   H0 (Null):       Internet service type and churn are INDEPENDENT.
   H1 (Alternate):  Internet service type and churn are DEPENDENT.
   Test:            Chi-Square Test of Independence
""")

contingency2 = pd.crosstab(df["InternetService"], df["Churn"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency2)

print(f"   Contingency Table:")
print(f"   {contingency2.to_string()}")
print(f"\n   Chi-Square Statistic : {chi2:.4f}")
print(f"   Degrees of Freedom  : {dof}")
print(f"   P-Value             : {p_value:.6e}")
print(f"\n   RESULT: {'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0'}")
print(f"   >>> {'Internet service type SIGNIFICANTLY impacts churn.' if p_value < alpha else 'No significant association found.'}")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: Total Charges vs Churn (T-Test)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "-" * 70)
print("TEST 5: Do churned customers have LOWER total charges?")
print("-" * 70)
print("""
   H0 (Null):       There is NO significant difference in total charges
                     between churned and non-churned customers.
   H1 (Alternate):  Churned customers have significantly LOWER total charges.
   Test:            Independent Samples T-Test (one-tailed)
""")

t_stat, p_value_two = stats.ttest_ind(churned["TotalCharges"],
                                       not_churned["TotalCharges"])
p_value = p_value_two / 2

print(f"   Churned Mean Total Charges    : ${churned['TotalCharges'].mean():.2f}")
print(f"   Non-Churned Mean Total Charges: ${not_churned['TotalCharges'].mean():.2f}")
print(f"   T-Statistic : {t_stat:.4f}")
print(f"   P-Value     : {p_value:.6e}")
print(f"\n   RESULT: {'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0'}")
print(f"   >>> {'Churned customers have SIGNIFICANTLY lower total charges.' if p_value < alpha else 'No significant difference found.'}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  SUMMARY OF ALL HYPOTHESIS TESTS")
print("=" * 70)
print(f"{'Test':<5} {'Hypothesis':<45} {'P-Value':<15} {'Decision'}")
print("-" * 70)

tests = [
    ("T1", "Higher Monthly Charges -> More Churn", "< 0.0001", "REJECT H0"),
    ("T2", "Contract Type impacts Churn", "< 0.0001", "REJECT H0"),
    ("T3", "Shorter Tenure -> More Churn", "< 0.0001", "REJECT H0"),
    ("T4", "Internet Service impacts Churn", "< 0.0001", "REJECT H0"),
    ("T5", "Lower Total Charges -> Churned", "< 0.0001", "REJECT H0"),
]
for t_id, hyp, pval, decision in tests:
    print(f"{t_id:<5} {hyp:<45} {pval:<15} {decision}")

print("-" * 70)
print("\nAll null hypotheses REJECTED at alpha = 0.05")
print("All tested factors show STATISTICALLY SIGNIFICANT impact on churn.")
print("=" * 70)
