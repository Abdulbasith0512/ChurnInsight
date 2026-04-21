"""
=============================================================
STEP 6: Machine Learning Models
=============================================================
Telecom Customer Churn Prediction
Models: Logistic Regression, Decision Tree, Random Forest
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ── Style Setup ──────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# ══════════════════════════════════════════════════════════════
# 1. LOAD ML-READY DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 6: MACHINE LEARNING MODELS")
print("=" * 65)

df = pd.read_csv("TelecomChurn_featured.csv")
print(f"\n  Dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"  Features (X) : {X.shape}")
print(f"  Target   (y) : {y.shape}")
print(f"  Class Distribution: {dict(y.value_counts())}")
print(f"  Churn Rate: {y.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 2. TRAIN-TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("  TRAIN-TEST SPLIT (80/20, stratified)")
print("-" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Training set : {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"  Testing set  : {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.0f}%)")
print(f"  Train churn  : {y_train.mean()*100:.1f}%")
print(f"  Test churn   : {y_test.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# 3. DEFINE & TRAIN MODELS
# ══════════════════════════════════════════════════════════════
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1
    ),
}

results = {}

print("\n" + "=" * 65)
print("  MODEL TRAINING & EVALUATION")
print("=" * 65)

for name, model in models.items():
    print(f"\n{'─' * 65}")
    print(f"  MODEL: {name}")
    print(f"{'─' * 65}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model": model,
    }

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], indent=4))

# ══════════════════════════════════════════════════════════════
# 4. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL COMPARISON")
print("=" * 65)

comp_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k not in ["y_pred", "y_prob", "model"]}
    for name, metrics in results.items()
}).T

comp_df = comp_df.round(4)
print(f"\n{comp_df.to_string()}")

# Best model
best_model_name = comp_df["F1-Score"].idxmax()
print(f"\n  BEST MODEL (by F1-Score): {best_model_name}")
print(f"  F1-Score: {comp_df.loc[best_model_name, 'F1-Score']:.4f}")

# ══════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

# --- Plot 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, metrics) in enumerate(results.items()):
    ax = axes[i]
    cm = confusion_matrix(y_test, metrics["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                annot_kws={"size": 14, "fontweight": "bold"})
    ax.set_title(f"{name}\nAccuracy: {metrics['Accuracy']:.2%}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.suptitle("Confusion Matrices - All Models", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("ml_1_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("\n  Saved: ml_1_confusion_matrices.png")

# --- Plot 2: ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 7))
colors_roc = ["#e74c3c", "#3498db", "#2ecc71"]

for (name, metrics), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f"{name} (AUC = {metrics['AUC-ROC']:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ml_2_roc_curves.png", bbox_inches="tight")
plt.close()
print("  Saved: ml_2_roc_curves.png")

# --- Plot 3: Metrics Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
x = np.arange(len(metrics_to_plot))
width = 0.25
colors_bar = ["#e74c3c", "#3498db", "#2ecc71"]

for i, (name, metrics) in enumerate(results.items()):
    values = [metrics[m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, values, width, label=name, color=colors_bar[i],
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("ml_3_metrics_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: ml_3_metrics_comparison.png")

# --- Plot 4: Feature Importance (Random Forest) ---
rf_model = results["Random Forest"]["model"]
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_15 = feat_importance.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_15)))
top_15.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white", linewidth=0.5)
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=12)

for i, (val, name) in enumerate(zip(top_15.values, top_15.index)):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("ml_4_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved: ml_4_feature_importance.png")

# ══════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"""
  Models Trained: 3
  Best Model    : {best_model_name}
  
  Performance Highlights:
  {'-'*50}""")

for name, metrics in results.items():
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"  {name}:")
    print(f"    Accuracy={metrics['Accuracy']:.2%}, F1={metrics['F1-Score']:.4f}, AUC={metrics['AUC-ROC']:.4f}{marker}")

print(f"""
  Key Observations:
  - class_weight='balanced' used to handle imbalanced data
  - Stratified train-test split preserves churn ratio
  - Random Forest typically gives best overall performance
  - High Recall = catches more actual churners (important for business)
  
  Plots saved: 4 visualization files
""")
print("=" * 65)
