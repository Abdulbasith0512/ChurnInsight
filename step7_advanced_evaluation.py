"""
=============================================================
STEP 7: Advanced Model Evaluation
=============================================================
Telecom Customer Churn Prediction
- Confusion Matrix (detailed breakdown)
- ROC Curve & AUC Analysis
- Precision-Recall Curve
- Feature Importance (multiple methods)
- Threshold Optimization
- Comprehensive Performance Report
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titleweight": "bold", "axes.titlesize": 13,
    "font.family": "sans-serif"
})

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA & RETRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 7: ADVANCED MODEL EVALUATION")
print("=" * 65)

df = pd.read_csv("TelecomChurn_featured.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Dataset     : {df.shape}")
print(f"  Train / Test: {X_train.shape[0]} / {X_test.shape[0]}")
print(f"  Churn Rate  : {y.mean()*100:.1f}%")

# Train all three models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
    }

# Identify best model by F1-Score
comp_df = pd.DataFrame({
    n: {k: v for k, v in m.items() if k not in ["y_pred", "y_prob", "model"]}
    for n, m in results.items()
}).T
best_name = comp_df["F1-Score"].idxmax()
best = results[best_name]
best_model = best["model"]

print(f"\n  Best Model (F1): {best_name}")
print(f"  F1={best['F1-Score']:.4f}  AUC={best['AUC-ROC']:.4f}")

# ══════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATION (Best Model)
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("  CROSS-VALIDATION (5-Fold) - Best Model")
print("-" * 65)

cv_scores = {}
for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
    scores = cross_val_score(best_model, X, y, cv=5, scoring=metric, n_jobs=-1)
    cv_scores[metric] = scores
    print(f"  {metric:>12s}: {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(folds: {', '.join(f'{s:.3f}' for s in scores)})")

# ══════════════════════════════════════════════════════════════
# 3. PLOT 1 — DETAILED CONFUSION MATRIX (Best Model)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  GENERATING ADVANCED VISUALIZATIONS")
print("=" * 65)

cm = confusion_matrix(y_test, best["y_pred"])
tn, fp, fn, tp = cm.ravel()
total = cm.sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            annot_kws={"size": 18, "fontweight": "bold"},
            linewidths=2, linecolor="white")
axes[0].set_title(f"Confusion Matrix - {best_name}\n(Counts)", fontsize=13)
axes[0].set_ylabel("Actual", fontsize=12)
axes[0].set_xlabel("Predicted", fontsize=12)

# Right: Percentages (normalized by true label)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
labels_pct = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
sns.heatmap(cm_pct, annot=labels_pct, fmt="", cmap="RdYlGn", ax=axes[1],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            annot_kws={"size": 16, "fontweight": "bold"},
            linewidths=2, linecolor="white", vmin=0, vmax=100)
axes[1].set_title(f"Confusion Matrix - {best_name}\n(Row-Normalized %)", fontsize=13)
axes[1].set_ylabel("Actual", fontsize=12)
axes[1].set_xlabel("Predicted", fontsize=12)

plt.suptitle("CONFUSION MATRIX ANALYSIS", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("eval_1_confusion_matrix_detailed.png", bbox_inches="tight")
plt.close()
print("\n  [OK] Saved: eval_1_confusion_matrix_detailed.png")

# Print detailed breakdown
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
print(f"\n  Confusion Matrix Breakdown:")
print(f"  {'-'*45}")
print(f"  True Negatives  (TN): {tn:>5d}  (correctly predicted No Churn)")
print(f"  True Positives  (TP): {tp:>5d}  (correctly predicted Churn)")
print(f"  False Positives (FP): {fp:>5d}  (wrongly predicted Churn)")
print(f"  False Negatives (FN): {fn:>5d}  (missed Churners)")
print(f"  {'─'*45}")
print(f"  Sensitivity/Recall : {best['Recall']:.4f}  (TP / (TP+FN))")
print(f"  Specificity        : {specificity:.4f}  (TN / (TN+FP))")
print(f"  Precision (PPV)    : {best['Precision']:.4f}  (TP / (TP+FP))")
print(f"  Neg Pred Value     : {npv:.4f}  (TN / (TN+FN))")

# ══════════════════════════════════════════════════════════════
# 4. PLOT 2 — ROC CURVE (All Models + Best Highlighted)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: All models ROC
colors = {"Logistic Regression": "#e74c3c",
          "Decision Tree": "#f39c12",
          "Random Forest": "#27ae60"}

for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
    lw = 3.5 if name == best_name else 1.8
    ls = "-" if name == best_name else "--"
    label = f"{name} (AUC={m['AUC-ROC']:.4f})"
    if name == best_name:
        label += " * BEST"
    axes[0].plot(fpr, tpr, color=colors[name], lw=lw, ls=ls, label=label)

axes[0].plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random (AUC=0.50)")
axes[0].fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
axes[0].set_xlabel("False Positive Rate", fontsize=12)
axes[0].set_ylabel("True Positive Rate", fontsize=12)
axes[0].set_title("ROC Curves - All Models", fontsize=13)
axes[0].legend(fontsize=9, loc="lower right")
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.05])
axes[0].grid(True, alpha=0.3)

# Right: Precision-Recall Curve (best model)
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
    y_test, best["y_prob"])
avg_prec = average_precision_score(y_test, best["y_prob"])

axes[1].plot(recall_vals, precision_vals, color=colors.get(best_name, "#3498db"),
             lw=2.5, label=f"{best_name} (AP={avg_prec:.4f})")
axes[1].axhline(y=y_test.mean(), color="gray", ls="--", lw=1,
                label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_xlabel("Recall", fontsize=12)
axes[1].set_ylabel("Precision", fontsize=12)
axes[1].set_title("Precision-Recall Curve - Best Model", fontsize=13)
axes[1].legend(fontsize=10, loc="upper right")
axes[1].set_xlim([-0.02, 1.05])
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3)

plt.suptitle("ROC & PRECISION-RECALL ANALYSIS", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eval_2_roc_pr_curves.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eval_2_roc_pr_curves.png")

# ══════════════════════════════════════════════════════════════
# 5. PLOT 3 — FEATURE IMPORTANCE (3 Methods)
# ══════════════════════════════════════════════════════════════

# Method 1: Built-in feature importance (tree-based)
if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.Series(best_model.feature_importances_,
                         index=X.columns).sort_values(ascending=False)
else:
    feat_imp = pd.Series(np.abs(best_model.coef_[0]),
                         index=X.columns).sort_values(ascending=False)

# Method 2: Permutation importance
print("\n  Computing permutation importance (may take a moment)...")
perm_result = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm_result.importances_mean,
                     index=X.columns).sort_values(ascending=False)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
top_n = 15

# Left: Built-in importance
top_builtin = feat_imp.head(top_n).sort_values()
colors_bar = plt.cm.viridis(np.linspace(0.25, 0.85, top_n))
top_builtin.plot(kind="barh", ax=axes[0], color=colors_bar, edgecolor="white", lw=0.5)
axes[0].set_title(f"Built-in Feature Importance\n({best_name})", fontsize=13)
axes[0].set_xlabel("Importance Score", fontsize=11)
for i, (val, nm) in enumerate(zip(top_builtin.values, top_builtin.index)):
    axes[0].text(val + feat_imp.max() * 0.01, i, f"{val:.4f}",
                 va="center", fontsize=8.5, fontweight="bold")

# Right: Permutation importance
top_perm = perm_imp.head(top_n).sort_values()
colors_bar2 = plt.cm.magma(np.linspace(0.25, 0.75, top_n))
top_perm.plot(kind="barh", ax=axes[1], color=colors_bar2, edgecolor="white", lw=0.5)
axes[1].set_title(f"Permutation Importance\n({best_name})", fontsize=13)
axes[1].set_xlabel("Mean Accuracy Decrease", fontsize=11)
for i, (val, nm) in enumerate(zip(top_perm.values, top_perm.index)):
    axes[1].text(val + perm_imp.max() * 0.01, i, f"{val:.4f}",
                 va="center", fontsize=8.5, fontweight="bold")

plt.suptitle(f"TOP {top_n} FEATURE IMPORTANCE ANALYSIS",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eval_3_feature_importance.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eval_3_feature_importance.png")

# ══════════════════════════════════════════════════════════════
# 6. PLOT 4 — THRESHOLD ANALYSIS
# ══════════════════════════════════════════════════════════════
fpr_b, tpr_b, roc_thresh = roc_curve(y_test, best["y_prob"])

thresholds = np.arange(0.1, 0.91, 0.05)
thresh_results = []
for t in thresholds:
    y_t = (best["y_prob"] >= t).astype(int)
    if y_t.sum() == 0 or y_t.sum() == len(y_t):
        continue
    thresh_results.append({
        "Threshold": t,
        "Precision": precision_score(y_test, y_t, zero_division=0),
        "Recall": recall_score(y_test, y_t, zero_division=0),
        "F1": f1_score(y_test, y_t, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_t),
    })
thresh_df = pd.DataFrame(thresh_results)

fig, ax = plt.subplots(figsize=(12, 6))
for metric, color, ls in [("Precision", "#e74c3c", "-"), ("Recall", "#3498db", "-"),
                           ("F1", "#2ecc71", "-"), ("Accuracy", "#9b59b6", "--")]:
    ax.plot(thresh_df["Threshold"], thresh_df[metric],
            color=color, lw=2.5, ls=ls, label=metric, marker="o", markersize=4)

opt_idx = thresh_df["F1"].idxmax()
opt_t = thresh_df.loc[opt_idx, "Threshold"]
opt_f1 = thresh_df.loc[opt_idx, "F1"]
ax.axvline(x=opt_t, color="gray", ls=":", lw=1.5, alpha=0.7)
ax.annotate(f"Optimal Threshold\n= {opt_t:.2f} (F1={opt_f1:.3f})",
            xy=(opt_t, opt_f1), xytext=(opt_t + 0.08, opt_f1 - 0.08),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.axvline(x=0.5, color="orange", ls="--", lw=1, alpha=0.5, label="Default (0.5)")

ax.set_xlabel("Classification Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Threshold Optimization - {best_name}", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="center left")
ax.set_xlim([0.08, 0.92])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eval_4_threshold_analysis.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eval_4_threshold_analysis.png")

# ══════════════════════════════════════════════════════════════
# 7. PLOT 5 — CLASS-WISE PERFORMANCE RADAR
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
bar_colors = ["#e74c3c", "#f39c12", "#27ae60"]
x = np.arange(len(metric_names))
w = 0.25

for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metric_names]
    edge = "gold" if name == best_name else "white"
    lw_edge = 2.5 if name == best_name else 0.5
    bars = ax.bar(x + i * w, vals, w, label=name, color=bar_colors[i],
                  edgecolor=edge, linewidth=lw_edge)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + w)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison (* = Best)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("eval_5_model_comparison.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eval_5_model_comparison.png")

# ══════════════════════════════════════════════════════════════
# 8. COMPREHENSIVE REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  COMPREHENSIVE EVALUATION REPORT")
print("=" * 65)

print(f"""
  +----------------------------------------------------------+
  |  BEST MODEL: {best_name:<43s}|
  +----------------------------------------------------------+

  > TEST SET PERFORMANCE:
    Accuracy   : {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)
    Precision  : {best['Precision']:.4f}
    Recall     : {best['Recall']:.4f}
    F1-Score   : {best['F1-Score']:.4f}
    AUC-ROC    : {best['AUC-ROC']:.4f}
    Avg Prec   : {avg_prec:.4f}

  > CROSS-VALIDATION (5-Fold):
    CV F1      : {cv_scores['f1'].mean():.4f} +/- {cv_scores['f1'].std():.4f}
    CV AUC     : {cv_scores['roc_auc'].mean():.4f} +/- {cv_scores['roc_auc'].std():.4f}

  > CONFUSION MATRIX:
    True Positives  : {tp:>5d}  (churners caught)
    False Negatives : {fn:>5d}  (churners missed)
    True Negatives  : {tn:>5d}  (non-churners correct)
    False Positives : {fp:>5d}  (false alarms)

  > THRESHOLD OPTIMIZATION:
    Default threshold (0.50) -> F1 = {best['F1-Score']:.4f}
    Optimal threshold ({opt_t:.2f})  -> F1 = {opt_f1:.4f}

  > TOP 5 MOST IMPORTANT FEATURES:""")

for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
    perm_val = perm_imp.get(feat, 0)
    print(f"    {i}. {feat:<30s}  Built-in: {imp:.4f}  Perm: {perm_val:.4f}")

print(f"""
  > BUSINESS INTERPRETATION:
    - Recall of {best['Recall']:.2%} means we catch {best['Recall']*100:.0f}% of actual churners
    - Precision of {best['Precision']:.2%} means {best['Precision']*100:.0f}% of churn predictions are correct
    - {fp} customers would receive unnecessary retention offers (false alarms)
    - {fn} churners would be missed without intervention
    
  > RECOMMENDATIONS:
    - If minimizing missed churners is priority -> lower threshold
    - If minimizing false alarms is priority -> raise threshold
    - Optimal F1 threshold: {opt_t:.2f}

  Visualizations saved: 5 files (eval_1 through eval_5)
""")
print("=" * 65)
print("  STEP 7 COMPLETE - Advanced Evaluation Done!")
print("=" * 65)
