# ═══════════════════════════════════════════════════════════════════
# FILE: 03_train_models.py
# WHAT THIS FILE DOES: Trains 3 models, evaluates, plots curves,
#                      saves best model (XGBoost) to disk.
# RUN IT WITH: python 03_train_models.py
# ═══════════════════════════════════════════════════════════════════

import os

# Agg backend MUST come before any other matplotlib import
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed arrays
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_sm.npy"))
X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train_sm.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

print("=" * 60)
print("STEP 3 — MODEL TRAINING & EVALUATION")
print("=" * 60)
print(f"\nTrain: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# ── Define 3 models ────────────────────────────────────────────────
pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='aucpr', random_state=42,
        scale_pos_weight=pos_weight)
}

results  = {}
roc_data = {}
pr_data  = {}

for name, model in models.items():
    print(f"\n{'─'*60}\nTraining: {name}\n{'─'*60}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=['Normal','Fraud']))
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    print(f"AUC-ROC : {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}")

    results[name]  = {'AUC-ROC': auc, 'Avg Precision': ap}
    fpr, tpr, _    = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    prec, rec, _   = precision_recall_curve(y_test, y_prob)
    pr_data[name]  = (prec, rec, ap)

# ── ROC Curve plot ─────────────────────────────────────────────────
colours = ['#3498db', '#2ecc71', '#e74c3c']
fig, ax = plt.subplots(figsize=(9, 7))
for (nm, (fpr, tpr, auc)), col in zip(roc_data.items(), colours):
    ax.plot(fpr, tpr, label=f"{nm} (AUC={auc:.4f})", color=col, linewidth=2)
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random Baseline (0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All 3 Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_roc_curves.png"), dpi=150)
plt.close()
print(f"\n[PLOT SAVED] 05_roc_curves.png")

# ── Precision-Recall Curve plot ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for (nm, (prec, rec, ap)), col in zip(pr_data.items(), colours):
    ax.plot(rec, prec, label=f"{nm} (AP={ap:.4f})", color=col, linewidth=2)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', linewidth=1,
           label=f'Baseline ({y_test.mean():.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — All 3 Models', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_precision_recall_curves.png"), dpi=150)
plt.close()
print(f"[PLOT SAVED] 06_precision_recall_curves.png")

# ── Comparison table ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\n{'Model':<25} {'AUC-ROC':>10} {'Avg Precision':>14}")
print("-" * 52)
for nm, m in results.items():
    print(f"{nm:<25} {m['AUC-ROC']:>10.4f} {m['Avg Precision']:>14.4f}")

# ── Save XGBoost model ─────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "xgboost_fraud_model.joblib")
joblib.dump(models["XGBoost"], model_path)
print(f"\n[MODEL SAVED] {model_path}")
print("\nDone. Run 04_shap_explain.py next.")
