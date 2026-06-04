# ═══════════════════════════════════════════════════════════════════
# FILE: 04_shap_explain.py
# WHAT THIS FILE DOES: SHAP explainability on the saved XGBoost model.
# RUN IT WITH: python 04_shap_explain.py
# ═══════════════════════════════════════════════════════════════════

import os

# Agg MUST come before any other matplotlib import
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4 — SHAP EXPLAINABILITY")
print("=" * 60)

model  = joblib.load(os.path.join(MODELS_DIR, "xgboost_fraud_model.joblib"))
X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))

# Use 500 random samples — full test set SHAP takes several minutes
np.random.seed(42)
idx      = np.random.choice(X_test.shape[0], 500, replace=False)
X_sample = X_test[idx]

feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

print(f"\n[1] Loaded model + sampled 500 test transactions.")
print(f"[2] Computing SHAP values (30-60 seconds)...")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print(f"    SHAP values shape: {shap_values.shape}")

# Top 5 features
mean_abs    = np.abs(shap_values).mean(axis=0)
top5        = np.argsort(mean_abs)[::-1][:5]
print(f"\n[3] Top 5 Features by mean |SHAP|:")
print(f"  {'Rank':<5} {'Feature':<10} {'Mean |SHAP|':>12}")
print("  " + "-" * 30)
for r, i in enumerate(top5, 1):
    print(f"  {r:<5} {feature_names[i]:<10} {mean_abs[i]:>12.4f}")

# ── Plot 1: SHAP Bar (feature importance) ─────────────────────────
print(f"\n[4] Saving SHAP bar plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                  plot_type='bar', show=False, max_display=20)
plt.title("SHAP Feature Importance", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_shap_summary_bar.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"[PLOT SAVED] 07_shap_summary_bar.png")

# ── Plot 2: SHAP Beeswarm (impact direction) ──────────────────────
print(f"[5] Saving SHAP beeswarm plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                  show=False, max_display=20)
plt.title("SHAP Beeswarm — Feature Impact Direction",
          fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_shap_beeswarm.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"[PLOT SAVED] 08_shap_beeswarm.png")

print("\nSHAP complete. Run 05_inference.py next.")
