# ═══════════════════════════════════════════════════════════════════
# FILE: 05_inference.py
# WHAT THIS FILE DOES: Loads saved model, predicts on 5 transactions.
# RUN IT WITH: python 05_inference.py
# ═══════════════════════════════════════════════════════════════════

import os
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "xgboost_fraud_model.joblib"))

print("=" * 60)
print("STEP 5 — INFERENCE")
print("=" * 60)
print(f"\n[1] Model loaded: {type(model).__name__}")


def predict_fraud(features, threshold=0.5):
    """Takes 30 feature values, returns fraud probability and label."""
    arr       = np.array(features).reshape(1, -1)
    prob      = model.predict_proba(arr)[0][1]
    is_fraud  = prob >= threshold
    if prob >= 0.8:   confidence = "HIGH RISK"
    elif prob >= 0.5: confidence = "MEDIUM RISK"
    elif prob >= 0.2: confidence = "LOW RISK"
    else:             confidence = "VERY LOW RISK"
    return {'fraud_probability': round(float(prob), 4),
            'is_fraud': bool(is_fraud),
            'confidence': confidence}


X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

fraud_idx  = np.where(y_test == 1)[0]
normal_idx = np.where(y_test == 0)[0]
np.random.seed(0)
s_fraud  = np.random.choice(fraud_idx,  3, replace=False)
s_normal = np.random.choice(normal_idx, 2, replace=False)
samples  = [s_fraud[0], s_normal[0], s_fraud[1], s_normal[1], s_fraud[2]]
labels   = [1, 0, 1, 0, 1]

print(f"\n[2] Predictions on 5 sample transactions:\n")
print(f"{'#':<4} {'True Label':<14} {'Fraud Prob':>11} {'Prediction':>14} {'Confidence':>14} {'OK?':>5}")
print("─" * 68)

correct = 0
for i, (idx, true) in enumerate(zip(samples, labels), 1):
    r    = predict_fraud(X_test[idx])
    pred = 1 if r['is_fraud'] else 0
    ok   = "✓" if pred == true else "✗"
    if pred == true: correct += 1
    tstr = "FRAUD 🚨" if true == 1 else "Normal ✅"
    pstr = "FRAUD 🚨" if r['is_fraud'] else "Normal ✅"
    print(f"{i:<4} {tstr:<14} {r['fraud_probability']:>11.4f} {pstr:>16} {r['confidence']:>14} {ok:>5}")

print("─" * 68)
print(f"\nScore: {correct}/5 correct")
print("\nPipeline complete! Check outputs/plots/ for all 8 visualisations.")
