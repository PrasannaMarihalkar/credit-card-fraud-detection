# ═══════════════════════════════════════════════════════════════════
# FILE: 02_preprocessing.py
# WHAT THIS FILE DOES: Scale, split, SMOTE, save arrays to disk.
# RUN IT WITH: python 02_preprocessing.py
# ═══════════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2 — PREPROCESSING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Scale Amount and Time (V1-V28 are already PCA-scaled)
scaler       = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']].values)
df['Time']   = scaler.fit_transform(df[['Time']].values)
print(f"\n[2] Scaled Amount and Time.")
print(f"    Amount → mean: {df['Amount'].mean():.4f}, std: {df['Amount'].std():.4f}")
print(f"    Time   → mean: {df['Time'].mean():.4f},   std: {df['Time'].std():.4f}")

X = df.drop('Class', axis=1).values
y = df['Class'].values
print(f"\n[3] X shape: {X.shape}   y shape: {y.shape}")

# stratify=y preserves the 0.17% fraud ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[4] Train/Test split (80/20 stratified):")
print(f"    Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"    Train fraud: {y_train.sum()}  Test fraud: {y_test.sum()}")

# SMOTE on training data ONLY — never on test (that would be data leakage)
print(f"\n[5] Applying SMOTE to training data only...")
print(f"    Before: Normal={( y_train==0).sum():,}  Fraud={(y_train==1).sum():,}")
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"    After:  Normal={(y_train_sm==0).sum():,}  Fraud={(y_train_sm==1).sum():,}")
print(f"    New training size: {X_train_sm.shape[0]:,}")

np.save(os.path.join(OUTPUT_DIR, "X_train_sm.npy"), X_train_sm)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),     X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train_sm.npy"), y_train_sm)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),     y_test)

print(f"\n[6] Saved 4 arrays to outputs/")
print("Preprocessing complete. Run 03_train_models.py next.")
