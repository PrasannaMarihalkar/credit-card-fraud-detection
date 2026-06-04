# ═══════════════════════════════════════════════════════════════════
# FILE: 01_eda.py
# WHAT THIS FILE DOES: Loads the dataset and explores it visually.
# RUN IT WITH: python 01_eda.py
# ═══════════════════════════════════════════════════════════════════

import os

# matplotlib.use('Agg') MUST be called before ANY other matplotlib import.
# 'Agg' = Anti-Grain Geometry backend: saves plots to files without
# needing a display, a window, or Tkinter. Fixes the Tcl/Tkinter crash
# that happens with Python 3.13 on Windows.
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION — EDA")
print("=" * 60)
print(f"\n[1] Shape: {df.shape}  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\n[2] Columns:\n    {list(df.columns)}")
print(f"\n[3] Data Types:\n{df.dtypes}")

missing = df.isnull().sum()
print(f"\n[4] Missing Values: {'None — clean dataset!' if not missing.any() else missing[missing>0]}")

class_counts = df['Class'].value_counts()
fraud_pct    = (class_counts[1] / len(df)) * 100
print(f"\n[5] Class Distribution:")
print(f"    Normal: {class_counts[0]:,}   Fraud: {class_counts[1]:,}   ({fraud_pct:.4f}% fraud)")
print(f"    Ratio: 1 fraud per {int(class_counts[0]/class_counts[1])} normal transactions")
print(f"\n[6] Stats:\n{df[['Time','Amount']].describe().round(2)}")

fraud_df  = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0]

# ── Plot 1: Class Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='Class', data=df, palette=['#2ecc71','#e74c3c'], ax=ax)
ax.set_title('Class Distribution: Normal vs Fraud', fontsize=14, fontweight='bold')
ax.set_xlabel('Class  (0=Normal, 1=Fraud)')
ax.set_ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=150)
plt.close()
print(f"\n[PLOT 1 SAVED] 01_class_distribution.png")

# ── Plot 2: Correlation Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.3, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.close()
print(f"[PLOT 2 SAVED] 02_correlation_heatmap.png")

# ── Plot 3: Amount Distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(normal_df['Amount'], bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[0].set_title('Amount — Normal', fontweight='bold')
axes[0].set_xlabel('Amount (USD)')
axes[0].set_xlim(0, np.percentile(normal_df['Amount'], 99))
axes[1].hist(fraud_df['Amount'], bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_title('Amount — Fraud', fontweight='bold')
axes[1].set_xlabel('Amount (USD)')
axes[1].set_xlim(0, np.percentile(fraud_df['Amount'], 99))
plt.suptitle('Amount Distribution: Fraud vs Normal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_amount_distribution.png"), dpi=150)
plt.close()
print(f"[PLOT 3 SAVED] 03_amount_distribution.png")

# ── Plot 4: Time Distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(normal_df['Time']/3600, bins=48, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].set_title('Time — Normal', fontweight='bold')
axes[0].set_xlabel('Hours Since First Transaction')
axes[1].hist(fraud_df['Time']/3600, bins=48, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_title('Time — Fraud', fontweight='bold')
axes[1].set_xlabel('Hours Since First Transaction')
plt.suptitle('Time Distribution: Fraud vs Normal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_time_distribution.png"), dpi=150)
plt.close()
print(f"[PLOT 4 SAVED] 04_time_distribution.png")

print(f"""
KEY OBSERVATIONS:
1. Severe imbalance: {fraud_pct:.2f}% fraud. Accuracy is a useless metric here.
2. Avg fraud amount  ${fraud_df['Amount'].mean():.2f} vs normal ${normal_df['Amount'].mean():.2f}
3. V1-V28 are PCA-transformed — already scaled. Only Amount/Time need scaling.
4. Fraud is spread uniformly across time; normal dips at night.
5. No missing values — clean dataset.

EDA complete. All 4 plots saved to outputs/plots/
""")
