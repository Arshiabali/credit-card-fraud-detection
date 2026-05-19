# ============================================================
# Credit Card Fraud Detection System
# Author: Arshia Bali
# Tools: Python, Pandas, Scikit-learn
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Dataset ─────
# Download from Kaggle: creditcard.csv (~144 MB)
# Place it in the same folder as this script.

print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"Normal cases : {(df['Class'] == 0).sum()}\n")

# ── 2. Pre-processing ─────────────────────────────────────────
# 'Amount' needs scaling; 'Time' is dropped (not predictive)
df = df.drop(columns=["Time"])

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

# ── 3. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. SMOTE – handle class imbalance ─────────────────────────
print("Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE – Fraud: {y_train_res.sum()}  Normal: {(y_train_res==0).sum()}\n")

# ── 5. Train Models ───────────────────────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train_res, y_train_res)

# ── 6. Evaluate ───────────────────────────────────────────────
for name, model in [("Logistic Regression", lr), ("Decision Tree", dt)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"Model : {name}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"True Positives (Fraud caught)   : {tp}")
    print(f"False Positives (Normal flagged): {fp}")
    print(f"False Negatives (Fraud missed)  : {fn}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Normal", "Fraud"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ','_')}.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix → confusion_matrix_{name.replace(' ','_')}.png")

# ── 7. Business Impact Estimate ───────────────────────────────
print("\n" + "="*50)
avg_fraud_amount = 122.21  # average from dataset
tp_rate = tp / (tp + fn)
prevented_per_10k = int(10000 * df["Class"].mean() * tp_rate)
savings_per_10k   = prevented_per_10k * avg_fraud_amount

print("Business Impact (per 10,000 transactions):")
print(f"  Expected fraud txns  : {int(10000 * df['Class'].mean())}")
print(f"  Fraud caught by model: {prevented_per_10k}")
print(f"  Estimated savings    : ${savings_per_10k:,.2f}")
print("="*50)
