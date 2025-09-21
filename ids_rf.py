"""
ids_rf.py
Beginner-friendly Intrusion Detection System using RandomForest on NSL-KDD.
Place KDDTrain+.TXT and KDDTest+.TXT in the same folder before running.

Requires:
  numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, imbalanced-learn
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE   # optional but helpful for imbalance

# -------------------- CONFIG --------------------
TRAIN_FILE = "KDDTrain+.TXT"
TEST_FILE  = "KDDTest+.TXT"
MODEL_OUT  = "rf_ids_model.joblib"
SCALER_OUT = "scaler.joblib"
ENC_OUT    = "label_encoders.pkl"
RANDOM_SEED = 42

# -------------------- 1. LOAD DATA --------------------
# NSL-KDD files are CSV-like, no header. We'll provide column names used commonly.
col_names = [
 "duration","protocol_type","service","flag","src_bytes","dst_bytes",
 "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
 "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
 "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
 "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
 "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
 "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
 "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

print("Loading data...")
df_train = pd.read_csv(TRAIN_FILE, names=col_names, header=None)
df_test  = pd.read_csv(TEST_FILE,  names=col_names, header=None)
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# -------------------- 2. QUICK EXPLORATION --------------------
print("\nSample labels (train):")
print(df_train['label'].value_counts().head(10))

# -------------------- 3. CONVERT TO BINARY LABEL (normal vs attack) --------------------
def to_binary_label(lbl):
    return 'normal' if lbl == 'normal' else 'attack'

df_train['binary_label'] = df_train['label'].apply(to_binary_label)
df_test['binary_label']  = df_test['label'].apply(to_binary_label)

# -------------------- 4. ENCODE CATEGORICAL FEATURES --------------------
cat_cols = ['protocol_type','service','flag']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined values to avoid unseen labels during transform
    le.fit(pd.concat([df_train[col], df_test[col]], axis=0))
    df_train[col] = le.transform(df_train[col])
    df_test[col]  = le.transform(df_test[col])
    encoders[col] = le

# -------------------- 5. FEATURES & TARGET --------------------
# We'll use all columns except original 'label' and 'binary_label' as features
feature_cols = [c for c in col_names if c != 'label']
if 'binary_label' in feature_cols:
    feature_cols.remove('binary_label')

X_train = df_train[feature_cols].copy()
y_train = df_train['binary_label'].map({'normal':0, 'attack':1})

X_test  = df_test[feature_cols].copy()
y_test  = df_test['binary_label'].map({'normal':0, 'attack':1})

print(f"\nFeature count: {X_train.shape[1]}")

# -------------------- 6. SCALE NUMERIC FEATURES --------------------
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
scaler.fit(pd.concat([X_train[num_cols], X_test[num_cols]], axis=0))
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# -------------------- 7. HANDLE CLASS IMBALANCE (optional but recommended) --------------------
print("\nOriginal class distribution (train):")
print(pd.Series(y_train).value_counts())
sm = SMOTE(random_state=RANDOM_SEED)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE (train):")
print(pd.Series(y_train_res).value_counts())

# -------------------- 8. TRAIN RandomForest --------------------
print("\nTraining RandomForest...")
rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_res, y_train_res)
print("Training complete.")

# -------------------- 9. EVALUATE --------------------
print("\nEvaluating on test set...")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['normal','attack']))

# Confusion matrix plot (save figure)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal','attack'])
fig, ax = plt.subplots(figsize=(6,4))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Test Set")
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=150)
print("Saved confusion_matrix.png")

# -------------------- 10. FEATURE IMPORTANCE --------------------
fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
fi.sort_values().plot(kind='barh')
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
print("Saved feature_importances.png")
print("\nTop features:\n", fi.head(15))

# -------------------- 11. SAVE MODEL & ARTIFACTS --------------------
joblib.dump(rf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
with open(ENC_OUT, 'wb') as f:
    pickle.dump(encoders, f)
print(f"\nSaved model to {MODEL_OUT}, scaler to {SCALER_OUT}, encoders to {ENC_OUT}")

# -------------------- 12. SAMPLE INFERENCE --------------------
# Take a random sample from test and show prediction
sample_idx = 5
sample_row = X_test.iloc[sample_idx:sample_idx+1]
pred = rf.predict(sample_row)[0]
actual = y_test.iloc[sample_idx]
print(f"\nSample inference (index {sample_idx}): predicted={'attack' if pred==1 else 'normal'}, actual={'attack' if actual==1 else 'normal'}")

# -------------------- 13. OPTIONAL: Save evaluation report --------------------
report_text = (
    f"RF IDS Report\nAccuracy: {acc:.4f}\n\nClassification Report:\n"
    + classification_report(y_test, y_pred, target_names=['normal','attack'])
)
with open("rf_eval_report.txt", "w") as f:
    f.write(report_text)
print("Saved rf_eval_report.txt")
