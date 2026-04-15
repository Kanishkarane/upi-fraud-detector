import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import pickle

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")  # PaySim CSV name

# ── DEBUG: print columns so you can verify ───────────────────────────────────
print("Columns in CSV:", df.columns.tolist())

# ── 2. FILTER (fraud only happens in TRANSFER & CASH_OUT) ────────────────────
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
print(f"Filtered shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# ── COLUMN NAME FIX ───────────────────────────────────────────────────────────
# PaySim uses 'newbalanceOrig' (with 'i') but the code previously used
# 'newbalanceOrg' (without 'i'). Rename to a consistent internal name.
df = df.rename(columns={
    'oldbalanceOrg':  'oldbalanceOrg',   # already correct
    'newbalanceOrig': 'newbalanceOrg',   # fix: 'Orig' → 'Org' for consistency
})

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
# Balance discrepancy — sender's balance should drop by exactly the amount
df['balanceDiscrepancyOrig'] = (df['oldbalanceOrg'] - df['newbalanceOrg']) - df['amount']

# Did receiver's balance NOT increase? (money laundering signal)
df['balanceDiscrepancyDest'] = (df['newbalanceDest'] - df['oldbalanceDest']) - df['amount']

# Hour of day (fraud spikes at odd hours)
df['hour'] = df['step'] % 24

# Is it a suspiciously round amount?
df['isRoundAmount'] = (df['amount'] % 1000 == 0).astype(int)

# Did the sender drain their account completely?
df['drainedAccount'] = (df['newbalanceOrg'] == 0).astype(int)

# Amount relative to sender's original balance
df['amountRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # +1 to avoid division by zero

# Encode transaction type
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# ── 4. DEFINE FEATURES ────────────────────────────────────────────────────────
FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrg',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiscrepancyOrig', 'balanceDiscrepancyDest',
    'hour', 'isRoundAmount', 'drainedAccount',
    'amountRatio', 'type_encoded'
]

X = df[FEATURES]
y = df['isFraud']

# ── 5. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. ISOLATION FOREST (Unsupervised Anomaly Detection) ─────────────────────
print("\nTraining Isolation Forest...")
iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_train)

# Anomaly score: more negative = more suspicious
df.loc[X_test.index, 'iso_score'] = iso.decision_function(X_test)
X_train['iso_score'] = iso.decision_function(X_train)
X_test['iso_score']  = iso.decision_function(X_test)

# ── 7. XGBOOST CLASSIFIER ─────────────────────────────────────────────────────
print("Training XGBoost...")

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.1f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',        # area under precision-recall curve
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── 8. EVALUATE ───────────────────────────────────────────────────────────────
print("\n── Classification Report ──")
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# ── 9. SHAP VALUES (save for dashboard) ──────────────────────────────────────
print("\nComputing SHAP values (sample of 500)...")
explainer = shap.TreeExplainer(xgb_model)
shap_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(shap_sample)

# ── 10. SAVE EVERYTHING ───────────────────────────────────────────────────────
print("\nSaving models...")
with open("model_artifacts.pkl", "wb") as f:
    pickle.dump({
        "xgb_model":   xgb_model,
        "iso_model":   iso,
        "explainer":   explainer,
        "features":    FEATURES + ['iso_score'],
        "shap_values": shap_values,
        "shap_sample": shap_sample
    }, f)

print("✅ Done! Run: streamlit run app.py")