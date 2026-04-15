# 🛡️ UPI Fraud Detection System

An ML-powered fraud detection system for UPI transactions using an ensemble of **Isolation Forest** (unsupervised anomaly detection) and **XGBoost** (supervised classification), with a **SHAP explainability dashboard**.

---

## 🚀 Demo
> Add a screenshot or GIF of your Streamlit app here after running it

---

## 🧠 Tech Stack
| Component | Tool |
|---|---|
| Anomaly Detection | Isolation Forest (scikit-learn) |
| Classification | XGBoost with scale_pos_weight |
| Explainability | SHAP TreeExplainer |
| Dashboard | Streamlit + Plotly |
| Dataset | PaySim (synthetic mobile money) |

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap streamlit plotly imbalanced-learn
```

### 2. Download dataset
- Go to https://www.kaggle.com/datasets/ealaxi/paysim1
- Download and place the CSV in the project root

### 3. Train the model
```bash
python train_model.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
upi-fraud-detector/
├── train_model.py       # Feature engineering + model training
├── app.py               # Streamlit dashboard (3 pages)
├── model_artifacts.pkl  # Saved models (generated after training)
└── README.md
```

---

## 🔍 Features Engineered
- **Balance Discrepancy** — sender/receiver balance mismatch signals
- **Account Drain Detection** — sender emptied their account
- **Round Amount Flag** — unusually round amounts are a fraud signal
- **Hour of Day** — fraud spikes at odd hours
- **Amount Ratio** — transaction size relative to sender's balance
- **Isolation Forest Anomaly Score** — used as an additional feature for XGBoost

---

## 📊 Why Not Accuracy?
With a ~0.1% fraud rate, a naive model predicting "Not Fraud" for everything achieves 99.9% accuracy — completely useless. This project uses **Precision-Recall AUC** as the evaluation metric, which is the industry standard for imbalanced fraud detection.

---
