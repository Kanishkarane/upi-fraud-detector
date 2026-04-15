import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_models()
xgb_model   = artifacts["xgb_model"]
iso_model   = artifacts["iso_model"]
explainer   = artifacts["explainer"]
FEATURES    = artifacts["features"]
shap_values = artifacts["shap_values"]
shap_sample = artifacts["shap_sample"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ UPI Fraud Detector")
st.sidebar.markdown("AI-powered fraud detection using Isolation Forest + XGBoost ensemble")
page = st.sidebar.radio("Navigate", ["🔍 Check Transaction", "📊 Model Insights", "🎲 Simulate Feed"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TRANSACTION CHECKER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Check Transaction":
    st.title("🔍 Check a Transaction")
    st.markdown("Enter transaction details to check if it's fraudulent.")

    col1, col2 = st.columns(2)

    with col1:
        txn_type     = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount       = st.number_input("Amount (₹)", min_value=1.0, value=10000.0, step=500.0)
        hour         = st.slider("Hour of Day", 0, 23, 14)
        old_bal_orig = st.number_input("Sender's Balance Before (₹)", min_value=0.0, value=15000.0)
        new_bal_orig = st.number_input("Sender's Balance After (₹)", min_value=0.0, value=5000.0)

    with col2:
        old_bal_dest = st.number_input("Receiver's Balance Before (₹)", min_value=0.0, value=0.0)
        new_bal_dest = st.number_input("Receiver's Balance After (₹)", min_value=0.0, value=10000.0)

    if st.button("🚀 Analyze Transaction", use_container_width=True):
        # Build feature row
        type_encoded           = 1 if txn_type == "TRANSFER" else 0
        balance_disc_orig      = (old_bal_orig - new_bal_orig) - amount
        balance_disc_dest      = (new_bal_dest - old_bal_dest) - amount
        is_round               = int(amount % 1000 == 0)
        drained                = int(new_bal_orig == 0)
        amount_ratio           = amount / (old_bal_orig + 1)

        row = pd.DataFrame([{
            'amount': amount,
            'oldbalanceOrg': old_bal_orig,
            'newbalanceOrg': new_bal_orig,
            'oldbalanceDest': old_bal_dest,
            'newbalanceDest': new_bal_dest,
            'balanceDiscrepancyOrig': balance_disc_orig,
            'balanceDiscrepancyDest': balance_disc_dest,
            'hour': hour,
            'isRoundAmount': is_round,
            'drainedAccount': drained,
            'amountRatio': amount_ratio,
            'type_encoded': type_encoded
        }])

        # Isolation Forest score
        iso_score = iso_model.decision_function(row)[0]
        row['iso_score'] = iso_score

        # XGBoost prediction
        fraud_prob = xgb_model.predict_proba(row)[0][1]
        is_fraud   = fraud_prob > 0.5

        # ── RESULT ────────────────────────────────────────────────────────────
        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if is_fraud:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **TRANSACTION SAFE**")

        with res_col2:
            st.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")

        with res_col3:
            st.metric("Anomaly Score", f"{iso_score:.3f}", help="More negative = more suspicious")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if is_fraud else "green"},
                'steps': [
                    {'range': [0, 30],  'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#fff3cd"},
                    {'range': [70, 100],'color': "#f8d7da"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation for this transaction
        st.subheader("🔍 Why was this flagged?")
        shap_vals_single = explainer.shap_values(row)
        feature_names    = FEATURES
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals_single[0],
            'Raw Value': row.values[0]
        }).sort_values('SHAP Value', key=abs, ascending=True).tail(8)

        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP Value']]
        fig2 = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors
        ))
        fig2.update_layout(
            title="Feature contributions (red = pushes toward fraud)",
            xaxis_title="SHAP Value",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global SHAP Feature Importance")
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df   = pd.DataFrame({'Feature': FEATURES, 'Importance': mean_shap})
        shap_df   = shap_df.sort_values('Importance', ascending=True)

        fig = go.Figure(go.Bar(
            x=shap_df['Importance'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color='#3498db'
        ))
        fig.update_layout(title="Average |SHAP| across 500 test samples", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud by Hour of Day")
        hour_fraud = shap_sample.copy()
        hour_fraud['isFraud'] = xgb_model.predict(shap_sample)
        hourly = hour_fraud.groupby('hour')['isFraud'].mean().reset_index()

        fig2 = px.bar(hourly, x='hour', y='isFraud',
                      labels={'isFraud': 'Fraud Rate', 'hour': 'Hour of Day'},
                      color='isFraud', color_continuous_scale='RdYlGn_r')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Key Model Facts")
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "XGBoost + Isolation Forest")
    m2.metric("Imbalance Handling", "scale_pos_weight")
    m3.metric("Eval Metric", "PR-AUC (not accuracy!)")

    st.info("""
    **Why not accuracy?** With ~0.1% fraud rate, a model predicting 'Not Fraud' for everything
    achieves 99.9% accuracy — completely useless. Precision-Recall AUC is the right metric for
    imbalanced fraud detection.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIMULATE LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Simulate Feed":
    st.title("🎲 Live Transaction Feed Simulation")
    st.markdown("Simulates a stream of UPI transactions and flags suspicious ones in real-time.")

    if st.button("▶️ Generate 20 Random Transactions", use_container_width=True):
        np.random.seed(np.random.randint(0, 9999))
        n = 20

        # Generate synthetic transactions
        sim = pd.DataFrame({
            'amount':          np.random.exponential(5000, n),
            'oldbalanceOrg':   np.random.uniform(0, 50000, n),
            'newbalanceOrg':   np.random.uniform(0, 50000, n),
            'oldbalanceDest':  np.random.uniform(0, 50000, n),
            'newbalanceDest':  np.random.uniform(0, 50000, n),
            'hour':            np.random.randint(0, 24, n),
            'isRoundAmount':   np.random.randint(0, 2, n),
            'drainedAccount':  np.random.randint(0, 2, n),
            'type_encoded':    np.random.randint(0, 2, n),
        })
        sim['balanceDiscrepancyOrig'] = (sim['oldbalanceOrg'] - sim['newbalanceOrg']) - sim['amount']
        sim['balanceDiscrepancyDest'] = (sim['newbalanceDest'] - sim['oldbalanceDest']) - sim['amount']
        sim['amountRatio']            = sim['amount'] / (sim['oldbalanceOrg'] + 1)
        sim['iso_score']              = iso_model.decision_function(sim[FEATURES[:-1]])

        sim['fraud_prob'] = xgb_model.predict_proba(sim[FEATURES])[:, 1]
        sim['status']     = sim['fraud_prob'].apply(lambda p: "🚨 FRAUD" if p > 0.5 else "✅ SAFE")
        sim['amount']     = sim['amount'].round(2)
        sim['fraud_prob'] = (sim['fraud_prob'] * 100).round(1).astype(str) + "%"

        display = sim[['amount', 'hour', 'isRoundAmount', 'drainedAccount', 'fraud_prob', 'status']]
        display.columns = ['Amount (₹)', 'Hour', 'Round Amount?', 'Account Drained?', 'Fraud Risk', 'Status']

        st.dataframe(
            display.style.apply(
                lambda row: ['background-color: #ffe6e6' if '🚨' in str(row['Status'])
                             else 'background-color: #e6ffe6' for _ in row],
                axis=1
            ),
            use_container_width=True
        )
