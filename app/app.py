
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
DATA_PATH = ROOT / "data" / "transactions.csv"

THRESHOLD = 0.40

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection System")
st.caption("Machine learning project | Educational demonstration")

st.warning(
    "This app uses a model trained on synthetic data. "
    "It is for educational purposes only and must not be used "
    "to approve, decline, or block real financial transactions."
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run src/train.py first to train and save the model.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

model = load_model(str(MODEL_PATH))
df = load_dataset(str(DATA_PATH))

# Use the dataset schema to keep model inputs consistent.
feature_columns = [
    column for column in df.columns
    if column not in ["transaction_id", "is_fraud"]
]

st.subheader("Enter transaction details")

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=1500.0,
            step=100.0,
        )
        hour = st.slider("Hour of day", 0, 23, 14)
        day_of_week = st.selectbox(
            "Day of week (0 = Monday)",
            list(range(7)),
        )
        channel = st.selectbox(
            "Channel",
            sorted(df["channel"].dropna().unique().tolist()),
        )

    with col2:
        merchant_category = st.selectbox(
            "Merchant category",
            sorted(
                df["merchant_category"].dropna().unique().tolist()
            ),
        )
        distance_from_home_km = st.number_input(
            "Distance from home (km)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )
        transactions_last_24h = st.number_input(
            "Transactions in last 24 hours",
            min_value=0,
            value=2,
            step=1,
        )
        account_age_days = st.number_input(
            "Account age (days)",
            min_value=0,
            value=365,
            step=30,
        )

    with col3:
        card_present = st.selectbox(
            "Card present?",
            ["Yes", "No"],
        )
        international_transaction = st.selectbox(
            "International transaction?",
            ["No", "Yes"],
        )
        device_mismatch = st.selectbox(
            "Device mismatch?",
            ["No", "Yes"],
        )

    submitted = st.form_submit_button(
        "Analyze transaction",
        type="primary",
        use_container_width=True,
    )

if submitted:
    transaction = {
        "amount": float(amount),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "channel": channel,
        "merchant_category": merchant_category,
        "distance_from_home_km": float(distance_from_home_km),
        "transactions_last_24h": int(transactions_last_24h),
        "card_present": 1 if card_present == "Yes" else 0,
        "account_age_days": int(account_age_days),
        "international_transaction": (
            1 if international_transaction == "Yes" else 0
        ),
        "device_mismatch": 1 if device_mismatch == "Yes" else 0,
    }

    # Enforce the same feature order used by the dataset.
    input_df = pd.DataFrame(
        [transaction],
        columns=feature_columns,
    )

    try:
        fraud_probability = model.predict_proba(input_df)[0, 1]
        prediction = int(fraud_probability >= THRESHOLD)

        st.subheader("Analysis result")

        metric1, metric2 = st.columns(2)
        metric1.metric(
            "Estimated fraud risk",
            f"{fraud_probability:.1%}",
        )
        metric2.metric(
            "Decision threshold",
            f"{THRESHOLD:.0%}",
        )

        if prediction == 1:
            st.error(
                "The model flagged this transaction as potentially "
                "fraudulent."
            )
        else:
            st.success(
                "The model did not flag this transaction at the "
                "selected threshold."
            )

        st.progress(float(fraud_probability))

        st.caption(
            "This is a model score, not a verified fraud probability. "
            "Predictions may be unreliable because the model uses "
            "synthetic educational data."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the saved model was trained with the same "
            "feature columns and data types."
        )

st.divider()
st.caption("Portfolio project — synthetic transaction data only.")