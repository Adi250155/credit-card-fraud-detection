
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Enter transaction details below to check whether "
    "a credit card transaction is legitimate or fraudulent."
)

st.divider()

st.subheader("Transaction Details")

with st.form("transaction_form"):

    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    st.subheader("Transaction Features")

    feature_values = {}

    columns = st.columns(2)

    for i, feature in enumerate(feature_columns):
        if feature == "Time":
            feature_values[feature] = time

        elif feature == "Amount":
            feature_values[feature] = amount

        else:
            with columns[i % 2]:
                feature_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=feature
                )

    submitted = st.form_submit_button(
        "Predict Transaction",
        use_container_width=True
    )

if submitted:

    input_data = pd.DataFrame(
        [[feature_values[col] for col in feature_columns]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    with col2:
        st.metric(
            "Estimated Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    st.progress(float(probability))

    st.caption(
        "Prediction generated using the trained Logistic Regression model."
    )
