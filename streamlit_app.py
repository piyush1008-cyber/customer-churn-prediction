"""Streamlit Interactive Web Application for Telecom Customer Churn Prediction."""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Inject src path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import ChurnPredictor

st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_predictor():
    return ChurnPredictor()


try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.title("📊 Telecom Customer Churn Prediction AI")
st.markdown(
    """
    **Production Machine Learning Automated Valuation & Churn Risk Engine.**  
    Predict customer attrition probability, risk tiers, and personalized retention actions in real-time.
    """
)

tab1, tab2 = st.tabs(["🎯 Single Customer Inference", "📁 Batch CSV Scoring"])

with tab1:
    st.subheader("Customer Profile & Contract Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown("##### 🌐 Services Subscribed")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("##### 💳 Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges), step=10.0)

    st.markdown("---")
    if st.button("🚀 Run Churn Prediction Model", type="primary", use_container_width=True):
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Analyzing customer telemetry..."):
            res = predictor.predict_single(input_data)

        res_col1, res_col2, res_col3 = st.columns(3)
        prob_pct = res["churn_probability"] * 100

        with res_col1:
            st.metric(label="Predicted Outcome", value=res["prediction"])

        with res_col2:
            st.metric(label="Churn Probability", value=f"{prob_pct:.1f}%")

        with res_col3:
            st.metric(label="Risk Tier", value=res["risk_level"])

        st.progress(float(res["churn_probability"]))

        # Actionable Recommendations
        st.subheader("💡 Strategic Retention Recommendations")
        if res["risk_level"] == "HIGH RISK":
            st.error("🚨 High probability of cancellation within 60 days.")
            st.markdown("""
            - **Offer Annual Commitment Incentive:** Transition to a 1-year contract with a 15% discount.
            - **Complimentary Value-Add:** Provide 3 months of free Premium Tech Support.
            - **Payment Method Transition:** Offer a \$5 bill credit for switching from Electronic Check to Auto-Debit.
            """)
        elif res["risk_level"] == "MEDIUM RISK":
            st.warning("⚠️ Moderate attrition risk. Proactive customer success check-in advised.")
            st.markdown("""
            - Send targeted customer satisfaction survey.
            - Provide device protection discount bundle.
            """)
        else:
            st.success("✅ Loyal customer with healthy retention telemetry.")
            st.markdown("""
            - Candidate for premium plan upsell or loyalty referral reward.
            """)

with tab2:
    st.subheader("Batch Dataset Scoring")
    st.write("Upload a CSV file containing telecom customer columns to run batch predictions.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} customer rows.")
        if st.button("Score Batch Customers", type="primary"):
            with st.spinner("Processing batch predictions..."):
                scored_df = predictor.predict_batch(batch_df)
            st.success("Scoring complete!")
            st.dataframe(scored_df.head(20))

            csv = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV Predictions",
                data=csv,
                file_name="churn_predictions_scored.csv",
                mime="text/csv",
            )
