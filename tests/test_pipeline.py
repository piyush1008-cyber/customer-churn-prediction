"""Unit tests for customer churn prediction inference and feature pipeline."""

import os
import sys
from pathlib import Path
import pytest
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import ChurnPredictor


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


def test_predictor_artifacts_loaded(predictor):
    assert predictor.model is not None
    assert predictor.pipeline is not None
    assert len(predictor.model_name) > 0


def test_single_customer_prediction(predictor):
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
    }

    result = predictor.predict_single(sample_customer)
    assert "prediction" in result
    assert result["prediction"] in ("Churn", "Retain")
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ("LOW RISK", "MEDIUM RISK", "HIGH RISK")


def test_batch_prediction_dataframe(predictor):
    df = pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 85.0,
        },
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0,
        }
    ])

    batch_res = predictor.predict_batch(df)
    assert len(batch_res) == 2
    assert "Prediction" in batch_res.columns
    assert "Churn_Probability" in batch_res.columns
    assert "Risk_Level" in batch_res.columns
