"""
Inference and Prediction Module.

Provides real-time single-customer scoring and high-throughput batch churn
prediction with automated risk tiering and risk-driver diagnostics.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from feature_engineering import ChurnFeaturePipeline


class ChurnPredictor:
    """
    Production-grade inference engine for telecom customer churn prediction.

    Loads champion model and feature transformation pipeline to provide
    low-latency single customer or scalable batch inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline_path: Optional[str] = None
    ) -> None:
        """
        Initialize the predictor by loading artifacts from disk.

        Parameters
        ----------
        model_path : str, optional
            Path to saved champion model joblib file.
        pipeline_path : str, optional
            Path to saved feature engineering pipeline joblib file.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(base_dir, "models", "best_model.joblib")
        self.pipeline_path = pipeline_path or os.path.join(base_dir, "models", "feature_pipeline.joblib")

        self.model_artifact: Optional[Dict[str, Any]] = None
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.pipeline: Optional[ChurnFeaturePipeline] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and feature pipeline from storage."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Champion model artifact not found at {self.model_path}. Run training and evaluation first."
            )
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(
                f"Feature pipeline artifact not found at {self.pipeline_path}. Run feature engineering first."
            )

        self.model_artifact = joblib.load(self.model_path)
        self.model = self.model_artifact["model"]
        self.model_name = self.model_artifact.get("model_name", "Champion Model")
        self.pipeline = joblib.load(self.pipeline_path)

        print(f"[ChurnPredictor] Loaded model: '{self.model_name}' and feature pipeline successfully.")

    @staticmethod
    def _categorize_risk(prob: float) -> Tuple[str, str]:
        """
        Assign business risk level and suggested retention action based on churn probability.

        Parameters
        ----------
        prob : float
            Churn probability between 0.0 and 1.0.

        Returns
        -------
        Tuple[str, str]
            (risk_level, recommended_action)
        """
        if prob >= 0.65:
            return "HIGH RISK", "Immediate proactive retention intervention (Discount / Contract Upgrade offer)"
        elif prob >= 0.35:
            return "MEDIUM RISK", "Monitor engagement, send loyalty survey, review customer service tickets"
        else:
            return "LOW RISK", "Satisfied customer; standard service nurture & cross-sell eligibility"

    @staticmethod
    def _diagnose_risk_drivers(data: Dict[str, Any]) -> List[str]:
        """Identify key contextual attributes contributing to churn risk."""
        drivers = []
        if str(data.get("Contract", "")).lower().startswith("month"):
            drivers.append("Short-term Month-to-month commitment (High Churn Correlation)")
        if float(data.get("tenure", 0)) <= 12:
            drivers.append(f"Early tenure stage ({data.get('tenure')} months, customer onboarding phase)")
        if float(data.get("MonthlyCharges", 0)) >= 75.0:
            drivers.append(f"High monthly recurring spend (${data.get('MonthlyCharges'):.2f}/mo)")
        if str(data.get("PaymentMethod", "")).lower().startswith("electronic"):
            drivers.append("Manual Electronic check payment friction")
        if str(data.get("InternetService", "")).lower() == "fiber optic":
            drivers.append("Fiber optic tier (competitive pricing sensitivity)")
        if str(data.get("TechSupport", "")).lower() == "no":
            drivers.append("Lack of dedicated Tech Support add-on")
        if str(data.get("OnlineSecurity", "")).lower() == "no":
            drivers.append("No Online Security add-on protection")

        return drivers if drivers else ["No major high-risk indicators observed"]

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform churn inference on a single customer record.

        Parameters
        ----------
        customer_data : Dict[str, Any]
            Dictionary of customer demographic and service features.

        Returns
        -------
        Dict[str, Any]
            Structured prediction payload including probability, risk level, and diagnostics.
        """
        df_single = pd.DataFrame([customer_data])
        X_trans = self.pipeline.transform(df_single)

        pred_class = int(self.model.predict(X_trans)[0])
        if hasattr(self.model, "predict_proba"):
            churn_prob = float(self.model.predict_proba(X_trans)[0, 1])
        else:
            churn_prob = 1.0 if pred_class == 1 else 0.0

        risk_level, action = self._categorize_risk(churn_prob)
        drivers = self._diagnose_risk_drivers(customer_data)

        result = {
            "prediction": "Churn" if pred_class == 1 else "Retain",
            "churn_probability": round(churn_prob, 4),
            "retention_probability": round(1.0 - churn_prob, 4),
            "risk_level": risk_level,
            "recommended_action": action,
            "key_risk_drivers": drivers,
            "model_used": self.model_name
        }
        return result

    def predict_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform high-throughput churn prediction for a batch DataFrame.

        Parameters
        ----------
        batch_df : pd.DataFrame
            Customer records with raw feature columns.

        Returns
        -------
        pd.DataFrame
            Original DataFrame augmented with Prediction, Churn_Probability, and Risk_Level columns.
        """
        df_result = batch_df.copy()
        X_trans = self.pipeline.transform(batch_df)

        preds = self.model.predict(X_trans)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[:, 1]
        else:
            probs = preds.astype(float)

        df_result["Prediction"] = np.where(preds == 1, "Churn", "Retain")
        df_result["Churn_Probability"] = np.round(probs, 4)
        df_result["Risk_Level"] = [self._categorize_risk(p)[0] for p in probs]

        return df_result


def pretty_print_prediction(customer_id: str, result: Dict[str, Any]) -> None:
    """Print an attractive summary report for a single prediction."""
    risk_color = (
        "\033[91m" if result["risk_level"] == "HIGH RISK"
        else ("\033[93m" if result["risk_level"] == "MEDIUM RISK" else "\033[92m")
    )
    reset = "\033[0m"

    print("\n" + "=" * 65)
    print(f" CUSTOMER CHURN PREDICTION REPORT: {customer_id}")
    print("=" * 65)
    print(f" Outcome Prediction   : {result['prediction'].upper()}")
    print(f" Churn Probability    : {result['churn_probability'] * 100:.2f}%")
    print(f" Retention Likelihood : {result['retention_probability'] * 100:.2f}%")
    print(f" Risk Classification  : {risk_color}{result['risk_level']}{reset}")
    print(f" Recommended Strategy : {result['recommended_action']}")
    print("-" * 65)
    print(" Key Observed Risk Drivers:")
    for driver in result["key_risk_drivers"]:
        print(f"   * {driver}")
    print("=" * 65)


def predict_single_customer(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for single customer churn scoring."""
    predictor = ChurnPredictor()
    return predictor.predict_single(customer_data)


def predict_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for batch churn scoring."""
    predictor = ChurnPredictor()
    return predictor.predict_batch(batch_df)


if __name__ == "__main__":
    # Test with representative high-risk customer profile
    high_risk_customer = {
        "customerID": "DEMO-8821-HIGH",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.50,
        "TotalCharges": 189.00
    }

    # Test with representative loyal retained customer profile
    loyal_customer = {
        "customerID": "DEMO-1044-LOYAL",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 54.20,
        "TotalCharges": 3685.60
    }

    predictor = ChurnPredictor()

    res1 = predictor.predict_single(high_risk_customer)
    pretty_print_prediction(high_risk_customer["customerID"], res1)

    res2 = predictor.predict_single(loyal_customer)
    pretty_print_prediction(loyal_customer["customerID"], res2)
