"""
Feature Engineering Module.

Handles domain-specific feature creation, categorical transformations,
numerical feature scaling, and train-test splitting for churn modeling.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class ChurnFeaturePipeline:
    """
    Production-grade feature transformation pipeline for telecom churn prediction.

    Applies domain feature engineering, categorical encoding, and standard scaling.
    Persists scaler and column signatures to allow seamless single-record or batch inference.
    """

    ONLINE_SERVICES: List[str] = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"
    ]

    ALL_SERVICES: List[str] = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    NUMERICAL_FEATURES: List[str] = [
        "tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"
    ]

    BINARY_COLS: Dict[str, str] = {
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "PaperlessBilling": "Yes",
    }

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.categorical_categories: Dict[str, List[Any]] = {}
        self.is_fitted: bool = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-engineered features from raw attributes."""
        df = df.copy()

        # 1. AvgMonthlySpend = TotalCharges / tenure (safe division for tenure=0)
        safe_tenure = df["tenure"].replace(0, 1)
        df["AvgMonthlySpend"] = (df["TotalCharges"] / safe_tenure).round(2)
        # For zero tenure, AvgMonthlySpend equals MonthlyCharges
        df.loc[df["tenure"] == 0, "AvgMonthlySpend"] = df.loc[df["tenure"] == 0, "MonthlyCharges"]

        # 2. HasOnlineServices = 1 if customer subscribed to any cyber/device security
        online_mask = pd.Series(0, index=df.index)
        for col in self.ONLINE_SERVICES:
            if col in df.columns:
                online_mask = online_mask | (df[col] == "Yes").astype(int)
        df["HasOnlineServices"] = online_mask

        # 3. NumServices = Count of total telecom services subscribed
        service_count = pd.Series(0, index=df.index)
        for col in self.ALL_SERVICES:
            if col in df.columns:
                if col == "InternetService":
                    service_count += (df[col] != "No").astype(int)
                elif col == "MultipleLines":
                    service_count += (df[col] == "Yes").astype(int)
                else:
                    service_count += (df[col] == "Yes").astype(int)
        df["NumServices"] = service_count

        # 4. tenure_group = Binned tenure into life-cycle cohorts
        bins = [-1, 12, 24, 48, 60, 120]
        labels = ["0-12", "12-24", "24-48", "48-60", "60+"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "Churn"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the transformer on training dataframe and return transformed X and y.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned raw dataframe.
        target_col : str, default='Churn'
            Target column name.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Transformed feature matrix X and target vector y.
        """
        df_feat = self._engineer_features(df)

        # Extract target if present
        y = None
        if target_col in df_feat.columns:
            y = (df_feat[target_col] == "Yes").astype(int)

        # Drop ID and target
        drop_cols = ["customerID", target_col]
        X_raw = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

        # Binary mapping
        for col, positive_val in self.BINARY_COLS.items():
            if col in X_raw.columns:
                X_raw[col] = (X_raw[col] == positive_val).astype(int)

        # One-hot encoding for multi-category variables
        cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_categories = {}
        for col in cat_cols:
            cats = [x for x in X_raw[col].dropna().unique().tolist()]
            self.categorical_categories[col] = cats
            X_raw[col] = pd.Categorical(X_raw[col], categories=cats)

        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=int)

        # Fit Scaler on numerical features
        num_cols = [c for c in self.NUMERICAL_FEATURES if c in X_encoded.columns]
        X_encoded[num_cols] = self.scaler.fit_transform(X_encoded[num_cols])

        # Store feature schema
        self.feature_columns = X_encoded.columns.tolist()
        self.is_fitted = True

        print(f"[FeatureEngineering] Pipeline fitted. Total engineered features: {len(self.feature_columns)}")
        return X_encoded, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new unseen customer data using the fitted pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            New data with matching input schema.

        Returns
        -------
        pd.DataFrame
            Transformed features aligned exactly with model input schema.
        """
        if not self.is_fitted or self.feature_columns is None:
            raise RuntimeError("Pipeline must be fitted before transforming new data.")

        df_feat = self._engineer_features(df)
        drop_cols = ["customerID", "Churn"]
        X_raw = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors="ignore")

        # Binary mapping
        for col, positive_val in self.BINARY_COLS.items():
            if col in X_raw.columns:
                X_raw[col] = (X_raw[col] == positive_val).astype(int)

        # One-hot encode using stored categories
        cat_cols = [c for c in self.categorical_categories if c in X_raw.columns]
        for col in cat_cols:
            X_raw[col] = pd.Categorical(X_raw[col], categories=self.categorical_categories[col])
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=int)

        # Align with training features (add missing as 0, drop extraneous)
        X_aligned = X_encoded.reindex(columns=self.feature_columns, fill_value=0)

        # Scale numerical features
        num_cols = [c for c in self.NUMERICAL_FEATURES if c in X_aligned.columns]
        X_aligned[num_cols] = self.scaler.transform(X_aligned[num_cols])

        return X_aligned

    def save(self, filepath: str) -> None:
        """Serialize pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[FeatureEngineering] Pipeline persisted to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ChurnFeaturePipeline":
        """Load serialized pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        return joblib.load(filepath)


def prepare_datasets(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    pipeline_save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ChurnFeaturePipeline]:
    """
    Orchestrate train-test split and feature transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    test_size : float, default=0.20
        Proportion of dataset for testing.
    random_state : int, default=42
        Reproducibility seed.
    pipeline_save_path : str, optional
        Path to save the fitted pipeline artifact.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ChurnFeaturePipeline]
        X_train, X_test, y_train, y_test, pipeline
    """
    # Split raw data first to prevent data leakage during scaling/encoding
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Churn"]
    )

    pipeline = ChurnFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df, target_col="Churn")
    X_test = pipeline.transform(test_df)
    y_test = (test_df["Churn"] == "Yes").astype(int)

    if pipeline_save_path:
        pipeline.save(pipeline_save_path)

    print("\n" + "=" * 50)
    print(" [FEATURE ENGINEERING] SPLIT & SHAPE SUMMARY:")
    print(f"   X_train: {X_train.shape} | y_train Churn Rate: {y_train.mean():.2%}")
    print(f"   X_test : {X_test.shape} | y_test Churn Rate : {y_test.mean():.2%}")
    print(f"   Engineered Feature Count: {X_train.shape[1]}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    from data_loader import load_and_validate_data
    df = load_and_validate_data()
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    pipe_path = os.path.join(model_dir, "feature_pipeline.joblib")
    X_tr, X_te, y_tr, y_te, pipe = prepare_datasets(df, pipeline_save_path=pipe_path)
    print("\nFeature names sample:")
    print(X_tr.columns[:10].tolist())
