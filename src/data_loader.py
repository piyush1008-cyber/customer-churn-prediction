"""
Data Loading and Validation Module.

Provides robust functions to load, clean, validate, and summarize
telecom customer churn dataset.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


EXPECTED_COLUMNS: List[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

NUMERICAL_COLS: List[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS: List[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "Churn"
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load telecom churn CSV data from the specified path.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to telecom churn CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded as pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the given file path does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[DataLoader] Successfully loaded {len(df)} rows and {df.shape[1]} columns from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by handling whitespace, converting types, and handling missing values.

    Specific cleaning steps:
    1. Strip whitespace from string columns.
    2. Convert 'TotalCharges' from object/string to float, converting whitespace to NaN.
    3. For zero tenure records (where TotalCharges was blank), impute TotalCharges as 0.0.
    4. Ensure proper categorical and numerical data types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned and properly typed DataFrame.
    """
    df = df.copy()

    # 1. Clean string columns of leading/trailing whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. TotalCharges coercion: handle blank strings or spaces
    initial_null_tc = df["TotalCharges"].isin(["", " "]).sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Domain imputation: If tenure is 0, customer hasn't completed first billing cycle
    # Total billed charges are 0.0
    zero_tenure_mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    imputed_count = zero_tenure_mask.sum()
    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Any remaining NaNs imputed with monthly charges * tenure or median
    if df["TotalCharges"].isna().sum() > 0:
        remaining_nans = df["TotalCharges"].isna().sum()
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        print(f"[DataLoader] Imputed {remaining_nans} residual TotalCharges values using MonthlyCharges * tenure.")

    if initial_null_tc > 0:
        print(f"[DataLoader] Processed {initial_null_tc} blank TotalCharges entries ({imputed_count} resolved as 0.0 for tenure=0).")

    # 4. Enforce SeniorCitizen as integer
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Perform thorough validation checks on the dataset.

    Checks:
    - Presence of all expected columns
    - Absence of unexpected null values
    - Absence of duplicate customer IDs
    - Range sanity of numerical features
    - Target column valid values

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe to validate.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warning_or_error_messages)
    """
    issues: List[str] = []

    # Check 1: Missing columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check 2: Null values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Columns with null values detected: {cols_with_nulls.to_dict()}")

    # Check 3: Duplicate customer IDs
    if "customerID" in df.columns:
        duplicate_ids = df["customerID"].duplicated().sum()
        if duplicate_ids > 0:
            issues.append(f"Detected {duplicate_ids} duplicate customer IDs.")

    # Check 4: Numeric range validation
    if (df["tenure"] < 0).any():
        issues.append("Negative tenure values found.")
    if (df["MonthlyCharges"] < 0).any():
        issues.append("Negative MonthlyCharges values found.")
    if (df["TotalCharges"] < 0).any():
        issues.append("Negative TotalCharges values found.")

    # Check 5: Churn labels
    if "Churn" in df.columns:
        invalid_churn = set(df["Churn"].unique()) - {"Yes", "No", 0, 1}
        if invalid_churn:
            issues.append(f"Unexpected values in Churn column: {invalid_churn}")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_data_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate high-level metadata and statistical summary for reporting.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    Dict[str, any]
        Dictionary of key dataset metrics.
    """
    total_customers = len(df)
    churn_count = (df["Churn"] == "Yes").sum() if "Churn" in df.columns else 0
    churn_rate = churn_count / total_customers if total_customers > 0 else 0.0

    summary = {
        "total_records": total_customers,
        "total_features": df.shape[1],
        "churn_count": int(churn_count),
        "churn_rate_pct": round(churn_rate * 100, 2),
        "avg_tenure_months": round(float(df["tenure"].mean()), 2),
        "median_monthly_charges": round(float(df["MonthlyCharges"].median()), 2),
        "mean_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "total_revenue_recorded": round(float(df["TotalCharges"].sum()), 2),
    }
    return summary


def load_and_validate_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    End-to-end convenience loader: loads, cleans, and validates data.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to `../data/telecom_churn.csv` relative to module.

    Returns
    -------
    pd.DataFrame
        Ready-to-use validated DataFrame.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "telecom_churn.csv")

    raw_df = load_raw_data(filepath)
    clean_df = clean_data(raw_df)
    is_valid, issues = validate_data(clean_df)

    if not is_valid:
        print(f"[DataLoader WARNING] Validation identified issues: {issues}")
    else:
        print("[DataLoader] All data integrity and validation checks passed successfully.")

    summary = get_data_summary(clean_df)
    print(
        f"[DataLoader] Summary: {summary['total_records']} rows | "
        f"Churn Rate: {summary['churn_rate_pct']}% | "
        f"Avg Tenure: {summary['avg_tenure_months']} mos | "
        f"Mean Monthly: ${summary['mean_monthly_charges']}"
    )

    return clean_df


if __name__ == "__main__":
    df = load_and_validate_data()
    print("\nDataset Info:")
    df.info()
    print("\nFirst 3 rows:")
    print(df.head(3))
