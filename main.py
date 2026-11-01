"""
Main Pipeline Runner.

Orchestrates the end-to-end Telecom Customer Churn machine learning pipeline:
1. Data Loading and Validation
2. Exploratory Data Analysis (EDA) and Visualization Generation
3. Feature Engineering and Domain Transformation
4. Multi-Model Training with SMOTE Class Imbalance Handling
5. Diagnostic Evaluation and Champion Model Serialization
6. Inference Demonstration on Customer Personas
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Ensure src directory is in Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_and_validate_data
from eda import run_all_eda
from feature_engineering import prepare_datasets
from model_training import train_models
from evaluation import run_full_evaluation
from predict import ChurnPredictor, pretty_print_prediction
from generate_data import generate_telecom_churn_data


def print_banner() -> None:
    """Print project header banner."""
    banner = """
================================================================================
    TELECOM CUSTOMER CHURN PREDICTION PIPELINE
    End-to-End Enterprise ML Pipeline | Retention Risk Engine
    Author: Peeyoosh Kangle
================================================================================
    """
    print(banner)


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end Telecom Customer Churn ML pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV dataset (defaults to data/telecom_churn.csv)."
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip Exploratory Data Analysis plot generation."
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Skip EDA and execute only feature engineering, model training, and evaluation."
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Force regeneration of synthetic telecom churn dataset before running pipeline."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for model training (default: 5)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    return parser.parse_args()


def run_pipeline() -> None:
    """Execute complete churn prediction pipeline."""
    args = parse_arguments()
    print_banner()

    pipeline_start = time.perf_counter()
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    vis_dir = os.path.join(project_root, "visualizations")
    model_dir = os.path.join(project_root, "models")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    default_data_file = os.path.join(data_dir, "telecom_churn.csv")
    data_path = args.data_path or default_data_file

    # Step 0: Dataset existence check or regeneration
    if args.generate_data or not os.path.exists(data_path):
        print("\n>>> [STEP 0/6] Generating synthetic telecom churn dataset...")
        t0 = time.perf_counter()
        generate_telecom_churn_data(
            num_records=7043,
            random_seed=args.random_state,
            output_path=data_path
        )
        print(f"    Completed in {time.perf_counter() - t0:.2f}s")

    # Step 1: Load and Validate Data
    print("\n>>> [STEP 1/6] Loading and validating telecom dataset...")
    t1 = time.perf_counter()
    df = load_and_validate_data(data_path)
    print(f"    Completed in {time.perf_counter() - t1:.2f}s")

    # Step 2: Exploratory Data Analysis (EDA)
    skip_eda = args.skip_eda or args.model_only
    if not skip_eda:
        print("\n>>> [STEP 2/6] Running Exploratory Data Analysis & generating visual charts...")
        t2 = time.perf_counter()
        eda_plots = run_all_eda(df, output_dir=vis_dir)
        print(f"    Generated {len(eda_plots)} EDA plots in {time.perf_counter() - t2:.2f}s")
    else:
        print("\n>>> [STEP 2/6] Exploratory Data Analysis skipped (--skip-eda / --model-only).")

    # Step 3: Feature Engineering & Preprocessing Pipeline
    print("\n>>> [STEP 3/6] Applying feature engineering and train-test split...")
    t3 = time.perf_counter()
    pipeline_save_path = os.path.join(model_dir, "feature_pipeline.joblib")
    X_train, X_test, y_train, y_test, pipeline = prepare_datasets(
        df,
        test_size=0.20,
        random_state=args.random_state,
        pipeline_save_path=pipeline_save_path
    )
    print(f"    Engineered {X_train.shape[1]} features in {time.perf_counter() - t3:.2f}s")

    # Step 4: Model Training with SMOTE
    print("\n>>> [STEP 4/6] Training candidate models with SMOTE class balancing & 5-fold CV...")
    t4 = time.perf_counter()
    models, cv_scores = train_models(
        X_train,
        y_train,
        random_state=args.random_state,
        cv_folds=args.cv_folds
    )
    print(f"    Trained {len(models)} models in {time.perf_counter() - t4:.2f}s")

    # Step 5: Evaluation & Model Selection
    print("\n>>> [STEP 5/6] Evaluating models, generating diagnostic ROC & confusion matrices...")
    t5 = time.perf_counter()
    model_save_path = os.path.join(model_dir, "best_model.joblib")
    comparison_df, best_model_name = run_full_evaluation(
        models=models,
        X_test=X_test,
        y_test=y_test,
        output_dir=vis_dir,
        model_save_path=model_save_path
    )
    print(f"    Completed evaluation and model serialization in {time.perf_counter() - t5:.2f}s")

    # Step 6: Inference Demonstration
    print("\n>>> [STEP 6/6] Demonstrating real-time customer churn prediction...")
    t6 = time.perf_counter()

    demo_high_risk = {
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

    demo_loyal = {
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

    predictor = ChurnPredictor(model_path=model_save_path, pipeline_path=pipeline_save_path)

    res_high = predictor.predict_single(demo_high_risk)
    pretty_print_prediction(demo_high_risk["customerID"], res_high)

    res_loyal = predictor.predict_single(demo_loyal)
    pretty_print_prediction(demo_loyal["customerID"], res_loyal)
    print(f"    Inference completed in {time.perf_counter() - t6:.2f}s")

    total_time = time.perf_counter() - pipeline_start

    # Final Summary Banner
    best_row = comparison_df.iloc[0]
    print("\n" + "=" * 80)
    print("                    PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f" * Champion Model Selected : {best_model_name}")
    print(f" * Test ROC-AUC Score      : {best_row['ROC-AUC']:.4f}")
    print(f" * Test F1-Score           : {best_row['F1-Score']:.4f}")
    print(f" * Test Accuracy           : {best_row['Accuracy']:.4f}")
    print(f" * Saved Model Path        : {model_save_path}")
    print(f" * Feature Pipeline Path   : {pipeline_save_path}")
    print(f" * Visualizations Output   : {vis_dir} ({len(os.listdir(vis_dir))} artifacts)")
    print(f" * Total Pipeline Runtime  : {total_time:.2f} seconds")
    print("=" * 80)
    print(" Pipeline execution finished successfully!\n")


if __name__ == "__main__":
    run_pipeline()
