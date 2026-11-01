"""
Model Training Module.

Orchestrates multi-model training, 5-fold cross-validation with SMOTE
class imbalance correction, and model selection.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate production candidate classification models.

    Parameters
    ----------
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to sklearn estimator instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=random_state
        )
    }
    return models


def evaluate_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Perform Stratified 5-Fold Cross Validation with SMOTE inside each fold.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of candidate models.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training binary target labels.
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Cross-validation metrics summary per model.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy"
    }

    cv_results: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 65)
    print(f" [MODEL TRAINING] 5-FOLD CROSS-VALIDATION WITH SMOTE:")
    print(f" {'Model':<25} | {'CV ROC-AUC':<14} | {'CV F1-Score':<14}")
    print("-" * 65)

    for name, estimator in models.items():
        # Chain SMOTE with estimator inside fold to prevent leakage
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", estimator)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        roc_mean = float(np.mean(scores["test_roc_auc"]))
        roc_std = float(np.std(scores["test_roc_auc"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        f1_std = float(np.std(scores["test_f1"]))
        rec_mean = float(np.mean(scores["test_recall"]))
        prec_mean = float(np.mean(scores["test_precision"]))
        acc_mean = float(np.mean(scores["test_accuracy"]))

        cv_results[name] = {
            "cv_roc_auc_mean": round(roc_mean, 4),
            "cv_roc_auc_std": round(roc_std, 4),
            "cv_f1_mean": round(f1_mean, 4),
            "cv_f1_std": round(f1_std, 4),
            "cv_recall_mean": round(rec_mean, 4),
            "cv_precision_mean": round(prec_mean, 4),
            "cv_accuracy_mean": round(acc_mean, 4),
        }

        print(f" {name:<25} | {roc_mean:.4f} ± {roc_std:.4f} | {f1_mean:.4f} ± {f1_std:.4f}")

    print("=" * 65)
    return cv_results


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    cv_folds: int = 5
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train all candidate models on SMOTE-resampled training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    random_state : int, default=42
        Reproducibility seed.
    cv_folds : int, default=5
        Number of CV folds for validation.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]
        Trained models dictionary and cross-validation performance scores.
    """
    base_models = get_base_models(random_state=random_state)

    # 1. Evaluate with cross-validation first
    cv_scores = evaluate_cross_validation(
        base_models, X_train, y_train, cv_folds=cv_folds, random_state=random_state
    )

    # 2. Resample full training set using SMOTE
    print(f"\n[ModelTraining] Applying SMOTE oversampling on full training set...")
    print(f"  Pre-SMOTE class distribution: {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"  Post-SMOTE class distribution: {dict(pd.Series(y_train_res).value_counts())}")
    print(f"  Training set expanded from {len(X_train)} to {len(X_train_res)} balanced samples.")

    # 3. Fit each model on balanced training data
    trained_models: Dict[str, Any] = {}
    print("\n[ModelTraining] Fitting models on balanced training dataset:")
    for name, model in base_models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

    print("[ModelTraining] All models trained successfully.")
    return trained_models, cv_scores


if __name__ == "__main__":
    from data_loader import load_and_validate_data
    from feature_engineering import prepare_datasets

    df = load_and_validate_data()
    X_tr, X_te, y_tr, y_te, _ = prepare_datasets(df)
    models, cv_metrics = train_models(X_tr, y_tr)
