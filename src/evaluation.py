"""
Model Evaluation and Diagnostic Module.

Computes comprehensive classification metrics, generates ROC curves,
confusion matrices, feature importance plots, and serializes the best performing model.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import joblib


def set_visual_style() -> None:
    """Configure matplotlib styling consistent with EDA dark theme."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("dark_background")

    plt.rcParams.update({
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#C9D1D9",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "text.color": "#F0F6FC",
        "grid.color": "#21262D",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
    })


def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluate all models on test dataset and compile metrics.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of fitted models.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test ground truth labels.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        Summary metrics DataFrame and detailed metrics per model.
    """
    metrics_records: List[Dict[str, Any]] = []
    detailed_reports: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 80)
    print(" [MODEL EVALUATION] PERFORMANCE BENCHMARK (TEST SET - UNSEEN DATA):")
    print("=" * 80)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Retained (No)", "Churned (Yes)"])

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        detailed_reports[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": cr,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("Classification Report:")
        print(cr)

    comparison_df = pd.DataFrame(metrics_records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(" [LEADERBOARD] MODEL COMPARISON SUMMARY:")
    print(comparison_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format,
        "ROC-AUC": "{:.4f}".format,
    }))
    print("=" * 80)

    return comparison_df, detailed_reports


def plot_roc_curves(
    detailed_reports: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    output_dir: str
) -> str:
    """
    Plot combined ROC curves for all evaluated models on a single figure.

    Parameters
    ----------
    detailed_reports : Dict[str, Dict[str, Any]]
        Dictionary containing prediction probabilities.
    y_test : pd.Series
        Test true labels.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG.
    """
    set_visual_style()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    palette = {
        "Gradient Boosting": "#00D2BE",
        "Logistic Regression": "#FF5252",
        "Random Forest": "#FFA726",
        "Support Vector Machine": "#AB47BC",
    }

    # Reference baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6E7681", label="Random Classifier (AUC = 0.5000)", linewidth=1.5)

    for name, report in detailed_reports.items():
        fpr, tpr, _ = roc_curve(y_test, report["y_proba"])
        auc = report["roc_auc"]
        color = palette.get(name, "#58A6FF")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2.5)

    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(frameon=True, facecolor="#161B22", edgecolor="#30363D", loc="lower right", fontsize=11)

    out_path = os.path.join(output_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluation] ROC curves saved to {out_path}")
    return out_path


def plot_confusion_matrices(
    detailed_reports: Dict[str, Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Plot a 2x2 grid of confusion matrices for candidate models.

    Parameters
    ----------
    detailed_reports : Dict[str, Dict[str, Any]]
        Dictionary containing confusion matrices.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG.
    """
    set_visual_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    axes = axes.flatten()

    model_names = list(detailed_reports.keys())
    cm_palette = sns.light_palette("#00D2BE", as_cmap=True)

    for i, name in enumerate(model_names):
        ax = axes[i]
        cm = detailed_reports[name]["confusion_matrix"]
        total = cm.sum()
        pcts = cm / total * 100

        # Annotations with counts and percentages
        annot = np.array([
            [f"{cm[0, 0]:,d}\n({pcts[0, 0]:.1f}%)\nTrue Retained", f"{cm[0, 1]:,d}\n({pcts[0, 1]:.1f}%)\nFalse Churn"],
            [f"{cm[1, 0]:,d}\n({pcts[1, 0]:.1f}%)\nFalse Retained", f"{cm[1, 1]:,d}\n({pcts[1, 1]:.1f}%)\nTrue Churn"]
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cm_palette,
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor="#30363D",
            annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#0E1117"}
        )

        ax.set_title(f"{name}\nAcc: {detailed_reports[name]['accuracy']:.3f} | F1: {detailed_reports[name]['f1_score']:.3f}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticklabels(["Retained (0)", "Churned (1)"], fontsize=10)
        ax.set_yticklabels(["Retained (0)", "Churned (1)"], fontsize=10)

    plt.suptitle("Confusion Matrix Diagnostic Across Models", fontsize=15, fontweight="bold", y=0.98)
    out_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluation] Confusion matrices saved to {out_path}")
    return out_path


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    output_dir: str,
    top_n: int = 15,
    model_name: str = "Random Forest"
) -> str:
    """
    Plot and save feature importances from tree-based classifier.

    Parameters
    ----------
    model : Any
        Tree-based model with `feature_importances_` attribute.
    feature_names : List[str]
        Names of input features.
    output_dir : str
        Directory to save plot.
    top_n : int, default=15
        Number of top features to display.
    model_name : str, default='Random Forest'
        Name of model.

    Returns
    -------
    str
        Path to saved PNG.
    """
    set_visual_style()
    if not hasattr(model, "feature_importances_"):
        print(f"[Evaluation] Model {model_name} does not possess feature_importances_. Skipping plot.")
        return ""

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)

    print("\n" + "=" * 50)
    print(f" [EVALUATION] TOP {top_n} FEATURE IMPORTANCES ({model_name}):")
    for _, row in fi_df.iterrows():
        print(f"   {row['Feature']:<30}: {row['Importance'] * 100:.2f}%")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    norm_importances = fi_df["Importance"].values[::-1]
    norm_features = fi_df["Feature"].values[::-1]

    colors = sns.color_palette("mako", n_colors=top_n)[::-1]
    bars = ax.barh(norm_features, norm_importances, color=colors, height=0.6, edgecolor="#30363D", linewidth=1.2)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f" {width * 100:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#F0F6FC"
        )

    ax.set_title(f"Top {top_n} Churn Predictors ({model_name})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Gini Feature Importance Weight", fontsize=12)
    ax.set_xlim(0, max(norm_importances) * 1.22)

    out_path = os.path.join(output_dir, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluation] Feature importance saved to {out_path}")
    return out_path


def save_best_model(
    models: Dict[str, Any],
    comparison_df: pd.DataFrame,
    feature_names: List[str],
    output_path: str
) -> Tuple[str, Any]:
    """
    Select the top model by ROC-AUC and serialize to disk.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of models.
    comparison_df : pd.DataFrame
        Ranked comparison metrics.
    feature_names : List[str]
        List of feature column names.
    output_path : str
        Target filepath for joblib serialization.

    Returns
    -------
    Tuple[str, Any]
        (best_model_name, best_model_instance)
    """
    best_row = comparison_df.iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]

    artifact = {
        "model_name": best_name,
        "model": best_model,
        "feature_names": feature_names,
        "test_metrics": best_row.to_dict(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(artifact, output_path)

    print("\n" + "=" * 65)
    print(f" [MODEL SELECTION] CHAMPION MODEL CHOSEN: {best_name}")
    print(f"   ROC-AUC Score : {best_row['ROC-AUC']:.4f}")
    print(f"   F1-Score      : {best_row['F1-Score']:.4f}")
    print(f"   Accuracy      : {best_row['Accuracy']:.4f}")
    print(f"   Saved to      : {output_path}")
    print("=" * 65)

    return best_name, best_model


def run_full_evaluation(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None,
    model_save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    High-level orchestrator for full model evaluation suite.

    Parameters
    ----------
    models : Dict[str, Any]
        Trained model dictionary.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing ground truth labels.
    output_dir : str, optional
        Folder for visualization output.
    model_save_path : str, optional
        Path for saving best model artifact.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (comparison_df, best_model_name)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "visualizations")
    if model_save_path is None:
        model_save_path = os.path.join(base_dir, "models", "best_model.joblib")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Evaluate metrics
    comparison_df, detailed_reports = evaluate_models(models, X_test, y_test)

    # 2. Visualizations
    plot_roc_curves(detailed_reports, y_test, output_dir)
    plot_confusion_matrices(detailed_reports, output_dir)

    # 3. Feature importance from tree-based model (Random Forest / Gradient Boosting)
    rf_model = models.get("Random Forest") or models.get("Gradient Boosting")
    mname = "Random Forest" if "Random Forest" in models else "Gradient Boosting"
    if rf_model is not None:
        plot_feature_importance(rf_model, list(X_test.columns), output_dir, model_name=mname)

    # 4. Save champion model
    best_name, _ = save_best_model(models, comparison_df, list(X_test.columns), model_save_path)

    return comparison_df, best_name


if __name__ == "__main__":
    from data_loader import load_and_validate_data
    from feature_engineering import prepare_datasets
    from model_training import train_models

    df = load_and_validate_data()
    X_tr, X_te, y_tr, y_te, _ = prepare_datasets(df)
    trained, _ = train_models(X_tr, y_tr)
    run_full_evaluation(trained, X_te, y_te)
