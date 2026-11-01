"""
Exploratory Data Analysis (EDA) Module.

Generates insightful, publication-quality visualizations and prints
statistical summaries explaining the key business drivers behind customer churn.
"""

import os
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Setup consistent modern dark theme
def set_visual_style() -> None:
    """Configure matplotlib and seaborn styling with a modern dark theme."""
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
        "font.size": 11,
    })


# Color palette for churn status
PALETTE = {
    "No": "#00D2BE",    # Vibrant Teal / Green for Retained
    "Yes": "#FF5252",   # Coral Red for Churned
}


def plot_churn_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot and save target churn class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    counts = df["Churn"].value_counts()
    percentages = df["Churn"].value_counts(normalize=True) * 100

    print("\n" + "=" * 50)
    print(" [EDA] TARGET CLASS DISTRIBUTION:")
    for status in counts.index:
        print(f"   Churn = {status:<5}: {counts[status]:,d} customers ({percentages[status]:.2f}%)")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    bars = ax.bar(
        ["Retained (No)", "Churned (Yes)"],
        [counts.get("No", 0), counts.get("Yes", 0)],
        color=[PALETTE["No"], PALETTE["Yes"]],
        width=0.55,
        edgecolor="#30363D",
        linewidth=1.2,
    )

    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#F0F6FC",
        )

    ax.set_title("Customer Churn Distribution (Class Imbalance: ~26.5%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(output_dir, "churn_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Compute and plot correlation matrix for numerical features and churn.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    numeric_df = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
    numeric_df["Churn_Numeric"] = (df["Churn"] == "Yes").astype(int)

    corr = numeric_df.corr()

    print("\n" + "=" * 50)
    print(" [EDA] CORRELATION WITH CHURN:")
    churn_corr = corr["Churn_Numeric"].drop("Churn_Numeric").sort_values(ascending=False)
    for feat, val in churn_corr.items():
        print(f"   {feat:<18}: {val:+.4f}")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-0.6,
        vmax=0.6,
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )

    ax.set_title("Numerical Features Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    out_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_monthly_charges_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot Monthly Charges distribution segmented by churn status.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    churn_yes = df[df["Churn"] == "Yes"]["MonthlyCharges"]
    churn_no = df[df["Churn"] == "No"]["MonthlyCharges"]

    print("\n" + "=" * 50)
    print(" [EDA] MONTHLY CHARGES STATISTICS:")
    print(f"   Retained - Mean: ${churn_no.mean():.2f} | Median: ${churn_no.median():.2f}")
    print(f"   Churned  - Mean: ${churn_yes.mean():.2f} | Median: ${churn_yes.median():.2f}")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    sns.kdeplot(
        data=churn_no,
        ax=ax,
        color=PALETTE["No"],
        fill=True,
        alpha=0.35,
        linewidth=2.5,
        label=f"Retained (Median: ${churn_no.median():.2f})"
    )
    sns.kdeplot(
        data=churn_yes,
        ax=ax,
        color=PALETTE["Yes"],
        fill=True,
        alpha=0.35,
        linewidth=2.5,
        label=f"Churned (Median: ${churn_yes.median():.2f})"
    )

    ax.axvline(churn_no.median(), color=PALETTE["No"], linestyle="--", alpha=0.8)
    ax.axvline(churn_yes.median(), color=PALETTE["Yes"], linestyle="--", alpha=0.8)

    ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Monthly Charges ($)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(frameon=True, facecolor="#161B22", edgecolor="#30363D", loc="upper right")

    out_path = os.path.join(output_dir, "monthly_charges_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_tenure_vs_churn(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot Customer Tenure distribution by Churn status.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    tenure_yes = df[df["Churn"] == "Yes"]["tenure"]
    tenure_no = df[df["Churn"] == "No"]["tenure"]

    print("\n" + "=" * 50)
    print(" [EDA] TENURE (MONTHS) STATISTICS:")
    print(f"   Retained - Mean: {tenure_no.mean():.1f} mos | Median: {tenure_no.median():.1f} mos")
    print(f"   Churned  - Mean: {tenure_yes.mean():.1f} mos | Median: {tenure_yes.median():.1f} mos")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, gridspec_kw={"width_ratios": [1.4, 1]})

    # Histogram / KDE on left
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        palette=PALETTE,
        bins=36,
        kde=True,
        multiple="stack",
        ax=ax1,
        edgecolor="#30363D",
        alpha=0.85
    )
    ax1.set_title("Tenure Histogram & Density by Churn", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Tenure (Months)", fontsize=11)
    ax1.set_ylabel("Customer Count", fontsize=11)

    # Boxplot on right
    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure",
        hue="Churn",
        legend=False,
        palette=PALETTE,
        ax=ax2,
        width=0.45,
        boxprops=dict(edgecolor="#30363D", linewidth=1.5),
        whiskerprops=dict(color="#C9D1D9", linewidth=1.5),
        capprops=dict(color="#C9D1D9", linewidth=1.5),
        medianprops=dict(color="#F0F6FC", linewidth=2.5)
    )
    ax2.set_title("Tenure Distribution Boxplot", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Churn Status", fontsize=11)
    ax2.set_ylabel("Tenure (Months)", fontsize=11)

    out_path = os.path.join(output_dir, "tenure_vs_churn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_contract_type_churn(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot Churn rate broken down by Contract Type.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    contract_churn = df.groupby("Contract")["Churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reindex(["Month-to-month", "One year", "Two year"])
    contract_counts = df["Contract"].value_counts().reindex(["Month-to-month", "One year", "Two year"])

    print("\n" + "=" * 50)
    print(" [EDA] CHURN RATE BY CONTRACT TYPE:")
    for ctype, rate in contract_churn.items():
        print(f"   {ctype:<15}: {rate:.2f}% churn (out of {contract_counts[ctype]:,d} subscribers)")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    colors = ["#FF5252", "#FFA726", "#00D2BE"]
    bars = ax.bar(contract_churn.index, contract_churn.values, color=colors, width=0.5, edgecolor="#30363D", linewidth=1.2)

    for bar, ctype in zip(bars, contract_churn.index):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%\n({contract_counts[ctype]:,d} total)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#F0F6FC"
        )

    ax.set_title("Churn Rate by Contract Type (Month-to-month is 4x-10x Riskier)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_ylim(0, max(contract_churn.values) * 1.25)

    out_path = os.path.join(output_dir, "contract_type_churn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_payment_method_churn(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot Churn rate broken down by Payment Method.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    pm_churn = df.groupby("PaymentMethod")["Churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).sort_values(ascending=False)
    pm_counts = df["PaymentMethod"].value_counts().reindex(pm_churn.index)

    print("\n" + "=" * 50)
    print(" [EDA] CHURN RATE BY PAYMENT METHOD:")
    for pmethod, rate in pm_churn.items():
        print(f"   {pmethod:<28}: {rate:.2f}% churn")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    palette = ["#FF5252" if "Electronic" in m else "#4A90E2" for m in pm_churn.index]
    bars = ax.barh(pm_churn.index, pm_churn.values, color=palette, height=0.55, edgecolor="#30363D", linewidth=1.2)

    for bar, pmethod in zip(bars, pm_churn.index):
        width = bar.get_width()
        ax.annotate(
            f" {width:.1f}% ({pm_counts[pmethod]:,d})",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#F0F6FC"
        )

    ax.invert_yaxis()
    ax.set_title("Churn Rate by Payment Method (Electronic Check Peak Risk)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Churn Rate (%)", fontsize=12)
    ax.set_xlim(0, max(pm_churn.values) * 1.25)

    out_path = os.path.join(output_dir, "payment_method_churn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def plot_internet_service_churn(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot Churn rate broken down by Internet Service Type.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    output_dir : str
        Directory to save plot.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    set_visual_style()
    is_churn = df.groupby("InternetService")["Churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reindex(["Fiber optic", "DSL", "No"])
    is_counts = df["InternetService"].value_counts().reindex(["Fiber optic", "DSL", "No"])

    print("\n" + "=" * 50)
    print(" [EDA] CHURN RATE BY INTERNET SERVICE:")
    for iserv, rate in is_churn.items():
        print(f"   {iserv:<15}: {rate:.2f}% churn ({is_counts[iserv]:,d} subscribers)")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["#FF5252", "#4A90E2", "#00D2BE"]
    bars = ax.bar(is_churn.index, is_churn.values, color=colors, width=0.5, edgecolor="#30363D", linewidth=1.2)

    for bar, iserv in zip(bars, is_churn.index):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%\n({is_counts[iserv]:,d})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#F0F6FC"
        )

    ax.set_title("Churn Rate by Internet Service Type", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_ylim(0, max(is_churn.values) * 1.25)

    out_path = os.path.join(output_dir, "internet_service_churn.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def run_all_eda(df: pd.DataFrame, output_dir: Optional[str] = None) -> list:
    """
    Execute all exploratory data analysis routines and save plots.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer churn dataset.
    output_dir : str, optional
        Folder to save visualization artifacts.

    Returns
    -------
    list
        List of generated file paths.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "visualizations")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[EDA] Commencing Comprehensive Exploratory Analysis. Output dir: {output_dir}")

    plots = [
        plot_churn_distribution(df, output_dir),
        plot_correlation_heatmap(df, output_dir),
        plot_monthly_charges_distribution(df, output_dir),
        plot_tenure_vs_churn(df, output_dir),
        plot_contract_type_churn(df, output_dir),
        plot_payment_method_churn(df, output_dir),
        plot_internet_service_churn(df, output_dir),
    ]

    print(f"\n[EDA] Successfully generated and saved {len(plots)} visualization plots to {output_dir}")
    return plots


if __name__ == "__main__":
    from data_loader import load_and_validate_data
    data = load_and_validate_data()
    run_all_eda(data)
