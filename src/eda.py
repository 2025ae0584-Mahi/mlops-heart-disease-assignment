import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BASE_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from src.data_preprocessing import download_and_load_data, clean_data

# Directory to save high-res plots for the final report (Task 9)
EDA_PLOTS_DIR = BASE_DIR / "docs" / "eda_plots"


def run_eda():
    """Executes Exploratory Data Analysis and saves publication-ready visualizations."""
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Load Raw Data & Inspect Missing Values
    raw_df = download_and_load_data()
    print("=== 1. Raw Dataset Shape & Missing Values ===")
    print(f"Total Rows: {raw_df.shape[0]}, Total Columns: {raw_df.shape}")
    print("Missing values per column (encoded as '?' in raw UCI data):")
    print(raw_df.isna().sum()[raw_df.isna().sum() > 0])

    # 2. Clean Data (Binarize Target: 0 = No Disease, 1 = Heart Disease)
    df = clean_data(raw_df)
    print("\n=== 2. Summary Statistics (Cleaned Data) ===")
    print(df[NUMERIC_FEATURES].describe().T.round(2))

    # --- Plot 1: Target Class Distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET_COL, hue=TARGET_COL, palette="Set2", legend=False, ax=ax)
    ax.set_title("Target Class Distribution (0 = No Disease, 1 = Heart Disease)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Heart Disease Presence")
    ax.set_ylabel("Patient Count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "01_class_balance.png", dpi=300)
    plt.close(fig)

    # --- Plot 2: Numeric Features Histograms & KDE by Target ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_FEATURES):
        sns.histplot(
            data=df, x=col, hue=TARGET_COL, kde=True,
            palette="Set1", element="step", ax=axes[i]
        )
        axes[i].set_title(f"Distribution of {col.upper()} by Target", fontweight="bold")
    # Hide unused 6th subplot
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "02_numeric_histograms.png", dpi=300)
    plt.close(fig)

    # --- Plot 3: Feature Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix (UCI Heart Disease Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "03_correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # --- Plot 4: Boxplots for Outlier Detection ---
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 5))
    for i, col in enumerate(NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET_COL, y=col, hue=TARGET_COL, palette="Pastel1", legend=False, ax=axes[i])
        axes[i].set_title(f"{col.upper()} vs Target", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "04_numeric_boxplots.png", dpi=300)
    plt.close(fig)

    # --- Plot 5: Key Categorical Features vs Target ---
    key_cats = ["cp", "sex", "exang", "thal"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_cats):
        sns.countplot(data=df, x=col, hue=TARGET_COL, palette="viridis", ax=axes[i])
        axes[i].set_title(f"{col.upper()} Distribution by Heart Disease", fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "05_categorical_counts.png", dpi=300)
    plt.close(fig)

    print(f"\n✅ All 5 EDA visualizations successfully saved to: {EDA_PLOTS_DIR}")


if __name__ == "__main__":
    run_eda()
 