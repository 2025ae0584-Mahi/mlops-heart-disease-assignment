import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    DATA_RAW_PATH, DATA_PROCESSED_DIR, UCI_DATA_URL,
    COLUMN_NAMES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COL, RANDOM_STATE, TEST_SIZE
)


def download_and_load_data(filepath=DATA_RAW_PATH) -> pd.DataFrame:
    """Downloads the UCI Heart Disease dataset if missing and loads it."""
    os.makedirs(filepath.parent, exist_ok=True)

    if not filepath.exists():
        print(f"Downloading UCI Heart Disease dataset to {filepath}...")
        df = pd.read_csv(UCI_DATA_URL, names=COLUMN_NAMES, na_values="?")
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath, na_values="?")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataset and converts target into binary classification (0 vs 1)."""
    df = df.copy()

    # Ensure numeric types for columns that may have had '?' missing values
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where target is missing (if any) and binarize target: >0 means heart disease present
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Creates a Scikit-Learn ColumnTransformer for scaling and encoding."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def prepare_splits(df: pd.DataFrame):
    """Splits cleaned data into train/test sets and saves processed CSVs."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save processed splits for reproducibility & auditing
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    return X_train, X_test, y_train, y_test
 