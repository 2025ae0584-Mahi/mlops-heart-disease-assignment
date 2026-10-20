import pandas as pd
import numpy as np
import pytest
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from src.data_preprocessing import clean_data, build_preprocessor


@pytest.fixture
def sample_raw_df():
    """Creates a synthetic raw DataFrame mimicking UCI Heart Disease data with '?' missing values."""
    data = {
        "age": [63, 67, 67, 37, 41],
        "sex": [1, 1, 1, 0, 0],
        "cp": [3, 2, 4, 1, 2],
        "trestbps": [145, 160, 120, 130, 130],
        "chol": [233, 286, 229, 250, 204],
        "fbs": [0, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 0, 2],
        "thalach": [150, 108, 129, 187, 172],
        "exang": [0, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
        "slope": [2, 2, 2, 3, 1],
        "ca": [0, 3, 2, "?", 0],       # Contains '?' missing value
        "thal": [6, 3, 7, 3, "?"],     # Contains '?' missing value
        "target": [0, 1, 2, 3, 4]      # Multi-class target (0-4) to test binarization
    }
    return pd.DataFrame(data)


def test_clean_data_binarizes_target(sample_raw_df):
    """Verifies that target values > 0 are converted to 1 and 0 remains 0."""
    cleaned_df = clean_data(sample_raw_df)
    unique_targets = set(cleaned_df[TARGET_COL].unique())
    assert unique_targets.issubset({0, 1}), f"Expected binary target {{0, 1}}, got {unique_targets}"
    assert list(cleaned_df[TARGET_COL]) == [0, 1, 1, 1, 1]


def test_clean_data_coerces_missing_strings(sample_raw_df):
    """Verifies that '?' strings are coerced to numeric NaN values."""
    cleaned_df = clean_data(sample_raw_df)
    assert cleaned_df["ca"].isna().sum() == 1
    assert cleaned_df["thal"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(cleaned_df["ca"])


def test_preprocessor_output_has_no_nans(sample_raw_df):
    """Verifies that ColumnTransformer imputes missing values and scales/encodes without NaNs."""
    cleaned_df = clean_data(sample_raw_df)
    X = cleaned_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(cleaned_df)
    assert not np.isnan(X_transformed).any(), "Transformed feature matrix should not contain NaNs"

 