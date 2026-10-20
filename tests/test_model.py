import joblib
import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data_preprocessing import build_preprocessor
from src.train import evaluate_metrics


@pytest.fixture
def sample_patient_df():
    """Single patient feature record for testing model inference."""
    return pd.DataFrame([{
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0
    }])


def test_evaluate_metrics_computation():
    """Verifies that evaluate_metrics returns all 5 required rubric metrics within."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.85])

    metrics = evaluate_metrics(y_true, y_pred, y_prob)
    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}

    assert set(metrics.keys()) == expected_keys
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {key} out of range: {val}"


def test_pipeline_fit_and_predict(sample_patient_df):
    """Tests that the full Pipeline (Preprocessor + Classifier) fits and predicts probabilities."""
    # Create synthetic training batch of 10 rows
    X_train = pd.concat([sample_patient_df] * 10, ignore_index=True)
    y_train = np.array([0, 1, 1, 0, 1] * 2)

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(sample_patient_df)
    prob = pipeline.predict_proba(sample_patient_df)

    assert pred.shape == (1,)
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert np.isclose(prob.sum(), 1.0)


def test_saved_model_artifact_loads_and_predicts(sample_patient_df):
    """If the packaged model exists, verifies it loads cleanly and outputs valid predictions."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact not found at {MODEL_PATH}. Run 'python -m src.train' first.")

    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")

    X = sample_patient_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    assert prediction in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in probabilities)
 