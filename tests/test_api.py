import pytest
from fastapi.testclient import TestClient

# Gracefully skip API tests if app/main.py hasn't been created yet
app_module = pytest.importorskip("app.main", reason="FastAPI app/main.py not created yet (Task 6)")
client = TestClient(app_module.app)


def test_health_check_endpoint():
    """Verifies the /health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint_valid_payload():
    """Verifies /predict accepts valid patient JSON and returns prediction + risk probability."""
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0,1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_endpoint_invalid_payload():
    """Verifies Pydantic schema rejects missing or invalid data types with HTTP 422."""
    invalid_payload = {
        "age": "invalid_age",  # Should be numeric
        "sex": 1
        # Missing remaining required clinical features
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
 