import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from src.config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from app.schemas import PatientData, PredictionResponse

# --- Task 8: Configure Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-disease-api")

# --- Task 8: Custom Prometheus ML Metrics ---
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of heart disease predictions served",
    ["predicted_class", "risk_level"]
)
INFERENCE_LATENCY = Histogram(
    "heart_disease_inference_latency_seconds",
    "Time spent running model inference in seconds"
)

# --- Task 6: Initialize FastAPI Application ---
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production MLOps API serving the UCI Heart Disease classification pipeline.",
    version="1.0.0"
)

# Expose standard HTTP & FastAPI metrics at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def load_or_create_model():
    """Loads packaged pipeline or creates a lightweight fallback for isolated CI unit tests."""
    if MODEL_PATH.exists():
        logger.info(f"Loading packaged model pipeline from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    logger.warning(f"Model file not found at {MODEL_PATH}. Building fallback pipeline for testing...")
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from src.data_preprocessing import build_preprocessor

    dummy_df = pd.DataFrame([
        {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6},
        {"age": 37, "sex": 0, "cp": 3, "trestbps": 120, "chol": 210, "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}
    ])
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(dummy_df,)
    return pipeline


model_pipeline = load_or_create_model()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log every incoming HTTP request, status code, and processing duration."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Duration={duration_ms:.2f}ms"
    )
    return response


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for Docker and Kubernetes liveness/readiness probes."""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_heart_disease(patient: PatientData):
    """Accepts patient health JSON and returns heart disease prediction + probability."""
    try:
        start_infer = time.perf_counter()

        # Convert validated Pydantic model to DataFrame with exact column order
        input_dict = patient.model_dump()
        input_df = pd.DataFrame([input_dict])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        # Run inference through the Scikit-Learn Pipeline
        prediction = int(model_pipeline.predict(input_df)[0])
        probability = float(model_pipeline.predict_proba(input_df)[0][1])

        infer_seconds = time.perf_counter() - start_infer
        INFERENCE_LATENCY.observe(infer_seconds)

        # Assign clinical risk tier
        if probability < 0.35:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Update Prometheus counter & write structured audit log
        PREDICTION_COUNTER.labels(predicted_class=str(prediction), risk_level=risk_level).inc()
        logger.info(
            f"Prediction served | Input={input_dict} | "
            f"Pred={prediction} | Prob={probability:.4f} | Risk={risk_level}"
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(exc)}")
 