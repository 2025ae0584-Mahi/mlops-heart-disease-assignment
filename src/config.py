from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "heart.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.pkl"

# UCI Heart Disease Dataset URL (Cleveland Processed Dataset)
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column Names for UCI Cleveland Heart Disease Dataset (14 attributes)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Feature Groups for Preprocessing Pipeline
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"

# Training & MLflow Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "Heart_Disease_Classification"
 