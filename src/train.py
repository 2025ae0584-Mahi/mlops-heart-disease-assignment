import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

from src.config import (
    MODEL_DIR, MODEL_PATH, RANDOM_STATE, MLFLOW_EXPERIMENT_NAME
)
from src.data_preprocessing import (
    download_and_load_data, clean_data, build_preprocessor, prepare_splits
)


def evaluate_metrics(y_true, y_pred, y_prob) -> dict:
    """Computes classification metrics required by the assignment rubric."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def log_plots_to_mlflow(model, X_test, y_test, y_pred, model_name: str):
    """Generates and logs Confusion Matrix and ROC Curve plots as MLflow artifacts."""
    os.makedirs("temp_plots", exist_ok=True)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    cm_path = f"temp_plots/{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # 2. ROC Curve Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve - {model_name}")
    roc_path = f"temp_plots/{model_name}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close(fig)
    mlflow.log_artifact(roc_path, artifact_path="plots")


def train_and_evaluate():
    """Main execution function for training, MLflow logging, and model packaging."""
    # Step 1: Load, clean, and split data
    raw_df = download_and_load_data()
    cleaned_df = clean_data(raw_df)
    X_train, X_test, y_train, y_test = prepare_splits(cleaned_df)

    # Step 2: Configure MLflow Experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Step 3: Define candidate models and hyperparameter grids (Task 2)
    models_config = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"]
            }
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5]
            }
        }
    }

    best_overall_model = None
    best_overall_auc = -1.0
    best_model_name = ""

    # Step 4: Train, tune, and track each model in MLflow (Task 3)
    for model_name, cfg in models_config.items():
        with mlflow.start_run(run_name=model_name):
            print(f"\n--- Training & Tuning {model_name} ---")

            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", cfg["estimator"])
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=cfg["param_grid"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate_metrics(y_test, y_pred, y_prob)

            # Log parameters, CV score, and test metrics to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_best_roc_auc", round(grid_search.best_score_, 4))
            mlflow.log_metrics(metrics)

            # Log visual plots and model artifact to MLflow
            log_plots_to_mlflow(best_pipeline, X_test, y_test, y_pred, model_name)
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model",
                input_example=X_train.iloc[:3]
            )

            print(f"{model_name} Best Params: {grid_search.best_params_}")
            print(f"{model_name} Test Metrics: {metrics}")

            # Track best overall model based on ROC-AUC
            if metrics["roc_auc"] > best_overall_auc:
                best_overall_auc = metrics["roc_auc"]
                best_overall_model = best_pipeline
                best_model_name = model_name

    # Step 5: Package the winning model for FastAPI & Docker (Task 4)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_overall_model, MODEL_PATH)
    print(f"\n✅ Best Model ({best_model_name} | ROC-AUC: {best_overall_auc}) saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
 