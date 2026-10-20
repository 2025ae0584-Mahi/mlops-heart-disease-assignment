from pydantic import BaseModel, Field


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: float = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: float = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: float = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment (1-3)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., ge=3, le=7, description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Heart Disease, 1 = Heart Disease Present")
    probability: float = Field(..., description="Predicted probability of Heart Disease (0.0 to 1.0)")
    risk_level: str = Field(..., description="Clinical risk tier: Low, Moderate, or High")
 