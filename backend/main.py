from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load("diabetes_model.pkl")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: PatientData):
    X = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])
    pred_proba = model.predict_proba(X)[0]
    prediction = int(pred_proba[1] >= 0.5)
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = float(pred_proba[prediction])
    return {"prediction": prediction, "result": result, "confidence": confidence}
