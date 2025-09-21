from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam & Fake News Detector (Starter)")

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "nb_pipeline.joblib"
LABELS = {0: "ham", 1: "spam"}
model = joblib.load(MODEL_PATH)

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    reason: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    text = payload.text
    pred_array = model.predict([text])  # shape: (1,)
    pred = int(pred_array[0])  # 0 or 1

    proba_matrix = model.predict_proba([text])  # shape: (1, 2)
    proba_row = proba_matrix[0]  # shape: (2,) e.g. [P(ham), P(spam)]
    confidence = float(proba_row[pred])  # pick P of predicted class

    label = LABELS[pred]  # "ham" or "spam"
    return PredictOut(label=label, reason=f"confidence = {confidence:.3f}")
