from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Spam & Fake News Detector (Starter)")

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
    text = payload.text.lower()
    if "free" in text or "win" in text:
        return PredictOut(label="spam", reason="Matched spam keywords")
    elif "you won’t believe" in text or "shocking truth" in text:
        return PredictOut(label="fake_news", reason="Matched fake-news style phrase")
    return PredictOut(label="ham", reason="No spam/fake terms found")
