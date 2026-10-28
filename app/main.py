from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Locate model.joblib in the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Load the trained pipeline
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SMS Spam Detection API",
    description="Classifies SMS messages as spam or ham.",
    version="1.0.0",
)


class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="SMS text to classify",
    )


@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(request: MessageRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty or whitespace only.",
        )

    prediction = model.predict([message])[0]

    return {
        "message": message,
        "prediction": str(prediction),
        "is_spam": bool(prediction == "spam"),
    }