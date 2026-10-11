
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Bangalore House Price Prediction API")


class HouseDetails(BaseModel):
    location: str = Field(min_length=1)
    total_sqft: float = Field(gt=0)
    bhk: int = Field(ge=1)
    bath: float = Field(gt=0)
    balcony: float = Field(ge=0)
    area_type: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Bangalore House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_price(house: HouseDetails):
    input_data = pd.DataFrame([{
        "location": house.location.strip(),
        "total_sqft": house.total_sqft,
        "bhk": house.bhk,
        "bath": house.bath,
        "balcony": house.balcony,
        "area_type": house.area_type.strip()
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        return {
            "predicted_price_lakhs": round(prediction, 2),
            "predicted_price_rupees": round(prediction * 100000),
            "currency": "INR"
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
