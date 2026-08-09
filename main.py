import re
import json
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

MODEL_PATH = "car_price_xgb_model.json"
CATEGORIES_PATH = "categories.json"
CATEGORICAL_COLS = ["company", "model", "color", "transmission"]

app = FastAPI(
    title="Car Price API",
    description="API for car price prediction using a pre-trained XGBoost model.",
)

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    _categories = json.load(f)

_model = xgb.XGBRegressor(enable_categorical=True)
_model.load_model(MODEL_PATH)


def normalize_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).capitalize()


class CarRequest(BaseModel):
    company: str = Field(..., examples=["Kia"])
    model: str = Field(..., examples=["Rio"])
    year: int = Field(..., ge=1950, le=2027, examples=[2011])
    mileage: float = Field(..., ge=0, examples=[145000])
    color: str = Field(..., examples=["White"])
    transmission: str = Field(..., examples=["automatic"])


class PredictResponse(BaseModel):
    predicted_price_usd: float
    warnings: List[str] = []


@app.get("/")
def root():
    return {"status": "ok", "message": "Car price prediction API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(car: CarRequest):

    company_norm = normalize_text(car.company)
    model_norm = normalize_text(car.model)
    color_norm = normalize_text(car.color)
    transmission_norm = normalize_text(car.transmission)
    row = pd.DataFrame(
        [
            {
                "company": company_norm,
                "model": model_norm,
                "year": car.year,
                "mileage": car.mileage,
                "color": color_norm,
                "transmission": transmission_norm,
            }
        ]
    )

    warnings: List[str] = []
    for col in CATEGORICAL_COLS:
        row[col] = pd.Categorical(row[col], categories=_categories[col])
        if row[col].isnull().any():
            warnings.append(f"القيمة في '{col}' غير موجودة في بيانات التدريب، ستعامل كـ missing")

    prediction = float(_model.predict(row)[0])

    return PredictResponse(
        predicted_price_usd=round(prediction, 2),
        warnings=warnings,
    )