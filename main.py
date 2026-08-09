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

class NormalizeRequest(BaseModel):
    text: str = Field(..., examples=["Range Rover"])


class NormalizeResponse(BaseModel):
    original: str
    normalized: str


class CarRequest(BaseModel):
    company: str = Field(..., examples=["Range Rover"])
    model: str = Field(..., examples=["Evoque"])
    year: int = Field(..., ge=1950, le=2027, examples=[2021])
    mileage: float = Field(..., ge=0, examples=[45000])
    color: str = Field(..., examples=["White"])
    transmission: str = Field(..., pattern="^(automatic|manual)$", examples=["automatic"])


class PredictResponse(BaseModel):
    company_normalized: str
    model_normalized: str
    predicted_price_usd: float
    warnings: List[str] = []

@app.get("/")
def root():
    return {"status": "ok", "message": "Car price prediction API is running"}


# @app.post("/normalize", response_model=NormalizeResponse)
# def normalize(req: NormalizeRequest):
#    
#     return NormalizeResponse(original=req.text, normalized=normalize_text(req.text))


@app.post("/predict", response_model=PredictResponse)
def predict(car: CarRequest):

    company_norm = normalize_text(car.company)
    model_norm = normalize_text(car.model)

    row = pd.DataFrame(
        [
            {
                "company": company_norm,
                "model": model_norm,
                "year": car.year,
                "mileage": car.mileage,
                "color": car.color,
                "transmission": car.transmission,
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
        company_normalized=company_norm,
        model_normalized=model_norm,
        predicted_price_usd=round(prediction, 2),
        warnings=warnings,
    )