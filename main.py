import re
import json
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware
MODEL_PATH = "car_price_xgb_model.json"
CATEGORIES_PATH = "categories.json"
CATEGORICAL_COLS = ["company", "model", "color", "transmission"]

app = FastAPI(
    title="Car Price API",
    description="API for car price prediction using a pre-trained XGBoost model.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    invalid_fields = []
    for col in CATEGORICAL_COLS:
        row[col] = pd.Categorical(row[col], categories=_categories[col])
        if row[col].isnull().any():
            invalid_fields.append(
                {
                    "field": col,
                    "message": f"القيمة في '{col}' غير موجودة في بيانات التدريب",
                }
            )

    if invalid_fields:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "توجد قيم غير صحيحة في الطلب",
                "errors": invalid_fields,
            },
        )

    prediction = float(_model.predict(row)[0])

    return PredictResponse(
        predicted_price_usd=round(prediction, 2),
    )