# Car Price Prediction API — Model 2 (Chinese-Brand Coverage)

> Companion service to [Car_Prices](https://github.com/Ghaith772/Car_Prices).

The first model (`Car_Prices`) was trained on a Kaggle dataset centered on the US market, which doesn't include Chinese car brands. Since the target market for this project is Syria — where Chinese brands make up a large share of the cars on the road — this second model was trained on a separate dataset that primarily covers Chinese brands. In the full marketplace platform, an incoming prediction request is first sent to the first model; if it can't recognize the car (e.g. an unrecognized make/model), the request falls back to this model; if this model also can't recognize it, the API rejects the request with an error instead of returning an unreliable prediction.

## Features

- **`POST /predict`** — accepts car details (company, model, year, mileage, color, transmission) and returns a predicted price in USD.
- **`GET /`** — health check endpoint.
- Request validation with Pydantic, including category-membership checks: any `company`/`model`/`color`/`transmission` value outside the training data's known categories is rejected with a clear error before it reaches the model.
- Wide brand coverage, including major Chinese manufacturers (BYD, Chery, Changan, Geely, Haval, JAC, Jetour, Great Wall, MG, and others) alongside mainstream global brands.
- CORS configured for local frontend development (`localhost:5173`).

## Tech Stack

- **API:** Python, FastAPI, Pydantic, Uvicorn
- **Modeling:** XGBoost (`XGBRegressor`, native categorical feature support)
- **Data processing:** Pandas, NumPy, scikit-learn

## Data Source & Cleaning (`train.py`)

Trained on the [Used Cars in Egypt 2025](https://www.kaggle.com/datasets/mohamedsewid/used-cars-in-egypt-2025) dataset from Kaggle (listings collected from eg.hatla2ee.com), with a cleaning pipeline that handles the realities of noisy, user-submitted listing data:

1. **Price & mileage parsing** — strips currency symbols and thousands separators, converts to numeric.
2. **Fake-price detection** — removes listings with obviously fake repdigit prices (e.g. `9999999`).
3. **Brand-name repair** — fixes multi-word brand names (e.g. "Land" → "Land Rover", "Great" → "Great Wall") that were split incorrectly during scraping, using the listing title to recover the correct model.
4. **Misaligned-field repair** — detects rows where the model field was accidentally populated with the year, and recovers the real year from the title.
5. **Text normalization** — standardizes `company`, `model`, `color`, and `transmission` values.
6. **Deduplication & filtering** — drops duplicate rows, rows with `model == "Other"`, and rows missing `price` or `year`.
7. **Currency conversion** — converts price from EGP to USD.
8. **Outlier removal** — drops prices above $1,000,000 or below $200, mileage beyond an IQR-based upper bound, and listings priced more than 5x their peer group's median price (same company/model/year).
9. **Transmission extraction** — derives `transmission` (automatic/manual) from a free-text `features` column when not explicitly labeled.
10. **Training** — an `XGBRegressor` with native categorical support (`enable_categorical=True`), evaluated on a held-out test set with R², MAE, and RMSE, plus a feature-importance report.

The cleaned dataset is also saved to `hatla2ee_cars_cleaned.csv` for reuse.

## Project Structure

```
.
├── main.py                       # FastAPI app: /predict and / (health check) endpoints
├── train.py                       # Data cleaning, feature engineering, and model training
├── car_price_xgb_model.json        # Saved trained model
├── categories.json                  # Known category values used for request validation
└── requirements.txt                  # Python dependencies
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Send a `POST /predict` request with a JSON body matching the `CarRequest` schema:

```json
{
  "company": "Kia",
  "model": "Rio",
  "year": 2011,
  "mileage": 145000,
  "color": "White",
  "transmission": "automatic"
}
```

## Retraining

The raw dataset (`hatla2ee_cars_august_2025.csv`) is not included in this repository. To retrain, place a dataset with the same columns (`title`, `company`, `model`, `year`, `price`, `mileage`, `color`, `features`, `location`, `date_posted`, `detail_link`) in the project root and run:

```bash
python train.py
```

This regenerates `car_price_xgb_model.json` and the cleaned dataset.
