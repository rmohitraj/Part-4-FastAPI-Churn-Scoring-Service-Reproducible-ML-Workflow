# Part-4-FastAPI-Churn-Scoring-Service-Reproducible-ML-Workflow
# D2C Churn Scoring Service

A production-ready FastAPI microservice that loads a trained LightGBM churn model and exposes REST endpoints for single and batch customer churn predictions. Built as Part 4 of the D2C Customer Analytics Capstone.

---

## Project Structure

```
churn_api/
├── app/
│   └── main.py              # FastAPI application (endpoints, schemas, logic)
├── tests/
│   └── test_api.py          # API test suite (pytest)
├── model.pkl                # Trained LightGBM pipeline (from Part 3)
├── feature_meta.json        # Feature column names & decision threshold
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── monitoring_plan.md       # Post-deployment monitoring plan
├── responsible_use_note.md  # Ethical & responsible-use guidance
└── README.md                # This file
```

---

## Setup

### Prerequisites

- Python 3.11+
- `model.pkl` in the project root (trained in Part 3 using LightGBM)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

---

## Docker (Recommended for Reproducibility)

### Build the image

```bash
docker build -t churn-api:1.0 .
```

### Run the container

```bash
docker run -p 8000:8000 churn-api:1.0
```

The service starts at `http://localhost:8000`.

### Verify the container is running

```bash
curl http://localhost:8000/health
```

---

## Endpoints

### `GET /health`

Returns service health and model load status.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "decision_threshold": 0.25,
  "version": "1.0.0"
}
```

---

### `POST /predict`

Predicts churn risk for a single customer.

**Request body** (all fields required unless a default is shown)

| Field | Type | Description |
|---|---|---|
| `recency_days` | float ≥ 0 | Days since last purchase |
| `last_visit_days_ago` | float ≥ 0 | Days since last platform visit |
| `frequency_180d` | int ≥ 0 | Orders in last 180 days |
| `monetary_180d` | float ≥ 0 | Total spend (₹) in last 180 days |
| `days_since_signup` | int ≥ 0 | Account tenure in days |
| `product_views_30d` | int ≥ 0 | Product pages viewed in last 30 days (default 0) |
| `sessions_30d` | int ≥ 0 | App/web sessions in last 30 days (default 0) |
| `cart_adds_30d` | int ≥ 0 | Cart additions in last 30 days (default 0) |
| `wishlist_adds_30d` | int ≥ 0 | Wishlist additions in last 30 days (default 0) |
| `campaign_clicks_30d` | int ≥ 0 | Campaign link clicks in last 30 days (default 0) |
| `email_opens_30d` | int ≥ 0 | Email opens in last 30 days (default 0) |
| `avg_discount_pct_180d` | float 0–1 | Average discount fraction (default 0.0) |
| `avg_rating_180d` | float 1–5 | Average product rating given (default 3.0) |
| `support_ticket_count` | int ≥ 0 | Total support tickets raised (default 0) |
| `loyalty_tier` | string | `Gold`, `Silver`, `Bronze`, `Unknown` (default `Unknown`) |
| `city_tier` | string | `Tier1`, `Tier2`, `Tier3` (default `Tier2`) |
| `age_group` | string | `18-24`, `25-34`, `35-44`, `45+` (default `25-34`) |
| `acquisition_channel` | string | e.g., `Instagram`, `Google`, `Referral`, `Organic`, `Email` |

**Sample request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "recency_days": 45,
    "last_visit_days_ago": 30,
    "frequency_180d": 3,
    "monetary_180d": 850.0,
    "days_since_signup": 400,
    "product_views_30d": 8,
    "sessions_30d": 5,
    "cart_adds_30d": 2,
    "wishlist_adds_30d": 1,
    "campaign_clicks_30d": 0,
    "email_opens_30d": 1,
    "avg_discount_pct_180d": 0.12,
    "avg_rating_180d": 3.8,
    "support_ticket_count": 2,
    "loyalty_tier": "Silver",
    "city_tier": "Tier2",
    "age_group": "35-44",
    "acquisition_channel": "Instagram"
  }'
```

**Sample response**

```json
{
  "churn_probability": 0.4127,
  "predicted_class": 1,
  "risk_explanation": "Medium churn risk (41%): No purchase for 45 days; minimal recent platform engagement. Monitor this customer closely."
}
```

---

### `POST /batch_predict`

Predicts churn risk for up to 500 customers in one call.

**Sample request**

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "recency_days": 150, "last_visit_days_ago": 80,
        "frequency_180d": 1, "monetary_180d": 60.0,
        "days_since_signup": 900, "product_views_30d": 1,
        "sessions_30d": 0, "cart_adds_30d": 0,
        "wishlist_adds_30d": 0, "campaign_clicks_30d": 0,
        "email_opens_30d": 0, "avg_discount_pct_180d": 0.45,
        "avg_rating_180d": 2.0, "support_ticket_count": 5,
        "loyalty_tier": "Unknown", "city_tier": "Tier3",
        "age_group": "45+", "acquisition_channel": "Organic"
      },
      {
        "recency_days": 5, "last_visit_days_ago": 2,
        "frequency_180d": 12, "monetary_180d": 4200.0,
        "days_since_signup": 600, "product_views_30d": 55,
        "sessions_30d": 30, "cart_adds_30d": 10,
        "wishlist_adds_30d": 8, "campaign_clicks_30d": 5,
        "email_opens_30d": 9, "avg_discount_pct_180d": 0.05,
        "avg_rating_180d": 4.8, "support_ticket_count": 0,
        "loyalty_tier": "Gold", "city_tier": "Tier1",
        "age_group": "25-34", "acquisition_channel": "Referral"
      }
    ]
  }'
```

**Sample response**

```json
{
  "predictions": [
    {
      "churn_probability": 0.8912,
      "predicted_class": 1,
      "risk_explanation": "High churn risk (89%): No purchase for 150 days; last site visit 80 days ago; very low spend in the past 180 days; minimal recent platform engagement; elevated support-ticket count (5); consistently low product ratings; high discount dependency (price-sensitive behaviour). Immediate retention outreach is recommended."
    },
    {
      "churn_probability": 0.0341,
      "predicted_class": 0,
      "risk_explanation": "Low churn risk (3%): Strong recent activity and purchase history. No immediate intervention required."
    }
  ],
  "total": 2
}
```

---

## Running Tests

```bash
pytest tests/test_api.py -v
```

Expected output: all tests pass (4 test classes, 20+ test cases covering health, predict, validation, and batch predict).

To run with coverage:

```bash
pip install pytest-cov
pytest tests/test_api.py -v --cov=app --cov-report=term-missing
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | LightGBM Classifier (`LGBMClassifier`) |
| Target | `churn_next_60d` (binary: 0 / 1) |
| Decision threshold | 0.25 (recall-optimised) |
| ROC-AUC (validation) | 0.8744 |
| Precision @ 0.25 | ~0.69 |
| Recall @ 0.25 | ~0.88 |
| Top features | `recency_days`, `monetary_180d`, `days_since_signup`, `last_visit_days_ago` |

The model was trained using a time-based split to prevent data leakage (train: 1728 samples, val: 336, test: 336).

---

## Responsible Use

See [`responsible_use_note.md`](./responsible_use_note.md) for guidance on how the retention team should and should not use the API output, including ethical considerations and known biases.

## Monitoring

See [`monitoring_plan.md`](./monitoring_plan.md) for post-deployment monitoring of data drift, prediction distribution, business KPIs, API errors, and retraining triggers.

---

## Reproducibility Checklist

- [x] `requirements.txt` with pinned versions
- [x] `Dockerfile` for fully isolated builds
- [x] `feature_meta.json` stores column names and threshold (no hardcoded magic)
- [x] `random_state=42` used throughout training
- [x] Time-based train/val/test split (no data leakage)
- [x] Model serialised with `joblib` for deterministic loading
