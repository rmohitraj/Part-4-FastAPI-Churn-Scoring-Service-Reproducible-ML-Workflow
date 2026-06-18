"""
Churn Scoring Service – FastAPI Application
===========================================
Endpoints
---------
GET  /health          → Service health check
POST /predict         → Single-customer churn prediction
POST /batch_predict   → Multi-customer churn prediction
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
META_PATH  = BASE_DIR / "feature_meta.json"

# Decision threshold (recall-optimised, matching Part 3)
DEFAULT_THRESHOLD = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="D2C Churn Scoring Service",
    version="1.0.0",
    description=(
        "Predicts whether a customer will churn within the next 60 days, "
        "returning a probability, predicted class, and risk explanation."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (once, at startup)
# ──────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    threshold = DEFAULT_THRESHOLD
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", DEFAULT_THRESHOLD)

    return model, threshold


try:
    MODEL, THRESHOLD = _load_artifacts()
except FileNotFoundError as exc:
    # Allow the app to start so /health can report degraded state
    MODEL, THRESHOLD = None, DEFAULT_THRESHOLD
    print(f"WARNING: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Feature payload for a single customer.
    All fields mirror the columns used during model training (Part 3).
    """

    # ── Recency / activity ───────────────────────────────────────────────────
    recency_days: float = Field(
        ..., ge=0, description="Days since last purchase"
    )
    last_visit_days_ago: float = Field(
        ..., ge=0, description="Days since last website/app visit"
    )

    # ── Frequency & monetary (180-day window) ────────────────────────────────
    frequency_180d: int = Field(
        ..., ge=0, description="Number of orders in last 180 days"
    )
    monetary_180d: float = Field(
        ..., ge=0, description="Total spend (₹) in last 180 days"
    )

    # ── Customer tenure ──────────────────────────────────────────────────────
    days_since_signup: int = Field(
        ..., ge=0, description="Days since account creation"
    )

    # ── 30-day engagement signals ────────────────────────────────────────────
    product_views_30d: int = Field(0, ge=0, description="Product pages viewed in last 30 days")
    sessions_30d: int = Field(0, ge=0, description="App/web sessions in last 30 days")
    cart_adds_30d: int = Field(0, ge=0, description="Items added to cart in last 30 days")
    wishlist_adds_30d: int = Field(0, ge=0, description="Wishlist additions in last 30 days")
    campaign_clicks_30d: int = Field(0, ge=0, description="Marketing email/campaign clicks in last 30 days")
    email_opens_30d: int = Field(0, ge=0, description="Marketing email opens in last 30 days")

    # ── Quality / discount signals ───────────────────────────────────────────
    avg_discount_pct_180d: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Average discount fraction applied in last 180 days (0–1)"
    )
    avg_rating_180d: float = Field(
        3.0, ge=1.0, le=5.0,
        description="Average product rating given in last 180 days"
    )

    # ── Support interactions ─────────────────────────────────────────────────
    support_ticket_count: int = Field(
        0, ge=0, description="Total support tickets raised"
    )

    # ── Categorical / demographic ────────────────────────────────────────────
    loyalty_tier: str = Field(
        "Unknown",
        description="Loyalty programme tier: Gold | Silver | Bronze | Unknown"
    )
    city_tier: str = Field(
        "Tier2",
        description="City tier of customer's location: Tier1 | Tier2 | Tier3"
    )
    age_group: str = Field(
        "25-34",
        description="Customer age bracket: 18-24 | 25-34 | 35-44 | 45+"
    )
    acquisition_channel: str = Field(
        "Organic",
        description="Channel through which customer was acquired"
    )

    @field_validator("loyalty_tier")
    @classmethod
    def validate_loyalty(cls, v: str) -> str:
        allowed = {"Gold", "Silver", "Bronze", "Unknown"}
        if v not in allowed:
            raise ValueError(f"loyalty_tier must be one of {allowed}")
        return v

    @field_validator("city_tier")
    @classmethod
    def validate_city(cls, v: str) -> str:
        allowed = {"Tier1", "Tier2", "Tier3"}
        if v not in allowed:
            raise ValueError(f"city_tier must be one of {allowed}")
        return v

    @field_validator("age_group")
    @classmethod
    def validate_age(cls, v: str) -> str:
        allowed = {"18-24", "25-34", "35-44", "45+"}
        if v not in allowed:
            raise ValueError(f"age_group must be one of {allowed}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "acquisition_channel": "Instagram",
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    predicted_class: int = Field(..., description="Binary churn prediction: 0 = retain, 1 = churn")
    risk_explanation: str = Field(..., description="Plain-language explanation of the prediction")


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=500,
        description="List of customer feature payloads (1–500)"
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


# ──────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _features_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a Pydantic model to the DataFrame format the pipeline expects."""
    return pd.DataFrame([customer.model_dump()])


def _build_explanation(prob: float, features: CustomerFeatures, predicted: int) -> str:
    """
    Generate a concise, human-readable risk explanation based on the top
    churn drivers identified in Part 3 (recency, monetary, recency of visit,
    support tickets, engagement).
    """
    reasons: list[str] = []

    if features.recency_days > 90:
        reasons.append(f"no purchase for {int(features.recency_days)} days")
    if features.last_visit_days_ago > 45:
        reasons.append(f"last site visit {int(features.last_visit_days_ago)} days ago")
    if features.monetary_180d < 200:
        reasons.append("very low spend in the past 180 days")
    if features.sessions_30d < 2:
        reasons.append("minimal recent platform engagement")
    if features.support_ticket_count >= 3:
        reasons.append(f"elevated support-ticket count ({features.support_ticket_count})")
    if features.avg_rating_180d < 2.5:
        reasons.append("consistently low product ratings")
    if features.avg_discount_pct_180d > 0.35:
        reasons.append("high discount dependency (price-sensitive behaviour)")

    if not reasons:
        if predicted == 1:
            reasons.append("combination of moderate engagement signals")
        else:
            reasons.append("strong recent activity and purchase history")

    risk_label = (
        "High" if prob >= 0.70
        else "Medium" if prob >= 0.40
        else "Low"
    )
    action = (
        "Immediate retention outreach is recommended."
        if predicted == 1 and prob >= 0.60
        else "Monitor this customer closely."
        if predicted == 1
        else "No immediate intervention required."
    )

    return (
        f"{risk_label} churn risk ({prob:.0%}): "
        + "; ".join(reasons).capitalize()
        + f". {action}"
    )


def _predict_one(customer: CustomerFeatures) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    df = _features_to_df(customer)
    prob = float(MODEL.predict_proba(df)[0][1])
    predicted = int(prob >= THRESHOLD)
    explanation = _build_explanation(prob, customer, predicted)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        predicted_class=predicted,
        risk_explanation=explanation,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
def health_check():
    """Returns service health and model load status."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "decision_threshold": THRESHOLD,
        "version": app.version,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a **single** customer.

    Returns:
    - **churn_probability** – score between 0 and 1
    - **predicted_class** – 0 (retain) or 1 (churn), using threshold 0.25
    - **risk_explanation** – plain-language explanation of key risk drivers
    """
    return _predict_one(customer)


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
def batch_predict(body: BatchPredictRequest):
    """
    Predict churn for up to **500 customers** in a single call.

    Each item in `customers` receives its own prediction result.
    """
    results = [_predict_one(c) for c in body.customers]
    return BatchPredictionResponse(predictions=results, total=len(results))
