"""
tests/test_api.py
=================
API test suite for the D2C Churn Scoring Service.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Ensure the app can find model.pkl relative to project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def high_risk_customer():
    """Customer with strong churn signals (high recency, low spend, low engagement)."""
    return {
        "recency_days": 150,
        "last_visit_days_ago": 80,
        "frequency_180d": 1,
        "monetary_180d": 60.0,
        "days_since_signup": 900,
        "product_views_30d": 1,
        "sessions_30d": 0,
        "cart_adds_30d": 0,
        "wishlist_adds_30d": 0,
        "campaign_clicks_30d": 0,
        "email_opens_30d": 0,
        "avg_discount_pct_180d": 0.45,
        "avg_rating_180d": 2.0,
        "support_ticket_count": 5,
        "loyalty_tier": "Unknown",
        "city_tier": "Tier3",
        "age_group": "45+",
        "acquisition_channel": "Organic",
    }


@pytest.fixture
def low_risk_customer():
    """Customer with strong retention signals (recent, high spend, active)."""
    return {
        "recency_days": 5,
        "last_visit_days_ago": 2,
        "frequency_180d": 12,
        "monetary_180d": 4200.0,
        "days_since_signup": 600,
        "product_views_30d": 55,
        "sessions_30d": 30,
        "cart_adds_30d": 10,
        "wishlist_adds_30d": 8,
        "campaign_clicks_30d": 5,
        "email_opens_30d": 9,
        "avg_discount_pct_180d": 0.05,
        "avg_rating_180d": 4.8,
        "support_ticket_count": 0,
        "loyalty_tier": "Gold",
        "city_tier": "Tier1",
        "age_group": "25-34",
        "acquisition_channel": "Referral",
    }


@pytest.fixture
def mid_risk_customer():
    """Customer with mixed signals (moderate activity, some negative indicators)."""
    return {
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


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_reports_model_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_contains_threshold(self):
        response = client.get("/health")
        data = response.json()
        assert "decision_threshold" in data
        assert 0.0 < data["decision_threshold"] < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – /predict endpoint: response structure & value ranges
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, mid_risk_customer):
        response = client.post("/predict", json=mid_risk_customer)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, mid_risk_customer):
        response = client.post("/predict", json=mid_risk_customer)
        data = response.json()
        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "risk_explanation" in data

    def test_predict_probability_is_between_0_and_1(self, mid_risk_customer):
        response = client.post("/predict", json=mid_risk_customer)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_class_is_binary(self, mid_risk_customer):
        response = client.post("/predict", json=mid_risk_customer)
        assert response.json()["predicted_class"] in (0, 1)

    def test_predict_explanation_is_nonempty_string(self, mid_risk_customer):
        response = client.post("/predict", json=mid_risk_customer)
        explanation = response.json()["risk_explanation"]
        assert isinstance(explanation, str) and len(explanation) > 10

    def test_high_risk_customer_gets_class_1(self, high_risk_customer):
        """Strongly churny customer should be predicted as churn (class 1)."""
        response = client.post("/predict", json=high_risk_customer)
        data = response.json()
        assert data["predicted_class"] == 1, (
            f"Expected predicted_class=1 for high-risk customer, "
            f"got {data['predicted_class']} (prob={data['churn_probability']})"
        )

    def test_low_risk_customer_gets_class_0(self, low_risk_customer):
        """Highly engaged customer should be predicted as retain (class 0)."""
        response = client.post("/predict", json=low_risk_customer)
        data = response.json()
        assert data["predicted_class"] == 0, (
            f"Expected predicted_class=0 for low-risk customer, "
            f"got {data['predicted_class']} (prob={data['churn_probability']})"
        )

    def test_high_risk_probability_exceeds_low_risk(self, high_risk_customer, low_risk_customer):
        """Churn probability must be higher for the at-risk customer."""
        high_prob = client.post("/predict", json=high_risk_customer).json()["churn_probability"]
        low_prob  = client.post("/predict", json=low_risk_customer).json()["churn_probability"]
        assert high_prob > low_prob

    def test_explanation_mentions_risk_level(self, mid_risk_customer):
        """Explanation should include a risk label."""
        explanation = client.post("/predict", json=mid_risk_customer).json()["risk_explanation"]
        assert any(word in explanation for word in ("High", "Medium", "Low"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Input validation (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_required_field_returns_422(self, mid_risk_customer):
        payload = mid_risk_customer.copy()
        del payload["recency_days"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_recency_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "recency_days": -5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_loyalty_tier_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "loyalty_tier": "Platinum"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_city_tier_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "city_tier": "Tier9"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_group_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "age_group": "100+"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_discount_above_1_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "avg_discount_pct_180d": 1.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rating_below_1_returns_422(self, mid_risk_customer):
        payload = {**mid_risk_customer, "avg_rating_180d": 0.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – /batch_predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, high_risk_customer, low_risk_customer, mid_risk_customer):
        payload = {"customers": [high_risk_customer, low_risk_customer, mid_risk_customer]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self, high_risk_customer, low_risk_customer):
        payload = {"customers": [high_risk_customer, low_risk_customer]}
        data = client.post("/batch_predict", json=payload).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_each_prediction_has_all_fields(self, mid_risk_customer):
        payload = {"customers": [mid_risk_customer]}
        predictions = client.post("/batch_predict", json=payload).json()["predictions"]
        for pred in predictions:
            assert "churn_probability" in pred
            assert "predicted_class" in pred
            assert "risk_explanation" in pred

    def test_batch_preserves_order(self, high_risk_customer, low_risk_customer):
        """First customer (high risk) should have a higher probability than second (low risk)."""
        payload = {"customers": [high_risk_customer, low_risk_customer]}
        preds = client.post("/batch_predict", json=payload).json()["predictions"]
        assert preds[0]["churn_probability"] > preds[1]["churn_probability"]

    def test_batch_empty_list_returns_422(self):
        response = client.post("/batch_predict", json={"customers": []})
        assert response.status_code == 422

    def test_single_customer_batch(self, mid_risk_customer):
        payload = {"customers": [mid_risk_customer]}
        data = client.post("/batch_predict", json=payload).json()
        assert data["total"] == 1

    def test_batch_invalid_customer_returns_422(self, mid_risk_customer):
        bad = {**mid_risk_customer, "loyalty_tier": "INVALID"}
        response = client.post("/batch_predict", json={"customers": [bad]})
        assert response.status_code == 422
