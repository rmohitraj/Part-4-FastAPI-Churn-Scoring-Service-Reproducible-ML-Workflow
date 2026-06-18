# Monitoring Plan — D2C Churn Scoring Service

This document describes what should be tracked **after** the churn API is deployed to production, who is responsible, and what triggers remedial action.

---

## 1. Data Drift (Input Feature Distributions)

Concept drift and data drift are the two most common causes of silent model degradation. We monitor input feature distributions weekly and compare against the training baseline.

| Feature | Metric | Alert Threshold |
|---|---|---|
| `recency_days` | KS-test p-value vs. training distribution | p < 0.05 |
| `monetary_180d` | Population Stability Index (PSI) | PSI > 0.2 |
| `sessions_30d` | Mean shift (z-score) | \|z\| > 3 |
| `loyalty_tier` (categorical) | Chi-squared test | p < 0.05 |
| `acquisition_channel` | Chi-squared test | p < 0.05 |
| All numeric features | PSI on 10-bin histogram | PSI > 0.25 |

**Tool recommendation**: Evidently AI or WhyLogs for automated drift reports.  
**Cadence**: Weekly batch comparison; real-time dashboarding for high-volume days.

---

## 2. Prediction Distribution (Model Output)

Shifts in prediction outputs indicate model drift even when inputs look stable.

| Metric | Baseline (validation set) | Alert Threshold |
|---|---|---|
| Mean `churn_probability` | ~0.38 | Deviation > ±0.10 |
| % predictions with class = 1 | ~38% | Deviation > ±8 pp |
| Score distribution (decile spread) | Uniform ± noise | Significant collapse/shift |
| P(churn) > 0.7 rate | ~15% | Deviation > ±5 pp |

**Cadence**: Daily rolling 7-day window. Dashboard alert if thresholds are crossed on two consecutive days.

---

## 3. Business Outcomes (Actual vs. Predicted Churn)

Model utility is only confirmed when predictions lead to correct business decisions.

| KPI | How to Measure | Frequency |
|---|---|---|
| **Actual churn rate** | Count of customers predicted churn=1 who actually churned within 60 days | Monthly |
| **Precision** on retained cohort | Of customers scored ≥ 0.25, what fraction actually churned? | Monthly |
| **Recall** on churned cohort | Of customers who churned, what fraction did the model flag? | Monthly |
| **Retention campaign conversion** | % of flagged customers who responded to retention offer | Per campaign |
| **CLTV saved** | Revenue from retained customers attributed to model-driven outreach | Quarterly |

**Note**: Requires a feedback loop — the CRM must log 60-day post-prediction churn outcomes back to the monitoring database.

---

## 4. API Performance & Error Monitoring

Infrastructure reliability is part of model health.

| Metric | Tool | Alert |
|---|---|---|
| HTTP 5xx error rate | Datadog / Prometheus | > 1% of requests in 5-min window |
| HTTP 422 validation error rate | Application logs | > 5% (indicates bad upstream data) |
| Latency (p95) | Grafana | > 500 ms |
| Request volume anomalies | CloudWatch / Datadog | Drop > 50% vs. same weekday |
| Model load failures | Structured logs (ERROR level) | Any occurrence → PagerDuty |

**Structured log fields** every `/predict` call should emit:
```json
{
  "timestamp": "ISO-8601",
  "customer_id": "...",
  "churn_probability": 0.72,
  "predicted_class": 1,
  "latency_ms": 23,
  "model_version": "1.0.0"
}
```

---

## 5. Retraining Triggers

Retraining should be triggered by **any** of the following:

| Trigger | Condition |
|---|---|
| **Scheduled** | Every 90 days regardless of drift status |
| **Metric degradation** | ROC-AUC on fresh labelled data drops below 0.82 |
| **Prediction drift** | Mean predicted churn probability shifts > 10 pp from baseline for 7+ consecutive days |
| **Input drift** | PSI > 0.25 for two or more top-5 features simultaneously |
| **Business event** | Major product change, pricing restructuring, or new acquisition channel launched |
| **Data pipeline change** | Schema change in upstream CRM/orders data that affects feature definitions |

**Retraining process**:
1. Pull fresh snapshot data from the data warehouse.
2. Re-run `churn_model.ipynb` (or equivalent pipeline script).
3. Validate new model on holdout set; confirm ROC-AUC ≥ 0.85.
4. Run A/B test for 2 weeks: 50% traffic to old model, 50% to new.
5. Promote new model if business KPIs improve; roll back if not.
6. Archive old `model.pkl` with version tag and date.

---

## 6. Responsible Monitoring Practices

- **Fairness monitoring**: Monthly audit of predicted churn rates disaggregated by `age_group` and `city_tier`. Flag if any segment's churn prediction rate deviates > 15 pp from the overall rate without a corresponding real-world explanation.
- **Privacy**: Log only customer IDs (not PII) in monitoring systems. Ensure logs comply with DPDP Act (India) / GDPR as applicable.
- **Audit trail**: Every model prediction should be logged with a `model_version` tag to support retrospective analysis.

---

*Last reviewed: 2025-09-30 | Owner: ML Platform Team*
