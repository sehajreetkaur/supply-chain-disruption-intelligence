# Supply Chain Disruption Intelligence Platform

> An end-to-end AI-powered supply chain analytics system modelled after EY's Supply Chain Intelligence Platform — combining predictive machine learning, SHAP explainability, and an executive Tableau dashboard to detect and prevent shipment delays before they occur.

---

## Live Demo

| Deliverable | Link |
|---|---|
| Tableau Dashboard | [View Live Dashboard](#) |
| GitHub Repository | [github.com/sehajreetkaur/supply-chain-disruption-intelligence](#) |

> Replace `#` with your actual URLs after publishing

---

## Project Overview

Global supply chains face increasing disruption risk. Late deliveries cost organisations millions annually through SLA penalties, expediting fees, and lost business. This project simulates how a consulting firm like EY would build an end-to-end intelligence platform for a logistics client — predicting delays before they happen and translating model outputs into actionable business recommendations.

**Business Question:** Which shipments are at risk of delay, why, and what does it cost if we miss them?

**Dataset:** SCMS Delivery History — 10,000+ real supply chain shipments from the USAID Supply Chain Management System, covering pharmaceutical and medical supply logistics across 40+ countries.

---

## Project Architecture

```
Raw Data (SCMS)
     │
     ▼
Phase 1 — Exploratory Data Analysis
     │  8 business charts · vendor analysis · cost profiling
     │
     ▼
Phase 2 — Feature Engineering
     │  6 feature groups · composite risk score · vendor risk tiers
     │
     ▼
Phase 3 — XGBoost Model + SHAP
     │  AUC-ROC 0.873 · operational risk scoring · EUR impact
     │
     ▼
Phase 4 — Tableau Dashboard
        Scatter · trend · vendor scorecard · live URL
```

---

## Key Results

| Metric | Value |
|---|---|
| Model | XGBoost (Extreme Gradient Boosting) |
| AUC-ROC | **0.873** |
| Recall (delay detection) | **80.3%** |
| F1 Score | 0.5992 |
| CV AUC 5-fold | 0.87 ± 0.02 |
| Baseline (Random Forest) | 82.8% accuracy |
| Top delay driver | `composite_risk_score` (SHAP = 0.152) |
| 2nd driver | `country_delay_rate` (SHAP = 0.113) |
| 3rd driver | `order_month` (SHAP = 0.108) |
| Estimated annual saving | EUR 2M+ (at €5,000/missed delay) |
| Critical Risk shipments | Flagged automatically |

---

## What Makes This Different

Most ML projects stop at model accuracy. This one goes further:

**1. Business cost framing** — Every model decision is evaluated in euros, not just accuracy percentages. A false negative (missed delay) costs €5,000. A false positive (false alarm) costs €200. The model is optimised for this asymmetry.

**2. SHAP explainability** — Every single prediction is explained in business language using SHAP (SHapley Additive exPlanations). A logistics manager can see exactly why a shipment was flagged as high risk — not just a black-box score.

**3. Operational risk tiers** — The model outputs are translated into 4 actionable tiers (Low / Medium / High / Critical) that a supply chain team can act on immediately without understanding ML.

**4. Consulting narrative** — The project is framed as an EY client engagement, not a data science exercise. Every chart answers a business question. Every finding has a recommendation.

---

## Feature Engineering (Phase 2)

6 feature groups were engineered from raw shipment data:

| Group | Features | Business Logic |
|---|---|---|
| Temporal | order_month, quarter, weekend_flag, lead_time | Seasonality and time pressure signals |
| Cost/Value | log_freight_cost, freight_vs_median_ratio, high_cost_flag | Financial anomaly detection |
| Vendor Risk | vendor_delay_rate, vendor_shipment_count, risk_tier | Historical supplier reliability |
| Country Risk | country_delay_rate, country_shipment_vol | Geographic risk encoding |
| Mode Encoding | mode_delay_rate, one-hot dummies | Shipment mode risk profile |
| Interactions | composite_risk_score, vendor_mode_combo | Combined risk signals |

The **composite_risk_score** (40% vendor + 35% country + 25% mode) became the single most predictive feature in the entire model.

---

## Model Performance

```
XGBOOST RESULTS
══════════════════════════════════════════════════
  Accuracy   : 76.8%  (baseline was 82.8%)
  AUC-ROC    : 0.8731
  Precision  : 47.8%
  Recall     : 80.3%
  F1         : 0.5992
  CV AUC     : 0.87 +/- 0.02
══════════════════════════════════════════════════

Note: Accuracy is lower than baseline because the model
is optimised for Recall — catching 80% of real delays
is more valuable than high accuracy on a skewed dataset.
In supply chain, a missed delay costs 25x more than a
false alarm.
```

---

## Tableau Dashboard

3-chart executive dashboard built in Tableau Desktop:

**Chart 1 — Freight Cost vs Delay Risk Scatter**
Every vendor plotted by average freight cost (X) vs delay probability (Y). Colour-coded by ML risk tier. Reference lines create 4 quadrants — top right = high cost AND high risk = immediate action required.

**Chart 2 — Monthly Delay Rate Trend (2006–2016)**
10-year timeline showing delay rate evolution with freight cost overlay. Reveals seasonal patterns and structural disruption periods.

**Chart 3 — Vendor Risk Scorecard**
Top 15 vendors ranked by delay rate. Heat-mapped bars show which suppliers are systematically above-average risk. Directly actionable for contract renegotiation.

---

## Tech Stack

```
Data & ML              Visualisation       Infrastructure
─────────────────      ─────────────────   ─────────────────
Python 3.9             Tableau Desktop     GitHub
pandas · numpy         Matplotlib          Jupyter Notebooks
scikit-learn           Seaborn             VS Code
XGBoost 2.1            Plotly
SHAP 0.49
joblib
```

---

## Repository Structure

```
supply-chain-disruption-intelligence/
│
├── data/
│   ├── SCMS_Delivery_History.csv        ← raw dataset (download from Kaggle)
│   ├── SCMS_cleaned.csv                 ← after Phase 1 cleaning
│   ├── SCMS_features.csv                ← after Phase 2 engineering
│   ├── SCMS_risk_scores.csv             ← ML model output
│   └── tableau_dashboard_data.csv       ← Tableau-ready combined file
│
├── notebooks/
│   ├── 01_eda.ipynb                     ← Phase 1: EDA + 8 business charts
│   ├── 02_feature_engineering.ipynb     ← Phase 2: 6 feature groups
│   └── 03_model_xgboost.ipynb           ← Phase 3: XGBoost + SHAP
│
├── outputs/
│   ├── figures/                         ← 17 charts auto-saved
│   └── models/
│       └── xgboost_delay_model.pkl      ← trained model
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── model.py
│
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/sehajreetkaur/supply-chain-disruption-intelligence
cd supply-chain-disruption-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to kaggle.com → search "Supply Chain Shipment Pricing Data"
# Download and place CSV in data/ folder as SCMS_Delivery_History.csv

# 4. Run notebooks in order
jupyter notebook
# Open: 01_eda.ipynb → 02_feature_engineering.ipynb → 03_model_xgboost.ipynb
```

---

## Business Recommendations

Based on the model findings, three immediate actions for the client:

**1. Vendor performance review**
MSD Latin America, Acouns Nigeria, and Janssen Sciences show 85-100% delay rates. Initiate contract review and SLA renegotiation immediately. Estimated saving: €500K-1M annually.

**2. Shipment mode optimisation**
Air freight is being used for shipments where ocean/road would deliver on time. The composite risk score shows mode choice is the 3rd strongest delay predictor. A mode optimisation programme could reduce freight costs by 30-40%.

**3. Proactive monitoring system**
Deploy the XGBoost model at order creation time. Flag Critical Risk shipments (>80% delay probability) for proactive intervention. With 80% recall, the model catches 4 out of 5 delays before they happen.

---

## About

**Sehajreet Kaur**
MSc Business Analytics & Data Science
Kühne Logistics University · Hamburg, Germany

- LinkedIn: [linkedin.com/in/sehajreet-kaur](https://linkedin.com/in/sehajreet-kaur)
- GitHub: [github.com/sehajreetkaur](https://github.com/sehajreetkaur)
- Email: Sehajreetkaur153@gmail.com

---

## Acknowledgements

Dataset: SCMS Delivery History — Supply Chain Management System (USAID)
Available on Kaggle: Supply Chain Shipment Pricing Data

Inspired by EY's Supply Chain Intelligence Platform and their framework for predictive analytics in logistics transformation engagements.

---

*This project was built as a portfolio piece demonstrating end-to-end data science capability for enterprise supply chain analytics roles in Germany.*
