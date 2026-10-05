# Project Aslaah: Valmo Intelligent RTO Reduction System

> **Meesho DICE Challenge Season 3 | Detailed Submission Round (Technical Codebase)**  
> Technical implementation, predictive ML models, geospatial address parsing, and unit economic simulation suite for mitigating Return-to-Origin (RTO) in Valmo's last-mile logistics.

---

## Technical Overview

In Indian e-commerce, Return-to-Origin (RTO) represents a severe logistics friction point—especially across Bharat (Tier-2/3+) markets. For every failed Cash-on-Delivery (COD) shipment, logistics platforms incur not only the forward transit spend but also the full reverse logistics penalty (**₹50 forward + ₹120 reverse = ₹170 total loss per failed order**).

**Project Aslaah** provides a production-grade algorithmic and simulation suite backing our proposal:

1. **Upstream Address Intelligence & Geocoding**: DIGIPIN 3.82 m national spatial grid and Indic address tokenizer to resolve Bharat address ambiguity before dispatch.
2. **Pre-Dispatch ML Risk Classification**: Calibrated LightGBM and XGBoost models for risk scoring and targeted customer pre-alerts.
3. **Last-Mile Pilot Protection & Anomaly Auditing**: Tripartite edge telemetry gateway using Isolation Forest (98.6% genuine precision floor) paired with distance-banded compensation to ensure pilot fairness.
4. **Doorstep Rescue Cascade & Causal Uplift**: Causal Uplift X-Learner for dynamic QR payment switching, conversational WhatsApp Non-Delivery Report (NDR) recovery, and 72-hour hub staging to avoid reverse shipping.

---

## Technical Architecture & Core Modules

```
Aslaah/
├── train_valmo_models.py      # ML training pipeline (LightGBM, XGBoost, Isolation Forest, Uplift X-Learner)
├── predict.py                 # Real-time inference engine and low-latency serving verification
├── live_inference.py          # Interactive and scenario-based order scoring CLI
├── address_tokenizer.py       # Indic geocoding and DIGIPIN spatial address parser
├── financial_model.py         # Unit economics, EBITDA waterfall, sensitivity & breakeven models
├── benchmark_serving.py       # P99 latency and serving throughput benchmark
├── intervention_sim.py        # Doorstep rescue cascade simulation
├── signal_ceiling.py          # Bayes error rate and information-theoretic signal ceiling analysis
├── generate_deck_visuals.py   # High-resolution architectural and performance chart generator
├── generate_research_figures.py # Research and empirical analysis figure generator
├── dataset_provenance.py      # Verifier for postal dataset (26,711 pincodes, DIGIPIN grid)
├── pipeline.py                # End-to-end pipeline build & reproduction runner
├── claims.py                  # Automated claims ledger & consistency verification
├── data/                      # Calibrated datasets (100k synthetic order book, evaluations)
├── models/                    # Trained model weights (.joblib) and verified metric ledgers (.json)
├── charts/                    # Technical figures, calibration curves, ROC/PR, and SHAP plots
└── docs/                      # Technical architecture blueprints and benchmark documentation
```

---

## Technical Highlights & Verified Models

1. **Pre-Dispatch RTO Risk Classifier (`models/rto_risk_lightgbm_calibrated.joblib`)**:
   - Holdout ROC-AUC: 0.6552 | PR-AUC: 0.2571
   - Isotonic calibration cuts Brier error by 41.3% (0.2305 -> 0.1352), providing true posterior probabilities for economic decision thresholds.
   - Operating Lift: 1.75x lift at top-10% budget.

2. **Causal Uplift Engine (`models/doorstep_uplift_xlearner.joblib`)**:
   - Two-stage X-Learner estimating Conditional Average Treatment Effect (CATE) for doorstep UPI QR rescue.
   - Normalised AUUC = 0.1221 (p=0.005 against 200-permutation null).
   - Only fires coupons when predicted uplift clears unit cost hurdle (uplift >= 29.17%).

3. **Last-Mile Pilot Protection Gateway (`models/telemetry_isolation_forest.joblib`)**:
   - Unsupervised Isolation Forest evaluated on held-out attempt telemetry.
   - 98.6% genuine precision floor: protects honest delivery pilots with human-in-the-loop appeal pathways and zero automated pay deductions.

4. **Bharat Address & DIGIPIN Spatial Engine (`address_tokenizer.py`)**:
   - Native integration with India Post's 10-character DIGIPIN 3.82 m x 3.82 m bounding cells.
   - Full coverage across 26,711 Indian pincodes and 57,384 post offices.

---

## Reproducibility & Running the Pipeline

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed with dependencies:
```bash
pip install -r requirements.txt
```

### 2. Live Order Scoring & Inference
To test single-order prediction, batch scoring, and telemetry audit:
```bash
# Run real-time scoring demonstration
python predict.py

# Test representative scenario profiles
python live_inference.py --profile 1    # High-Risk Peri-Urban COD
python live_inference.py --profile 2    # Persuadable Doorstep Rescue Candidate
python live_inference.py --profile 3    # Low-Risk Metro Prepaid
```

### 3. Financial Modeling & Sensitivity Simulation
To run the full financial waterfall, breakeven analysis, and coupon budget reconciliation:
```bash
python financial_model.py
```

### 4. Regenerate Visual Charts & Graphs
To regenerate all architectural, model evaluation, and simulation figures in `charts/`:
```bash
python generate_deck_visuals.py
python generate_research_figures.py
```

### 5. Full Pipeline Build & Verification
To run the complete automated data generation, training, benchmarking, and verification pipeline:
```bash
python pipeline.py all
```
or check ledger claims directly:
```bash
python claims.py --verify
```

---

## Data & Modeling Methodology

- **Order Book & Simulation**: In strict accordance with the DICE Challenge brief, our order book (`data/valmo_orders_dataset.csv`, 100,000 orders) is calibrated against official case assumptions: 80% COD share, 20% Prepaid share, ₹50 forward freight, ₹120 reverse freight, and Tier-1/2/3 geographic distribution.
- **Geographic Infrastructure**: Address tokenization leverages real Indian logistics infrastructure, spanning 26,711 India Post pincodes, 57,384 post offices, and India Post's official **DIGIPIN** 3.82 m national spatial grid.
- **Policy Integrity**: All rescue interventions and pilot incentives are governed by strict pre-registered guardrails, zero automated deductions for riders, and human appeal pathways.
