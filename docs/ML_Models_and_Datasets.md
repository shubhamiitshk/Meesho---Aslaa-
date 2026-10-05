# ML Models and Data — Prototype Reference

## Scope

This repository demonstrates a local prototype for a case-study proposal. It is not integrated with Meesho or Valmo. It contains no Valmo order-level data, production telemetry, operational labels, or approved live inference endpoint. Metrics and local latency in this document must not be used as production claims.

## Data sources

| Input | Origin | Appropriate interpretation |
|---|---|---|
| Case-pack payment mix, RTO assumptions, unit costs, forward cost split, distance pattern | DICE Season 3 case brief | Assignment assumptions, not a current audited Valmo baseline |
| India Post pincode/geospatial reference data | Open-source registry used by `bharataddress` | Geographic reference only; not transaction or delivery outcome data |
| 100,000 synthetic order rows | `generate_valmo_dataset.py` | Generated outcomes calibrated to selected case-pack marginals; not independent evidence |
| Simulated treatment records and labels | Training/simulation pipeline | Synthetic evaluation environment; does not prove intervention impact |
| Simulated attempt telemetry | Training pipeline | Demonstration labels; not real rider behavior or fraud prevalence |
| Meesho prospectus metrics | [NSE-filed prospectus](https://nsearchives.nseindia.com/corporate/ADV_INE0VDM01015_09DEC2025.pdf) | Marketplace-wide delivery success and shipped-order share; not Valmo RTO rates |

The prospectus defines its delivery-success measure as orders delivered to the consumer regardless of subsequent product return. This is not a cause-coded Valmo RTO metric. The code in `meesho_disclosure.py` retains the reported success values and does not derive RTO complements.

## Prototype components

`train_valmo_models.py` fits a tabular risk-ranking model, an uplift model against generated treatment outcomes, and an anomaly-detection model against generated attempt telemetry. `predict.py` loads local artifacts for single and batch demonstrations. `address_tokenizer.py` extracts simple address features. `intervention_sim.py` runs an exploratory scenario on the synthetic order book. `financial_model.py` calculates conditional scenario economics from explicit assumptions.

The address parser and DIGIPIN-related reference work can demonstrate software mechanics. They do not establish address correctness, feasibility of integrating with Valmo, rider suitability, privacy authorization, or causal delivery improvement.

## How to read model metrics

- Train/test metrics measure fit to generated labels under the configured synthetic split. They do not estimate real-world accuracy.
- Calibration scores reflect simulator-generated outcomes and labels. They do not show probability calibration on live orders.
- Uplift/Qini values are calculated from simulated randomized treatment. Random assignment inside a simulator does not constitute a field experiment.
- Fraud/anomaly metrics depend on generated anomaly classes and prevalence. They do not establish actual fraud rates or safe automated action thresholds.
- AUC ceiling and feature associations inherit the data-generating process; they are not statements about real Valmo predictability.
- Benchmark latency describes the machine, libraries, warm-up, batch shape, and code path used. It is not an end-to-end production SLA.

## Reproduction

Install dependencies from `requirements.txt`, then run:

```bash
python generate_valmo_dataset.py
python train_valmo_models.py
python intervention_sim.py
python financial_model.py
python generate_deck_visuals.py
```

These commands regenerate synthetic artifacts. They do not replace operational validation. Keep generated outputs labeled as synthetic in tables, charts, HTML, speaker notes, and any future export.

## Requirements before deployment could be considered

1. Authorized, privacy-reviewed, de-identified Valmo event data with stable outcome definitions and representative time periods.
2. Leakage review, temporal and geographic holdouts, segment-level calibration, drift monitoring, and comparison to simple operational baselines.
3. A prospectively designed controlled intervention test with power/sample-size analysis and guardrails.
4. Human review and appeal for any attempt anomaly that can affect pay or access to work.
5. Security, privacy, access-control, data-retention, incident-response, integration, and operational ownership review.
6. Full cost validation and finance approval; the current scenario is not deployment evidence.
