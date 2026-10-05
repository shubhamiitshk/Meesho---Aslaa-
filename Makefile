# Valmo RTO ML System — Meesho DICE Season 3
#
#   make all        regenerate models, metrics, and verify claims
#   make verify     check claims against the ledger
#
# On Windows without make: python pipeline.py all

PYTHON ?= python
RUNS   ?= 2000

.DEFAULT_GOAL := help
.PHONY: help all data models bench charts claims verify \
        predictions demo clean distclean

help:
	@echo "Valmo RTO ML System"
	@echo ""
	@echo "  make all         full pipeline: data -> models -> bench -> claims -> charts -> verify"
	@echo "  make data        generate the 100,000-order calibrated dataset"
	@echo "  make models      train and evaluate all three models"
	@echo "  make bench       latency and throughput benchmark (RUNS=$(RUNS))"
	@echo "  make charts      render architectural and performance figures"
	@echo "  make claims      rebuild the claims ledger"
	@echo "  make verify      assert claims match the ledger"
	@echo "  make predictions run sample batch predictions"
	@echo "  make demo        single-order scoring + telemetry audit demos"
	@echo "  make clean       remove generated charts and cache"
	@echo "  make distclean   remove models, data and cache"
	@echo ""

all: data models bench finance simulate ceiling provenance claims charts verify
	@echo ""
	@echo "Pipeline complete."

data: data/valmo_orders_dataset.csv

data/valmo_orders_dataset.csv: generate_valmo_dataset.py
	$(PYTHON) generate_valmo_dataset.py

models: models/rto_metrics.json

models/rto_metrics.json: train_valmo_models.py metrics_lib.py data/valmo_orders_dataset.csv
	$(PYTHON) train_valmo_models.py

models/benchmark_metrics.json: benchmark_serving.py predict.py models/rto_metrics.json
	$(PYTHON) benchmark_serving.py --runs $(RUNS)

bench: models/benchmark_metrics.json

finance: models/financial_model.json

models/financial_model.json: financial_model.py
	$(PYTHON) financial_model.py

simulate: models/intervention_sim.json

models/intervention_sim.json: intervention_sim.py
	$(PYTHON) intervention_sim.py

ceiling: models/signal_ceiling.json

models/signal_ceiling.json: signal_ceiling.py
	$(PYTHON) signal_ceiling.py

provenance: models/dataset_provenance.json

models/dataset_provenance.json: dataset_provenance.py
	$(PYTHON) dataset_provenance.py

charts: models/claims.json
	$(PYTHON) generate_deck_visuals.py

claims: models/claims.json

models/claims.json: claims.py models/rto_metrics.json models/uplift_metrics.json \
                   models/telemetry_metrics.json models/dataset_summary.json \
                   models/benchmark_metrics.json
	$(PYTHON) claims.py

verify:
	$(PYTHON) claims.py --verify

predictions: predict.py models/rto_metrics.json
	$(PYTHON) predict.py --batch data/sample_1000_orders.csv

demo:
	$(PYTHON) predict.py --single

clean:
	rm -rf charts/*.png
	rm -f data/sample_1000_orders_predictions.csv data/shap_summary_cache.joblib
	rm -rf __pycache__ .pytest_cache

distclean: clean
	rm -f models/*.joblib models/*.json models/feature_importance.csv
	rm -f data/valmo_orders_dataset.csv data/test_evaluation_preds.csv \
	      data/qini_eval_data.csv data/telemetry_eval_sample.csv \
	      data/cv_fold_metrics.csv data/shap_summary_cache.joblib \
	      data/telemetry_score_thresholds.csv
