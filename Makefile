PY := .venv/bin/python
PIP := .venv/bin/pip
CONFIG ?= configs/centralized_mitbih.yaml

.PHONY: setup data sanity test clean clean-results \
        train-central train-fed sweep-noniid sweep-dp crossdataset \
        figures paper all

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PY) -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

data:
	$(PY) scripts/preprocess_data.py --dataset all

sanity:
	$(PY) scripts/sanity_check_data.py

test:
	$(PY) -m pytest tests/

clean:
	rm -rf data/processed/ results/

clean-results:
	rm -rf results/

# ---- stubs for later phases ----

train-central:
	$(PY) scripts/run_experiment.py --config $(CONFIG) --mode centralized

train-fed:
	$(PY) scripts/run_experiment.py --config $(CONFIG) --mode federated

sweep-noniid:
	@echo "Phase 4 stub: non-IID sweep not yet implemented"

sweep-dp:
	@echo "Phase 5 stub: DP sweep not yet implemented"

crossdataset:
	@echo "Phase 6 stub: cross-dataset validation not yet implemented"

figures:
	@echo "Phase 7 stub: figure generation not yet implemented"

paper:
	@echo "Phase 8 stub: paper assembly not yet implemented"

all:
	@echo "Phase 0+1 ready: run 'make setup && make data && make sanity && make test'"
