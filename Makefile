.PHONY: install lint test smoke repro demo clean

install:
	pip install -e ".[dev]"

lint:
	ruff check kernelfork baselines scripts tests
	ruff format --check kernelfork baselines scripts tests

test:
	pytest -q

PYTHON ?= python3

smoke:
	$(PYTHON) -m kernelfork.train_grpo --arm kernelfork --fixture tests/mock_data/problem.json --seed 0 --forks 2 --output results/MOCK/smoke_step.json
	@echo "MOCK smoke step wrote results/MOCK/smoke_step.json"

repro:
	bash scripts/repro_all_arms.sh

demo:
	$(PYTHON) scripts/render_demo.py

clean:
	rm -rf .pytest_cache **/__pycache__ results/MOCK/*.json
