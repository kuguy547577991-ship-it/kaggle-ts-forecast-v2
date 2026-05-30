.PHONY: install train test lint clean

install:
	py -m pip install -e ".[dev]"

train:
	py pipelines/ts_forecast/pipeline.py

test:
	py -m pytest -v

lint:
	py -m ruff check .

clean:
	rm -rf data/processed/*
	rm -rf data/models/*
	rm -rf mlruns/*
