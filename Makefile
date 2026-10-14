.PHONY: all data features test notebook ieee help

all: data features test notebook

data:        ## download raw data, drop PII, write data/transactions.parquet
	python src/download_data.py

features:    ## build point-in-time features with DuckDB -> data/features.parquet
	python src/build_features.py

test:        ## prove features never use future information
	python -m pytest -q tests

notebook:    ## Part 1: run the synthetic-data analysis end to end
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 fraud_early_warning.ipynb

ieee:        ## Part 2: real IEEE-CIS data (needs data/raw/ieee/train_*.csv from Kaggle)
	python src/prepare_ieee.py
	python src/build_features.py ieee
	python -m pytest -q tests/test_point_in_time_ieee.py
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 real_data_ieee.ipynb

help:        ## list targets
	@grep -E "^[a-z]+:.*##" Makefile | sed "s/:.*##/  -/"
