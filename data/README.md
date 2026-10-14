# Data

Nothing in this folder is committed.

**Part 1 (synthetic):** `make data` downloads the raw CSVs (~500 MB) into `data/raw/` and writes `data/transactions.parquet`; `make features` writes `data/features.parquet`. Source: Credit Card Transactions Fraud Detection Dataset (Kaggle, kartik2112, CC0), mirrored on Hugging Face at NeerajCodz/creditCardFraudDetection.

**Part 2 (real):** download IEEE-CIS Fraud Detection from Kaggle (accept the competition rules first) and put `train_transaction.csv` and `train_identity.csv` in `data/raw/ieee/`. `make ieee` does the rest.
