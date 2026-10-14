"""Download the raw transactions and write a PII-free parquet file.

Source: "Credit Card Transactions Fraud Detection Dataset" (Kaggle, kartik2112,
CC0), generated with the Sparkov simulator. Mirrored on Hugging Face at
NeerajCodz/creditCardFraudDetection. It covers 2019-01-01 to 2020-12-31.

The data is synthetic, but it has fields shaped like real PII (names, street
addresses, card numbers). They are dropped or hashed at the first step, the
way a real pipeline would handle them.
"""
from pathlib import Path
import urllib.request
import duckdb

BASE = "https://huggingface.co/datasets/NeerajCodz/creditCardFraudDetection/resolve/main/"
FILES = {"fraudTrain.csv": "train/us.csv", "fraudTest.csv": "test/us.csv"}
RAW = Path("data/raw")
OUT = Path("data/transactions.parquet")


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        if not (RAW / name).exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE + remote, RAW / name)


def clean() -> None:
    con = duckdb.connect()
    # Card numbers become a short hash. Names, street, zip, job and the
    # original row index are never carried forward.
    con.execute(f"""
        COPY (
            SELECT DISTINCT ON (trans_num)
                trans_num                                          AS txn_id,
                substr(sha256(CAST(cc_num AS VARCHAR)), 1, 12)     AS card_id,
                CAST(trans_date_trans_time AS TIMESTAMP)           AS ts,
                amt, category, merchant, gender,
                CAST(dob AS DATE)                                  AS dob,
                city_pop, lat, long, merch_lat, merch_long,
                is_fraud
            FROM read_csv(['{RAW}/fraudTrain.csv', '{RAW}/fraudTest.csv'], header = true)
            ORDER BY ts, txn_id
        ) TO '{OUT}' (FORMAT PARQUET)
    """)
    n, f, c = con.execute(f"SELECT count(*), sum(is_fraud), count(DISTINCT card_id) FROM '{OUT}'").fetchone()
    print(f"Wrote {OUT}: {n:,} transactions, {f:,} frauds, {c} cards")


if __name__ == "__main__":
    download()
    clean()
