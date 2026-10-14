"""Prepare the real IEEE-CIS fraud data (Vesta e-commerce transactions).

Get the data from Kaggle (you must accept the competition rules first):
    kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee
    unzip data/raw/ieee/ieee-fraud-detection.zip -d data/raw/ieee
Only train_transaction.csv and train_identity.csv are needed (the Kaggle test
files have no labels). Data is not committed to this repo.

The data has no card or customer ID. A reconstructed account ID ("uid") is
built from three fields:
    card1 (card number fragment) + addr1 (billing region) + account start day
where account start day = transaction day - D1, and D1 is documented as
"days since the card/account started". This is a widely used approximation,
not a ground truth ID.
"""
from pathlib import Path
import duckdb

RAW = Path("data/raw/ieee")
OUT = Path("data/ieee_transactions.parquet")
VENDOR = ([f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)] + [f"M{i}" for i in range(1, 10)])


def prepare() -> None:
    con = duckdb.connect()
    vendor_cols = ", ".join(f"t.{c}" for c in VENDOR)
    con.execute(f"""
        COPY (
            SELECT
                CAST(t.TransactionID AS VARCHAR)                              AS txn_id,
                -- reconstructed account id
                t.card1 || '_' || coalesce(CAST(t.addr1 AS VARCHAR), 'na') || '_' ||
                    coalesce(CAST(floor(t.TransactionDT / 86400) - t.D1 AS VARCHAR), 'na') AS card_id,
                -- TransactionDT is seconds from an undisclosed start; 2017-12-01 is a
                -- common assumption. Only relative time matters here.
                TIMESTAMP '2017-12-01' + to_seconds(t.TransactionDT)          AS ts,
                t.TransactionAmt                                              AS amt,
                t.ProductCD                                                   AS product,
                t.card1, t.card4 AS card_network, t.card6 AS card_type, t.addr1,
                t.dist1,
                t.P_emaildomain                                               AS email_domain,
                t.R_emaildomain                                               AS recipient_email_domain,
                i.DeviceType                                                  AS device_type,
                i.DeviceInfo                                                  AS device_info,
                (i.TransactionID IS NOT NULL)::INT                            AS has_identity,
                {vendor_cols},
                t.isFraud                                                     AS is_fraud
            FROM read_csv('{RAW}/train_transaction.csv', header = true, sample_size = -1) t
            LEFT JOIN read_csv('{RAW}/train_identity.csv', header = true, sample_size = -1) i
                   USING (TransactionID)
            ORDER BY ts, txn_id
        ) TO '{OUT}' (FORMAT PARQUET)
    """)
    n, f, u = con.execute(f"SELECT count(*), sum(is_fraud), count(DISTINCT card_id) FROM '{OUT}'").fetchone()
    print(f"Wrote {OUT}: {n:,} transactions, {f:,} frauds, {u:,} reconstructed accounts")


if __name__ == "__main__":
    prepare()
