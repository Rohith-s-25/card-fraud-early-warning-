"""Build point-in-time feature tables with DuckDB.

    python src/build_features.py          # synthetic Sparkov data
    python src/build_features.py ieee     # real IEEE-CIS data
"""
from pathlib import Path
import sys
import time
import duckdb

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
DATASETS = {
    "sparkov": ("features.sql", "data/transactions.parquet", "data/features.parquet"),
    "ieee": ("features_ieee.sql", "data/ieee_transactions.parquet", "data/ieee_features.parquet"),
}


def feature_query(source: str, sql_file: str = "features.sql") -> str:
    """Return the feature SQL for any table name or read_parquet(...) expression."""
    return (SQL_DIR / sql_file).read_text().replace("{source}", source)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "sparkov"
    sql_file, src, out = DATASETS[name]
    t0 = time.time()
    con = duckdb.connect()
    query = feature_query(f"read_parquet('{src}')", sql_file)
    con.execute(f"COPY ({query}) TO '{out}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"Wrote {out}: {n:,} rows in {time.time() - t0:.0f}s")
