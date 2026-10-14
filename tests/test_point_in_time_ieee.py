"""Leakage tests for the real-data (IEEE-CIS) feature pipeline. Same idea as
test_point_in_time.py: features must only ever use earlier transactions."""
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_features import feature_query  # noqa: E402

SRC = "read_parquet('data/ieee_transactions.parquet')"
SQL = "features_ieee.sql"
COLS = ["n_txn_1h", "n_txn_24h", "n_txn_7d", "n_prior_txn", "amt_sum_24h", "acct_avg_amt_30d",
        "hours_since_prev", "new_product_for_acct", "new_email_for_acct", "card1_txn_1h", "card1_txn_30d"]

pytestmark = pytest.mark.skipif(not Path("data/ieee_transactions.parquet").exists(),
                                reason="IEEE-CIS data not prepared")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    # Sample whole card1 groups, so both account-level and card1-level windows are complete.
    c.execute(f"""
        CREATE TABLE sample AS SELECT * FROM {SRC}
        WHERE card1 IN (SELECT card1 FROM {SRC} GROUP BY card1 HAVING count(*) BETWEEN 200 AND 2000
                        ORDER BY card1 LIMIT 15)
    """)
    return c


def test_one_row_per_transaction(con):
    f = con.sql(feature_query("sample", SQL)).df()
    assert len(f) == con.execute("SELECT count(*) FROM sample").fetchone()[0]
    assert f.txn_id.is_unique


def test_future_changes_do_not_affect_past_features(con):
    cutoff, last = con.execute("SELECT CAST(quantile_disc(ts, 0.5) AS VARCHAR), max(ts) FROM sample").fetchone()
    con.execute(f"""
        CREATE OR REPLACE TABLE poisoned AS
        SELECT * REPLACE (CASE WHEN ts >= TIMESTAMP '{cutoff}' THEN amt * 100 ELSE amt END AS amt,
                          CASE WHEN ts >= TIMESTAMP '{cutoff}' THEN 'X' ELSE product END AS product)
        FROM sample
        UNION ALL
        SELECT * FROM (
            SELECT * REPLACE (txn_id || '_fake' AS txn_id,
                              TIMESTAMP '{last}' + (ts - (SELECT min(ts) FROM sample)) + INTERVAL 1 DAY AS ts)
            FROM sample WHERE ts < TIMESTAMP '{cutoff}' LIMIT 3000)
    """)
    q = "SELECT * FROM ({}) WHERE ts < TIMESTAMP '" + cutoff + "' ORDER BY txn_id"
    clean = con.sql(q.format(feature_query("sample", SQL))).df()
    dirty = con.sql(q.format(feature_query("poisoned", SQL))).df()
    assert len(clean) == len(dirty) > 0
    pd.testing.assert_frame_equal(clean[COLS], dirty[COLS])


def test_matches_brute_force_recomputation(con):
    raw = con.execute("SELECT * FROM sample ORDER BY ts").df()
    feats = con.sql(feature_query("sample", SQL)).df().set_index("txn_id")
    for _, r in raw.sample(300, random_state=0).iterrows():
        past = raw[(raw.card_id == r.card_id) & (raw.ts < r.ts)]
        f = feats.loc[r.txn_id]
        in_24h = past[past.ts >= r.ts - pd.Timedelta(hours=24)]
        assert f.n_txn_24h == len(in_24h)
        assert np.isclose(f.amt_sum_24h, in_24h.amt.sum())
        assert f.n_prior_txn == len(past)
        c1 = raw[(raw.card1 == r.card1) & (raw.ts < r.ts) & (raw.ts >= r.ts - pd.Timedelta(hours=1))]
        assert f.card1_txn_1h == len(c1)
        if len(past):
            assert np.isclose(f.hours_since_prev, (r.ts - past.ts.max()).total_seconds() / 3600)
            if pd.notna(r.email_domain):
                assert f.new_email_for_acct == int(not (past.email_domain == r.email_domain).any())
