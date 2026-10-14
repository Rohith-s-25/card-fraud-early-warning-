"""Tests that the feature pipeline never uses information from the future.

A fraud model scores a transaction the moment it happens. If any feature
quietly includes later transactions, offline results look great and the
live model fails. These tests guard against that.
"""
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_features import feature_query  # noqa: E402

TXNS = "read_parquet('data/transactions.parquet')"
pytestmark = pytest.mark.skipif(not Path("data/transactions.parquet").exists(),
                                reason="Synthetic data not prepared (run `make data`)")

FEATURE_COLS = ["n_txn_1h", "n_txn_24h", "n_txn_7d", "amt_sum_24h", "card_avg_amt_30d",
                "amt_vs_30d_avg", "category_share_90d", "first_time_category",
                "first_time_merchant", "hours_since_prev", "card_tenure_days"]


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    # A sample of 25 cards keeps the tests fast. Features are per card, so a
    # card's features do not depend on which other cards are loaded.
    c.execute(f"""
        CREATE TABLE sample AS
        SELECT * FROM {TXNS}
        WHERE card_id IN (SELECT DISTINCT card_id FROM {TXNS} ORDER BY card_id LIMIT 25)
    """)
    return c


def test_one_row_per_transaction(con):
    feats = con.sql(feature_query("sample")).df()
    n_src = con.execute("SELECT count(*) FROM sample").fetchone()[0]
    assert len(feats) == n_src
    assert feats.txn_id.is_unique


def test_future_changes_do_not_affect_past_features(con):
    """Rewrite everything after a cutoff date. Features before it must not move."""
    # cutoff in the middle of the data; fake rows are pushed past the last real row
    cutoff, last = con.execute("SELECT CAST(quantile_disc(ts, 0.5) AS VARCHAR), max(ts) FROM sample").fetchone()
    con.execute(f"""
        CREATE OR REPLACE TABLE poisoned AS
        SELECT * REPLACE (
            CASE WHEN ts >= TIMESTAMP '{cutoff}' THEN amt * 100 ELSE amt END AS amt,
            CASE WHEN ts >= TIMESTAMP '{cutoff}' THEN 'travel' ELSE category END AS category)
        FROM sample
        UNION ALL
        -- plus 5,000 fake future transactions
        SELECT * FROM (
            SELECT * REPLACE (txn_id || '_fake' AS txn_id, TIMESTAMP '{last}' + (ts - (SELECT min(ts) FROM sample)) + INTERVAL 1 DAY AS ts)
            FROM sample WHERE ts < TIMESTAMP '{cutoff}' LIMIT 5000)
    """)
    q = "SELECT * FROM ({}) WHERE ts < TIMESTAMP '" + cutoff + "' ORDER BY txn_id"
    clean = con.sql(q.format(feature_query("sample"))).df()
    dirty = con.sql(q.format(feature_query("poisoned"))).df()
    assert len(clean) == len(dirty) > 0
    pd.testing.assert_frame_equal(clean[FEATURE_COLS], dirty[FEATURE_COLS])


def test_matches_brute_force_recomputation(con):
    """Recompute features for random rows in plain pandas, using ts < now only."""
    raw = con.execute("SELECT * FROM sample ORDER BY ts").df()
    feats = con.sql(feature_query("sample")).df().set_index("txn_id")
    rows = raw.sample(300, random_state=0)
    for _, r in rows.iterrows():
        past = raw[(raw.card_id == r.card_id) & (raw.ts < r.ts)]
        f = feats.loc[r.txn_id]
        in_24h = past[past.ts >= r.ts - pd.Timedelta(hours=24)]
        in_30d = past[past.ts >= r.ts - pd.Timedelta(days=30)]
        assert f.n_txn_24h == len(in_24h)
        assert np.isclose(f.amt_sum_24h, in_24h.amt.sum())
        if len(in_30d):
            assert np.isclose(f.card_avg_amt_30d, in_30d.amt.mean())
        assert f.first_time_category == int(not (past.category == r.category).any())
        assert f.first_time_merchant == int(not (past.merchant == r.merchant).any())
        if len(past):
            assert np.isclose(f.hours_since_prev, (r.ts - past.ts.max()).total_seconds() / 3600)
