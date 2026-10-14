-- Point-in-time behavioural features for the real IEEE-CIS data.
-- Same rules as features.sql: each feature uses only the SAME account's
-- EARLIER transactions (EXCLUDE GROUP drops the current row and exact-time
-- ties), and past fraud labels are never used.
--
-- card_id here is the reconstructed account id (card1 + addr1 + start day).
-- card1 alone is shared by many accounts; its velocity is included as a
-- separate signal (bursts on a card range, e.g. card testing).

WITH base AS (
    SELECT *, ln(1 + amt) AS log_amt FROM {source}
),
windows AS (
    SELECT
        *,
        count(*)          OVER w_1h  AS n_txn_1h,
        count(*)          OVER w_24h AS n_txn_24h,
        count(*)          OVER w_7d  AS n_txn_7d,
        count(*)          OVER w_all AS n_prior_txn,
        coalesce(sum(amt) OVER w_24h, 0) AS amt_sum_24h,
        avg(amt)          OVER w_30d AS acct_avg_amt_30d,
        stddev_samp(amt)  OVER w_30d AS acct_std_amt_30d,
        date_diff('second', max(ts) OVER w_all, ts) / 3600.0 AS hours_since_prev,
        count(*) OVER (PARTITION BY card_id, product ORDER BY ts
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_prior_same_product,
        count(*) OVER (PARTITION BY card_id, email_domain ORDER BY ts
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_prior_same_email,
        count(*) OVER (PARTITION BY card_id, device_info ORDER BY ts
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_prior_same_device,
        count(*) OVER (PARTITION BY card1 ORDER BY ts
                       RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS card1_txn_1h,
        count(*) OVER (PARTITION BY card1 ORDER BY ts
                       RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS card1_txn_30d
    FROM base
    WINDOW
        w_1h  AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 1 HOUR   PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_24h AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_7d  AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 7 DAYS   PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_30d AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 30 DAYS  PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_all AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)
)
SELECT
    * EXCLUDE (acct_std_amt_30d, n_prior_same_product, n_prior_same_email, n_prior_same_device),
    amt / nullif(acct_avg_amt_30d, 0)                             AS amt_vs_30d_avg,
    (amt - acct_avg_amt_30d) / nullif(acct_std_amt_30d, 0)         AS amt_zscore_30d,
    CASE WHEN n_prior_txn = 0 THEN NULL
         WHEN n_prior_same_product = 0 THEN 1 ELSE 0 END           AS new_product_for_acct,
    CASE WHEN n_prior_txn = 0 OR email_domain IS NULL THEN NULL
         WHEN n_prior_same_email = 0 THEN 1 ELSE 0 END             AS new_email_for_acct,
    CASE WHEN n_prior_txn = 0 OR device_info IS NULL THEN NULL
         WHEN n_prior_same_device = 0 THEN 1 ELSE 0 END            AS new_device_for_acct
FROM windows
ORDER BY ts, txn_id
