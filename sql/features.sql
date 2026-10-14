-- Point-in-time feature table: one row per transaction.
--
-- Every behavioural feature is computed from the SAME card's EARLIER
-- transactions only. `EXCLUDE GROUP` drops the current row and any row with
-- the exact same timestamp, so nothing at or after the moment of scoring can
-- leak in. Past fraud labels are never used either: in real life chargebacks
-- arrive weeks later, so they would not be known at scoring time.
--
-- {source} is replaced by the input table or parquet path at run time.

WITH base AS (
    SELECT
        txn_id, card_id, ts, amt, category, merchant, is_fraud,
        ln(1 + amt)                                       AS log_amt,
        hour(ts)                                          AS hour,
        dayofweek(ts)                                     AS day_of_week,
        CASE WHEN hour(ts) >= 22 OR hour(ts) < 4 THEN 1 ELSE 0 END AS is_night,
        date_diff('day', dob, CAST(ts AS DATE)) / 365.25  AS age,
        CASE WHEN gender = 'F' THEN 1 ELSE 0 END          AS is_female,
        ln(1 + city_pop)                                  AS log_city_pop,
        -- great-circle distance from cardholder's home to the merchant (km)
        2 * 6371 * asin(sqrt(
            pow(sin(radians(merch_lat - lat) / 2), 2) +
            cos(radians(lat)) * cos(radians(merch_lat)) * pow(sin(radians(merch_long - long) / 2), 2)
        ))                                                AS distance_km
    FROM {source}
),

windows AS (
    SELECT
        *,
        -- Velocity: how busy has this card been recently?
        count(*)                 OVER w_1h   AS n_txn_1h,
        count(*)                 OVER w_24h  AS n_txn_24h,
        count(*)                 OVER w_7d   AS n_txn_7d,
        coalesce(sum(amt)        OVER w_24h, 0) AS amt_sum_24h,
        coalesce(sum(is_night)   OVER w_24h, 0) AS n_night_24h,
        -- Card's normal behaviour
        avg(amt)                 OVER w_30d  AS card_avg_amt_30d,
        avg(amt)                 OVER w_90d  AS card_avg_amt_90d,
        stddev_samp(amt)         OVER w_90d  AS card_std_amt_90d,
        count(*)                 OVER w_90d  AS n_txn_90d,
        sum(is_night)            OVER w_90d  AS n_night_90d,
        -- Time since the card's previous transaction, and card age
        date_diff('second', max(ts) OVER w_all, ts)        AS secs_since_prev,
        date_diff('second', min(ts) OVER w_all, ts) / 86400.0 AS card_tenure_days,
        -- Novelty: has this card used this category / merchant before?
        count(*) OVER (PARTITION BY card_id, category ORDER BY ts
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_prior_same_category,
        count(*) OVER (PARTITION BY card_id, category ORDER BY ts
                       RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_same_category_90d,
        count(*) OVER (PARTITION BY card_id, merchant ORDER BY ts
                       RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS n_prior_same_merchant
    FROM base
    WINDOW
        w_1h  AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 1 HOUR  PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_24h AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_7d  AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 7 DAYS  PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_30d AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_90d AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        w_all AS (PARTITION BY card_id ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)
)

SELECT
    txn_id, card_id, ts, amt, category, merchant, is_fraud,
    -- transaction-only ("static") features
    log_amt, hour, day_of_week, is_night, age, is_female, log_city_pop, distance_km,
    -- behavioural features
    n_txn_1h, n_txn_24h, n_txn_7d, amt_sum_24h, n_night_24h,
    card_avg_amt_30d,
    amt / nullif(card_avg_amt_30d, 0)                                  AS amt_vs_30d_avg,
    (amt - card_avg_amt_90d) / nullif(card_std_amt_90d, 0)              AS amt_zscore_90d,
    n_night_90d * 1.0 / nullif(n_txn_90d, 0)                           AS night_share_90d,
    n_same_category_90d * 1.0 / nullif(n_txn_90d, 0)                   AS category_share_90d,
    CASE WHEN n_prior_same_category = 0 THEN 1 ELSE 0 END              AS first_time_category,
    CASE WHEN n_prior_same_merchant = 0 THEN 1 ELSE 0 END              AS first_time_merchant,
    secs_since_prev / 3600.0                                           AS hours_since_prev,
    card_tenure_days
FROM windows
ORDER BY ts, txn_id
