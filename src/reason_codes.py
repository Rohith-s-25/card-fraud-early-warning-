"""Turn SHAP values into plain-English reasons an analyst can act on."""
import numpy as np
import pandas as pd


def _fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def describe(feature: str, row: pd.Series) -> str:
    """One readable sentence for why `feature` pushed this transaction towards fraud."""
    v = row.get(feature)
    amt, cat = row["amt"], row["category"]
    if feature.startswith("cat_"):
        return f"Category '{cat}' carries higher fraud risk"
    templates = {
        "amt_vs_30d_avg": lambda: f"Amount ${amt:,.0f} is {v:.1f}x this card's 30-day average",
        "amt_zscore_90d": lambda: f"Amount is {v:.1f} std devs away from this card's 90-day norm",
        "log_amt": lambda: f"Amount ${amt:,.0f} is in a high-risk range",
        "category_share_90d": lambda: (f"No '{cat}' purchases on this card in the last 90 days" if v == 0
                                       else f"Only {v:.0%} of this card's last-90-day purchases were in '{cat}'"),
        "first_time_category": lambda: f"First ever purchase in '{cat}' on this card",
        "first_time_merchant": lambda: "First purchase at this merchant",
        "n_txn_1h": lambda: ("No other transactions on this card in the past hour" if v == 0
                             else f"{v:.0f} other transaction{'s' if v != 1 else ''} on this card in the past hour"),
        "n_txn_24h": lambda: ("No other transactions on this card in the past 24h" if v == 0
                              else f"{v:.0f} transaction{'s' if v != 1 else ''} on this card in the past 24h"),
        "n_txn_7d": lambda: f"{v:.0f} transactions on this card in the past 7 days",
        "amt_sum_24h": lambda: f"${v:,.0f} already spent on this card in the past 24h",
        "hours_since_prev": lambda: f"{_fmt_hours(v)} since this card's previous transaction",
        "card_avg_amt_30d": lambda: f"Card usually spends little (30-day average ${v:,.0f})",
        "day_of_week": lambda: "Day-of-week pattern",
        "hour": lambda: f"Made at {int(v):02d}:00",
        "is_night": lambda: "Made late at night (22:00-04:00)",
        "night_share_90d": lambda: f"Card is rarely used at night ({v:.0%} of its last 90 days)",
        "n_night_24h": lambda: f"{v:.0f} late-night transaction{'s' if v != 1 else ''} on this card in the past 24h",
        "distance_km": lambda: f"Merchant is {v:.0f} km from cardholder's home",
    }
    try:
        return templates[feature]() if feature in templates else f"{feature} = {v}"
    except (TypeError, ValueError):
        return feature


def top_reasons(shap_row: np.ndarray, features: list, row: pd.Series, k: int = 3) -> list:
    """Top-k features pushing towards fraud (positive SHAP), as sentences."""
    order = np.argsort(-shap_row)
    out = []
    for i in order:
        if shap_row[i] <= 0 or len(out) == k:
            break
        out.append(describe(features[i], row))
    return out
