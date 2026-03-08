from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    spread = s.max() - s.min()
    if spread == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / spread


def build_product_intelligence(products_scored: pd.DataFrame, top_topics: pd.DataFrame) -> pd.DataFrame:
    df = products_scored.merge(top_topics, on="product_id", how="left")
    df["top_complaint_topic"] = df["top_complaint_topic"].fillna("no_major_negative_topic")

    rating_norm = minmax(df["rating"])
    review_norm = minmax(np.log1p(df["review_count"]))
    pos_norm = minmax(df["positive_rate"])
    neg_norm = minmax(df["negative_rate"])

    df["quality_score"] = (
        0.4 * rating_norm
        + 0.3 * review_norm
        + 0.2 * pos_norm
        - 0.1 * neg_norm
    )

    df["risk_flag"] = np.select(
        [
            (df["negative_rate"] >= 0.25) & (df["rating_gap"] <= -0.25),
            (df["negative_rate"] >= 0.15) | (df["rating_gap"] <= -0.15),
        ],
        ["High", "Medium"],
        default="Low",
    )

    return df.sort_values("quality_score", ascending=False)