from __future__ import annotations

import json

from .config import REPORTS_DIR
from .data_prep import prepare_data
from .rating_model import score_products, train_rating_model
from .scoring import build_product_intelligence
from .sentiment_model import score_reviews, train_sentiment_model
from .topic_model import assign_topics, top_topic_per_product, train_topic_model


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    products, reviews, features = prepare_data()

    sentiment_model, sentiment_metrics = train_sentiment_model(reviews)
    reviews_scored = score_reviews(sentiment_model, reviews)

    neg_rate_model = (
        reviews_scored.groupby("product_id")["negative_probability"]
        .mean()
        .reset_index(name="negative_rate_model")
    )

    features = features.merge(neg_rate_model, on="product_id", how="left")
    features["negative_rate"] = features["negative_rate_model"].fillna(features["negative_rate"])
    features["positive_rate"] = 1 - features["negative_rate"]
    features = features.drop(columns=["negative_rate_model"])

    rating_model, rating_metrics = train_rating_model(features)
    products_scored = score_products(rating_model, features)

    vectorizer, nmf, topic_names = train_topic_model(reviews_scored)
    reviews_topics = assign_topics(reviews_scored, vectorizer, nmf, topic_names)
    top_topics = top_topic_per_product(reviews_topics)

    product_intel = build_product_intelligence(products_scored, top_topics)

    product_intel.to_csv(REPORTS_DIR / "product_intel.csv", index=False)
    reviews_scored.to_parquet(REPORTS_DIR / "scored_reviews.parquet", index=False)

    metrics = {
        "sentiment": sentiment_metrics,
        "rating": rating_metrics,
        "topic_names": topic_names,
        "n_products": int(len(products)),
        "n_reviews": int(len(reviews)),
    }

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()