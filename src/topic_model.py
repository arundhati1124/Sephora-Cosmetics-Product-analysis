from __future__ import annotations

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import MAX_TOPIC_SAMPLES, N_TOPICS, RANDOM_STATE


def train_topic_model(reviews: pd.DataFrame):
    neg_reviews = reviews.loc[reviews["negative"] == 1, "review_text"].dropna()

    if neg_reviews.empty:
        raise ValueError("No negative reviews available for topic modeling.")

    if len(neg_reviews) > MAX_TOPIC_SAMPLES:
        neg_reviews = neg_reviews.sample(MAX_TOPIC_SAMPLES, random_state=RANDOM_STATE)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        min_df=5,
    )
    X = vectorizer.fit_transform(neg_reviews)

    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_STATE)
    nmf.fit(X)

    vocab = vectorizer.get_feature_names_out()
    topic_names = {}

    for i, weights in enumerate(nmf.components_):
        top_words = [vocab[j] for j in weights.argsort()[-4:][::-1]]
        topic_names[i] = ", ".join(top_words)

    return vectorizer, nmf, topic_names


def assign_topics(reviews: pd.DataFrame, vectorizer, nmf, topic_names: dict) -> pd.DataFrame:
    df = reviews.copy()

    mask = df["negative"] == 1
    X = vectorizer.transform(df.loc[mask, "review_text"])
    scores = nmf.transform(X)
    best_topics = scores.argmax(axis=1)

    df.loc[mask, "complaint_topic_id"] = best_topics
    df.loc[mask, "complaint_topic_name"] = [topic_names[int(t)] for t in best_topics]
    df["complaint_topic_name"] = df["complaint_topic_name"].fillna("non_negative_review")

    return df


def top_topic_per_product(reviews_with_topics: pd.DataFrame) -> pd.DataFrame:
    neg = reviews_with_topics[reviews_with_topics["negative"] == 1].copy()

    if neg.empty:
        return pd.DataFrame(columns=["product_id", "top_complaint_topic"])

    result = (
        neg.groupby(["product_id", "complaint_topic_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["product_id", "count"], ascending=[True, False])
        .drop_duplicates(subset="product_id")
        .rename(columns={"complaint_topic_name": "top_complaint_topic"})
    )

    return result[["product_id", "top_complaint_topic"]]