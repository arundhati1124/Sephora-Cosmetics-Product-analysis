from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, RANDOM_STATE


def train_sentiment_model(reviews: pd.DataFrame) -> tuple[Pipeline, dict]:
    df = reviews.dropna(subset=["review_text", "negative"]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"],
        df["negative"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["negative"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=5)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "f1_negative": float(f1_score(y_test, preds)),
        "precision_negative": float(precision_score(y_test, preds)),
        "recall_negative": float(recall_score(y_test, preds)),
        "classification_report": classification_report(y_test, preds),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "sentiment_model.joblib")

    return model, metrics


def score_reviews(model: Pipeline, reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()
    df["negative_probability"] = model.predict_proba(df["review_text"])[:, 1]
    df["pred_negative"] = (df["negative_probability"] >= 0.5).astype(int)
    return df