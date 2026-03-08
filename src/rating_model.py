from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, RANDOM_STATE

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGB = False


NUMERIC_FEATURES = [
    "price_usd",
    "log_price",
    "review_count",
    "review_volume",
    "negative_rate",
    "positive_rate",
    "popularity_score",
    "brand_total_reviews",
    "category_avg_rating",
    "avg_review_length",
]

CATEGORICAL_FEATURES = [
    "brand_name",
    "primary_category",
    "secondary_category",
    "tertiary_category",
]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERIC_FEATURES,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ])

    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_rating_model(features: pd.DataFrame) -> tuple[Pipeline, dict]:
    df = features.dropna(subset=["rating"]).copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "model_type": "XGBRegressor" if HAS_XGB else "RandomForestRegressor",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rating_model.joblib")

    return model, metrics


def score_products(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df["predicted_rating"] = model.predict(X)
    df["rating_gap"] = df["rating"] - df["predicted_rating"]
    return df