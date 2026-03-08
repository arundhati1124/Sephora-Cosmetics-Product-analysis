from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, PRODUCT_FILE, REVIEW_FILES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRODUCT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PRODUCT_FILE}")

    missing_review_files = [f for f in REVIEW_FILES if not f.exists()]
    if missing_review_files:
        raise FileNotFoundError(f"Missing review files: {missing_review_files}")

    products = pd.read_csv(PRODUCT_FILE)
    reviews = pd.concat([pd.read_csv(f) for f in REVIEW_FILES], ignore_index=True)

    return products, reviews


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "product_id",
        "product_name",
        "brand_name",
        "loves_count",
        "rating",
        "reviews",
        "price_usd",
        "primary_category",
        "secondary_category",
        "tertiary_category",
    ]

    missing = [c for c in required_cols if c not in products.columns]
    if missing:
        raise ValueError(f"Missing columns in product_info.csv: {missing}")

    df = products[required_cols].copy()
    df = df.rename(columns={"reviews": "review_count"})
    df = df.drop_duplicates(subset="product_id")

    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df["loves_count"] = pd.to_numeric(df["loves_count"], errors="coerce")

    df = df[(df["price_usd"] >= 1) & (df["price_usd"] <= 500)]

    df["brand_name"] = df["brand_name"].fillna("Unknown")
    df["primary_category"] = df["primary_category"].fillna("Unknown")
    df["secondary_category"] = df["secondary_category"].fillna("Unknown")
    df["tertiary_category"] = df["tertiary_category"].fillna("Unknown")
    df["review_count"] = df["review_count"].fillna(0)
    df["loves_count"] = df["loves_count"].fillna(0)
    df["rating"] = df["rating"].fillna(df["rating"].median())

    return df


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "product_id",
        "rating",
        "is_recommended",
        "submission_time",
        "review_text",
        "review_title",
        "product_name",
        "brand_name",
        "price_usd",
    ]

    missing = [c for c in required_cols if c not in reviews.columns]
    if missing:
        raise ValueError(f"Missing columns in review files: {missing}")

    df = reviews[required_cols].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["is_recommended"] = pd.to_numeric(df["is_recommended"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")

    df = df.dropna(subset=["product_id", "rating", "review_text"])

    df["review_title"] = df["review_title"].fillna("")
    df["review_text"] = df["review_text"].map(clean_text)

    df = df[df["review_text"].str.len() > 10]
    df = df.drop_duplicates(subset=["product_id", "review_text"])

    df["negative"] = ((df["rating"] < 3) | (df["is_recommended"].fillna(1) == 0)).astype(int)
    df["review_length"] = df["review_text"].str.split().str.len()

    return df


def build_product_features(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    review_agg = (
        reviews.groupby("product_id")
        .agg(
            review_volume=("review_text", "count"),
            avg_review_rating=("rating", "mean"),
            negative_rate=("negative", "mean"),
            avg_review_length=("review_length", "mean"),
        )
        .reset_index()
    )

    review_agg["positive_rate"] = 1 - review_agg["negative_rate"]

    feat = products.merge(review_agg, on="product_id", how="left")

    feat["review_volume"] = feat["review_volume"].fillna(0)
    feat["avg_review_rating"] = feat["avg_review_rating"].fillna(feat["rating"])
    feat["negative_rate"] = feat["negative_rate"].fillna(0)
    feat["positive_rate"] = feat["positive_rate"].fillna(1)
    feat["avg_review_length"] = feat["avg_review_length"].fillna(0)

    feat["log_price"] = np.log1p(feat["price_usd"])
    feat["popularity_score"] = feat["loves_count"] / (feat["review_count"] + 1)
    feat["brand_total_reviews"] = feat.groupby("brand_name")["review_volume"].transform("sum")
    feat["category_avg_rating"] = feat.groupby("primary_category")["rating"].transform("mean")

    return feat


def save_processed(products: pd.DataFrame, reviews: pd.DataFrame, features: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    products.to_csv(DATA_PROCESSED / "clean_products.csv", index=False)
    reviews.to_parquet(DATA_PROCESSED / "clean_reviews.parquet", index=False)
    features.to_parquet(DATA_PROCESSED / "product_features.parquet", index=False)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_products, raw_reviews = load_data()
    products = clean_products(raw_products)
    reviews = clean_reviews(raw_reviews)
    features = build_product_features(products, reviews)
    save_processed(products, reviews, features)
    return products, reviews, features