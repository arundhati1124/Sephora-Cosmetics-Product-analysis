from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = ROOT / "reports" / "product_intel.csv"

st.set_page_config(page_title="Sephora Intelligence", layout="wide")
st.title("Sephora Cosmetics Intelligence Platform")
st.caption("Mid-level ML project for product health, sentiment, complaints, and rating intelligence.")

if not REPORT_FILE.exists():
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

df = pd.read_csv(REPORT_FILE)

with st.sidebar:
    st.header("Filters")

    brand_options = sorted(df["brand_name"].dropna().unique().tolist())
    category_options = sorted(df["primary_category"].dropna().unique().tolist())
    risk_options = sorted(df["risk_flag"].dropna().unique().tolist())

    selected_brands = st.multiselect("Brand", brand_options)
    selected_categories = st.multiselect("Category", category_options)
    selected_risks = st.multiselect("Risk Flag", risk_options, default=risk_options)

filtered = df.copy()

if selected_brands:
    filtered = filtered[filtered["brand_name"].isin(selected_brands)]

if selected_categories:
    filtered = filtered[filtered["primary_category"].isin(selected_categories)]

if selected_risks:
    filtered = filtered[filtered["risk_flag"].isin(selected_risks)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products", len(filtered))
c2.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")
c3.metric("Avg Negative Rate", f"{filtered['negative_rate'].mean():.2%}")
c4.metric("Avg Quality Score", f"{filtered['quality_score'].mean():.2f}")

st.subheader("Top Products by Quality Score")
st.dataframe(
    filtered.sort_values("quality_score", ascending=False)[[
        "product_name",
        "brand_name",
        "primary_category",
        "price_usd",
        "rating",
        "negative_rate",
        "rating_gap",
        "top_complaint_topic",
        "quality_score",
        "risk_flag",
    ]].head(20),
    use_container_width=True,
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Distribution")
    fig = px.histogram(filtered, x="risk_flag")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Rating Gap vs Negative Rate")
    fig = px.scatter(
        filtered,
        x="rating_gap",
        y="negative_rate",
        color="risk_flag",
        size="review_count",
        hover_name="product_name",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Complaint Topics")
topic_df = filtered["top_complaint_topic"].value_counts().reset_index()
topic_df.columns = ["topic", "count"]
fig = px.bar(topic_df.head(10), x="topic", y="count")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Full Product Intelligence Table")
st.dataframe(filtered, use_container_width=True)