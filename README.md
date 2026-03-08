# Sephora Cosmetics Intelligence Platform (Mid-Level Version)

A portfolio-ready machine learning project that turns Sephora product and review data into product health insights.

This version is intentionally scoped as a mid-level project: strong business framing, solid modeling, explainable outputs, and a usable dashboard — without unnecessary production complexity.

## What this project solves

A beauty e-commerce team wants to:
- spot products with rising negative feedback,
- identify underperforming or overperforming products,
- understand the main complaint themes in reviews,
- rank products using a simple business-friendly health score.

## Project scope

This project includes four core parts:
1. Sentiment classification for negative vs non-negative reviews.
2. Rating prediction for expected product ratings.
3. Complaint topic discovery from negative reviews.
4. Product intelligence scoring to rank products and flag risks.

## Repository structure

```text
sephora_midlevel/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_prep.py
│   ├── sentiment_model.py
│   ├── rating_model.py
│   ├── topic_model.py
│   ├── scoring.py
│   └── pipeline.py
├── requirements.txt
└── README.md
