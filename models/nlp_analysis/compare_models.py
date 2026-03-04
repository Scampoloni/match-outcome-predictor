"""
Compare two sentiment analysis models (DistilBERT vs RoBERTa) on match articles.

Runs both models on the same article set and produces comparison metrics.
This satisfies the NLP block requirement for comparing at least one NLP approach.

Usage:
    python models/nlp_analysis/compare_models.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

from text_preprocessor import clean_text, chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEWS_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "news_articles.json"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}


def score_to_float(result: dict, model_key: str) -> float:
    """Normalize pipeline output to [-1, +1]."""
    label = result["label"].upper()
    score = result["score"]

    if model_key == "roberta":
        mapping = {"NEGATIVE": -1.0, "NEUTRAL": 0.0, "POSITIVE": 1.0}
        polarity = mapping.get(label, 0.0)
    else:
        polarity = 1.0 if "POSITIVE" in label else -1.0
    return polarity * score


def analyze_text(text: str, pipe, model_key: str) -> float:
    """Sentiment score for a piece of text (chunked for transformers)."""
    chunks = chunk_text(clean_text(text), max_chars=400)
    scores = []
    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        try:
            result = pipe(chunk)[0]
            scores.append(score_to_float(result, model_key))
        except Exception as exc:
            log.debug("Chunk error: %s", exc)
    return float(np.mean(scores)) if scores else 0.0


def main():
    if not NEWS_FILE.exists():
        log.error("No news articles found. Run collect_news.py first.")
        return

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    if not all_articles:
        log.error("Empty articles file.")
        return

    # Sample articles for comparison (use up to 200 for speed)
    sample_size = min(200, len(all_articles))
    sampled = all_articles[:sample_size]
    log.info("Comparing models on %d articles (total available: %d)", sample_size, len(all_articles))

    results = {key: [] for key in MODELS}

    for model_key, model_name in MODELS.items():
        log.info("Loading model: %s (%s)", model_key, model_name)
        try:
            pipe = hf_pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)
        except Exception as exc:
            log.error("Failed to load %s: %s", model_name, exc)
            continue

        for idx, art in enumerate(sampled):
            text = art.get("article_content") or art.get("text", "")
            if not text or len(text) < 50:
                results[model_key].append(0.0)
                continue

            score = analyze_text(text, pipe, model_key)
            results[model_key].append(score)

            if (idx + 1) % 50 == 0:
                log.info("  %s: processed %d/%d articles", model_key, idx + 1, sample_size)

    # Build comparison DataFrame
    comparison_df = pd.DataFrame({
        "article_index": range(sample_size),
        "title": [a.get("article_title", "") for a in sampled],
        "home_team": [a.get("home_team", "") for a in sampled],
        "away_team": [a.get("away_team", "") for a in sampled],
    })

    for model_key in MODELS:
        if results[model_key]:
            comparison_df[f"sentiment_{model_key}"] = results[model_key]

    # Save comparison
    out_path = PROCESSED_DIR / "nlp_model_comparison.csv"
    comparison_df.to_csv(out_path, index=False)

    # Print summary statistics
    log.info("=" * 60)
    log.info("NLP MODEL COMPARISON SUMMARY")
    log.info("=" * 60)

    for model_key in MODELS:
        col = f"sentiment_{model_key}"
        if col in comparison_df.columns:
            scores = comparison_df[col]
            log.info(
                "%s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                model_key, scores.mean(), scores.std(), scores.min(), scores.max()
            )
            pos_pct = (scores > 0.1).mean() * 100
            neg_pct = (scores < -0.1).mean() * 100
            neu_pct = 100 - pos_pct - neg_pct
            log.info(
                "  Distribution: %.1f%% positive, %.1f%% neutral, %.1f%% negative",
                pos_pct, neu_pct, neg_pct
            )

    # Correlation between models
    if "sentiment_distilbert" in comparison_df.columns and "sentiment_roberta" in comparison_df.columns:
        corr = comparison_df["sentiment_distilbert"].corr(comparison_df["sentiment_roberta"])
        log.info("Correlation between DistilBERT and RoBERTa: %.4f", corr)

        # Agreement rate (same polarity)
        same_sign = (
            (comparison_df["sentiment_distilbert"] > 0) == (comparison_df["sentiment_roberta"] > 0)
        ).mean()
        log.info("Polarity agreement rate: %.1f%%", same_sign * 100)

    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
