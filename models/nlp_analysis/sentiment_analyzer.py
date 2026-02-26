"""
Sentiment analysis pipeline for pre-match news articles.

Processes all collected news articles and produces per-match NLP features
saved to data/processed/nlp_features.csv.

Model used: distilbert-base-uncased-finetuned-sst-2-english
Output scores are mapped to [-1, +1] (negative → positive).

Usage:
    python models/nlp_analysis/sentiment_analyzer.py
    python models/nlp_analysis/sentiment_analyzer.py --model cardiffnlp/twitter-roberta-base-sentiment
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline

from text_preprocessor import clean_text, chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEWS_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "news_articles.json"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = PROCESSED_DIR / "nlp_features.csv"

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = 0 if torch.cuda.is_available() else -1


def load_sentiment_pipeline(model_name: str):
    log.info("Loading sentiment model: %s (device=%s)", model_name, "cuda" if DEVICE == 0 else "cpu")
    return hf_pipeline("sentiment-analysis", model=model_name, device=DEVICE, truncation=True, max_length=512)


def score_to_float(result: dict, model_name: str) -> float:
    """Normalize pipeline output to a [-1, +1] float."""
    label = result["label"].upper()
    score = result["score"]  # confidence

    if "ROBERTA" in model_name.upper():
        # cardiffnlp model: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
        mapping = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
        polarity = mapping.get(label, 0.0)
    else:
        # SST-2 model: NEGATIVE / POSITIVE
        polarity = 1.0 if "POSITIVE" in label else -1.0

    return polarity * score


def analyze_article(text: str, pipe, model_name: str) -> float:
    """Return mean sentiment score across text chunks."""
    chunks = chunk_text(clean_text(text), max_chars=400)
    scores = []
    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        try:
            result = pipe(chunk)[0]
            scores.append(score_to_float(result, model_name))
        except Exception as exc:
            log.debug("Chunk analysis failed: %s", exc)
    return float(np.mean(scores)) if scores else 0.0


def extract_injury_score(articles: list[dict], team_name: str) -> float:
    """Count articles mentioning injury-related keywords near a team name."""
    injury_keywords = {"injury", "injured", "doubt", "fitness", "sidelined", "out", "miss", "absence", "ruled out"}
    count = 0
    for art in articles:
        text = (art.get("article_content") or art.get("text", "")).lower()
        team_lower = team_name.lower()
        if team_lower in text and any(kw in text for kw in injury_keywords):
            count += 1
    return min(count / max(len(articles), 1), 1.0)


def extract_pressure_keywords(articles: list[dict], team_name: str) -> int:
    """Count articles mentioning pressure-related keywords for a team."""
    pressure_keywords = {"must-win", "crisis", "pressure", "sacking", "relegation", "desperate"}
    count = 0
    for art in articles:
        text = (art.get("article_content") or art.get("text", "")).lower()
        team_lower = team_name.lower()
        if team_lower in text and any(kw in text for kw in pressure_keywords):
            count += 1
    return count


def extract_morale_keywords(articles: list[dict], team_name: str) -> int:
    """Count articles mentioning positive morale keywords for a team."""
    morale_keywords = {"confident", "momentum", "motivated", "form", "winning streak", "boost"}
    count = 0
    for art in articles:
        text = (art.get("article_content") or art.get("text", "")).lower()
        team_lower = team_name.lower()
        if team_lower in text and any(kw in text for kw in morale_keywords):
            count += 1
    return count


def process_match(
    match_id: int,
    home_team: str,
    away_team: str,
    articles: list[dict],
    pipe,
    model_name: str,
) -> dict:
    """Compute all NLP features for one match."""
    home_sentiments = []
    away_sentiments = []

    for art in articles:
        text = art.get("article_content") or art.get("text", "")
        if not text or len(text) < 50:
            continue

        score = analyze_article(text, pipe, model_name)

        # Determine which team the article focuses on
        text_lower = text.lower()
        home_mentions = text_lower.count(home_team.lower())
        away_mentions = text_lower.count(away_team.lower())

        if home_mentions > away_mentions:
            home_sentiments.append(score)
        elif away_mentions > home_mentions:
            away_sentiments.append(score)
        else:
            # Equal mentions — attribute to both
            home_sentiments.append(score)
            away_sentiments.append(score)

    mean_home = float(np.mean(home_sentiments)) if home_sentiments else 0.0
    mean_away = float(np.mean(away_sentiments)) if away_sentiments else 0.0

    return {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "article_count": len(articles),
        "sentiment_mean_home": round(mean_home, 4),
        "sentiment_mean_away": round(mean_away, 4),
        "sentiment_std_home": round(float(np.std(home_sentiments)) if len(home_sentiments) > 1 else 0.0, 4),
        "sentiment_std_away": round(float(np.std(away_sentiments)) if len(away_sentiments) > 1 else 0.0, 4),
        "sentiment_gap": round(mean_home - mean_away, 4),
        "confidence_score_home": round(0.5 + (mean_home * 0.5), 4),
        "confidence_score_away": round(0.5 + (mean_away * 0.5), 4),
        "injury_concern_score_home": round(extract_injury_score(articles, home_team), 4),
        "injury_concern_score_away": round(extract_injury_score(articles, away_team), 4),
        "hype_level": round(min(len(articles) / 10.0, 1.0), 4),
        "pressure_keywords_home": extract_pressure_keywords(articles, home_team),
        "pressure_keywords_away": extract_pressure_keywords(articles, away_team),
        "morale_keywords_home": extract_morale_keywords(articles, home_team),
        "morale_keywords_away": extract_morale_keywords(articles, away_team),
    }


def main():
    parser = argparse.ArgumentParser(description="Run NLP sentiment pipeline on match articles.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    args = parser.parse_args()

    pipe = load_sentiment_pipeline(args.model)

    if not NEWS_FILE.exists():
        log.error("News articles file not found at %s. Run collect_news.py first.", NEWS_FILE)
        return

    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        all_articles = json.load(f)

    if not all_articles:
        log.error("No articles found in %s", NEWS_FILE)
        return

    log.info("Loaded %d articles from %s", len(all_articles), NEWS_FILE)

    # Group articles by match_id
    match_groups: dict[int, dict] = {}
    for art in all_articles:
        mid = art.get("match_id")
        if mid is None:
            continue
        if mid not in match_groups:
            match_groups[mid] = {
                "home_team": art.get("home_team", ""),
                "away_team": art.get("away_team", ""),
                "articles": [],
            }
        match_groups[mid]["articles"].append(art)

    log.info("Processing %d matches...", len(match_groups))

    rows = []
    for idx, (match_id, data) in enumerate(match_groups.items()):
        log.info(
            "Match %d/%d: %s vs %s (%d articles)",
            idx + 1, len(match_groups),
            data["home_team"], data["away_team"],
            len(data["articles"]),
        )
        row = process_match(match_id, data["home_team"], data["away_team"], data["articles"], pipe, args.model)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_FILE, index=False)
        log.info("NLP features saved to %s (%d matches)", OUT_FILE, len(df))
    else:
        log.warning("No data processed.")


if __name__ == "__main__":
    main()
