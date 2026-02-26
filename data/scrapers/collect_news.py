"""
Collect pre-match news articles for sentiment analysis.

Fetches news articles from NewsAPI (and optional web scraping) for matches
in the dataset. Articles are stored as JSON for NLP processing.

Usage:
    python data/scrapers/collect_news.py
    python data/scrapers/collect_news.py --matches data/raw/matches_raw.csv --limit 200
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    except ImportError:
        pass

NEWS_API_URL = "https://newsapi.org/v2/everything"

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
OUT_FILE = RAW_DIR / "news_articles.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def get_articles_newsapi(
    home_team: str, away_team: str, match_date: str, days_before: int = 7
) -> List[Dict]:
    """
    Fetch news articles about a match from NewsAPI.

    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date (ISO format)
        days_before: How many days before the match to search

    Returns:
        List of article dictionaries
    """
    if not NEWS_API_KEY:
        return []

    match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00").split("T")[0])
    from_date = match_dt - timedelta(days=days_before)
    to_date = match_dt - timedelta(days=1)

    # Simplify team names for search (remove FC, etc.)
    home_simple = home_team.replace(" FC", "").replace("FC ", "")
    away_simple = away_team.replace(" FC", "").replace("FC ", "")

    query = f'"{home_simple}" AND "{away_simple}"'

    params = {
        "q": query,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=15)
        response.raise_for_status()

        articles = response.json().get("articles", [])
        logger.info(
            "  NewsAPI: %d articles for %s vs %s", len(articles), home_simple, away_simple
        )

        time.sleep(1)
        return articles

    except requests.exceptions.RequestException as e:
        logger.warning("NewsAPI error for %s vs %s: %s", home_team, away_team, e)
        return []


def scrape_bbc_preview(home_team: str, away_team: str) -> List[Dict]:
    """
    Scrape match preview from BBC Sport (fallback if NewsAPI unavailable).

    Args:
        home_team: Home team name
        away_team: Away team name

    Returns:
        List of article dictionaries
    """
    query = f"{home_team} {away_team}".replace(" ", "+")
    url = f"https://www.bbc.co.uk/search?q={query}&filter=sport"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        articles = []
        links = soup.select("a[href*='/sport/football/']")[:3]

        for link in links:
            href = link.get("href", "")
            if not href.startswith("http"):
                href = "https://www.bbc.co.uk" + href

            try:
                art_resp = requests.get(href, headers=HEADERS, timeout=10)
                art_soup = BeautifulSoup(art_resp.text, "lxml")

                # Extract article text
                container = art_soup.select_one("article") or art_soup.select_one('[role="main"]')
                if container:
                    text = " ".join(p.get_text(strip=True) for p in container.find_all("p"))
                    if len(text) > 100:
                        articles.append(
                            {
                                "title": art_soup.title.get_text(strip=True) if art_soup.title else "",
                                "content": text,
                                "url": href,
                                "source": {"name": "BBC Sport"},
                                "publishedAt": datetime.utcnow().isoformat(),
                            }
                        )
                time.sleep(2)
            except Exception:
                continue

        return articles

    except requests.exceptions.RequestException as e:
        logger.warning("BBC scraping failed: %s", e)
        return []


def main():
    """Main news collection pipeline."""
    parser = argparse.ArgumentParser(description="Collect pre-match news articles.")
    parser.add_argument(
        "--matches", type=Path, default=RAW_DIR / "matches_raw.csv",
        help="Path to matches CSV"
    )
    parser.add_argument("--limit", type=int, default=100, help="Max matches to process")
    parser.add_argument("--days-before", type=int, default=7, help="Days before match to search")
    args = parser.parse_args()

    if not args.matches.exists():
        logger.error("Matches file not found: %s. Run collect_matches.py first.", args.matches)
        return

    matches_df = pd.read_csv(args.matches)
    logger.info("Loaded %d matches from %s", len(matches_df), args.matches)

    # Sample if too many
    if len(matches_df) > args.limit:
        matches_df = matches_df.sample(n=args.limit, random_state=42)
        logger.info("Sampled %d matches for article collection", args.limit)

    all_articles = []

    for idx, match in matches_df.iterrows():
        logger.info(
            "Processing %d/%d: %s vs %s (%s)",
            idx + 1, len(matches_df),
            match["home_team"], match["away_team"],
            match.get("date", "unknown date"),
        )

        # Try NewsAPI first
        articles = get_articles_newsapi(
            match["home_team"], match["away_team"],
            str(match["date"]), days_before=args.days_before,
        )

        # Fallback to BBC scraping
        if not articles and not NEWS_API_KEY:
            articles = scrape_bbc_preview(match["home_team"], match["away_team"])

        for article in articles:
            all_articles.append(
                {
                    "match_id": match.get("match_id"),
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "match_date": str(match["date"]),
                    "article_title": article.get("title"),
                    "article_content": article.get("content") or article.get("description", ""),
                    "article_url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name", "unknown"),
                }
            )

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info("NEWS COLLECTION SUMMARY")
    logger.info("=" * 50)
    logger.info("Total articles: %d", len(all_articles))
    logger.info("Matches covered: %d", len(set(a["match_id"] for a in all_articles if a.get("match_id"))))
    logger.info("Saved to: %s", OUT_FILE)


if __name__ == "__main__":
    main()
