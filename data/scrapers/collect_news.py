"""
Collect pre-match news articles for sentiment analysis.

Uses multiple strategies:
  1. NewsAPI (works for matches within last 30 days)
  2. The Guardian Open API (free, unlimited archive)
  3. Google News RSS (no key needed, good coverage)

Usage:
    python data/scrapers/collect_news.py
    python data/scrapers/collect_news.py --limit 300
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "test")  # "test" is the free tier key

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
OUT_FILE = RAW_DIR / "news_articles.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def _simplify_team(name: str) -> str:
    """Remove common suffixes for better search results."""
    for suffix in [" FC", "FC ", " CF", "CF ", " SC", "SC "]:
        name = name.replace(suffix, "")
    return name.strip()


# ── Source 1: NewsAPI ─────────────────────────────────────────────────────────

def get_articles_newsapi(
    home_team: str, away_team: str, match_date: str, days_before: int = 7
) -> List[Dict]:
    """Fetch from NewsAPI (only works for recent matches ~30 days)."""
    if not NEWS_API_KEY:
        return []

    try:
        match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00").split("T")[0])
    except ValueError:
        match_dt = datetime.strptime(match_date[:10], "%Y-%m-%d")

    from_date = match_dt - timedelta(days=days_before)
    to_date = match_dt - timedelta(days=1)

    home_simple = _simplify_team(home_team)
    away_simple = _simplify_team(away_team)
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
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if resp.status_code == 426:  # "Upgrade required" = date too old
            return []
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        time.sleep(0.5)
        return articles
    except requests.exceptions.RequestException as e:
        logger.debug("NewsAPI: %s", e)
        return []


# ── Source 2: The Guardian Open Platform ──────────────────────────────────────

def get_articles_guardian(
    home_team: str, away_team: str, match_date: str, days_before: int = 7
) -> List[Dict]:
    """Fetch from The Guardian API (free 'test' key, full archive)."""
    try:
        match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00").split("T")[0])
    except ValueError:
        match_dt = datetime.strptime(match_date[:10], "%Y-%m-%d")

    from_date = match_dt - timedelta(days=days_before)
    to_date = match_dt - timedelta(days=1)

    home_simple = _simplify_team(home_team)
    away_simple = _simplify_team(away_team)

    params = {
        "q": f"{home_simple} {away_simple}",
        "section": "football",
        "from-date": from_date.strftime("%Y-%m-%d"),
        "to-date": to_date.strftime("%Y-%m-%d"),
        "page-size": 5,
        "show-fields": "bodyText,headline",
        "order-by": "relevance",
        "api-key": GUARDIAN_API_KEY,
    }

    try:
        resp = requests.get("https://content.guardianapis.com/search", params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("response", {}).get("results", [])

        articles = []
        for r in results:
            body = r.get("fields", {}).get("bodyText", "")
            if body and len(body) > 100:
                articles.append({
                    "title": r.get("fields", {}).get("headline", r.get("webTitle", "")),
                    "content": body,
                    "url": r.get("webUrl", ""),
                    "source": {"name": "The Guardian"},
                    "publishedAt": r.get("webPublicationDate", ""),
                })

        time.sleep(0.5)
        return articles

    except requests.exceptions.RequestException as e:
        logger.debug("Guardian: %s", e)
        return []


# ── Source 3: Google News RSS ─────────────────────────────────────────────────

def get_articles_google_news(home_team: str, away_team: str) -> List[Dict]:
    """Scrape Google News RSS feed (no API key needed)."""
    home_simple = _simplify_team(home_team)
    away_simple = _simplify_team(away_team)
    query = quote_plus(f"{home_simple} vs {away_simple} preview")

    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml-xml")

        articles = []
        items = soup.find_all("item")[:5]

        for item in items:
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            source_tag = item.find("source")

            if title:
                articles.append({
                    "title": title.get_text(strip=True),
                    "content": title.get_text(strip=True),  # RSS only gives title
                    "url": link.get_text(strip=True) if link else "",
                    "source": {"name": source_tag.get_text(strip=True) if source_tag else "Google News"},
                    "publishedAt": pub_date.get_text(strip=True) if pub_date else "",
                })

        time.sleep(1)
        return articles

    except Exception as e:
        logger.debug("Google News RSS: %s", e)
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Main news collection pipeline with multi-source fallback."""
    parser = argparse.ArgumentParser(description="Collect pre-match news articles.")
    parser.add_argument(
        "--matches", type=Path, default=RAW_DIR / "matches_raw.csv",
        help="Path to matches CSV"
    )
    parser.add_argument("--limit", type=int, default=200, help="Max matches to process")
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
    matches_with_articles = 0

    for idx, (_, match) in enumerate(matches_df.iterrows()):
        home = match["home_team"]
        away = match["away_team"]
        date_str = str(match.get("date", ""))

        if (idx + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d matches processed (%d articles so far)",
                idx + 1, len(matches_df), len(all_articles)
            )

        # Try sources in order of quality
        articles = []

        # 1. Guardian API (best for historical, full article text)
        if not articles:
            articles = get_articles_guardian(home, away, date_str, args.days_before)

        # 2. NewsAPI (works for recent matches)
        if not articles:
            articles = get_articles_newsapi(home, away, date_str, args.days_before)

        # 3. Google News RSS (fallback, titles only)
        if not articles:
            articles = get_articles_google_news(home, away)

        if articles:
            matches_with_articles += 1

        for article in articles:
            all_articles.append({
                "match_id": match.get("match_id"),
                "home_team": home,
                "away_team": away,
                "match_date": date_str,
                "article_title": article.get("title", ""),
                "article_content": article.get("content") or article.get("description", ""),
                "article_url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "unknown"),
            })

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info("NEWS COLLECTION SUMMARY")
    logger.info("=" * 50)
    logger.info("Total articles: %d", len(all_articles))
    logger.info("Matches with articles: %d / %d (%.1f%%)",
                matches_with_articles, len(matches_df),
                matches_with_articles / max(len(matches_df), 1) * 100)
    logger.info("Sources: %s", dict(pd.Series([a["source"] for a in all_articles]).value_counts()))
    logger.info("Saved to: %s", OUT_FILE)


if __name__ == "__main__":
    main()
