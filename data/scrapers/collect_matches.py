"""
Collect match data from football-data.org API.

Fetches historical match results for the top 5 European leagues
and saves them as a structured CSV for feature engineering.

Usage:
    python data/scrapers/collect_matches.py
    python data/scrapers/collect_matches.py --seasons 2022 2023 2024 --leagues PL BL1 SA
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load API key from environment
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    except ImportError:
        pass


BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY or ""}

# Rate limiting: 10 calls per minute → 6 seconds between calls
RATE_LIMIT_DELAY = 6.5

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
OUT_FILE = RAW_DIR / "matches_raw.csv"

# League codes → names
LEAGUES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
}


def fetch_matches(league_code: str, season: int) -> List[Dict]:
    """
    Fetch all finished matches for a specific league and season.

    Args:
        league_code: League identifier (e.g., 'PL', 'BL1', 'SA')
        season: Starting year of the season (e.g., 2023 for 2023/24)

    Returns:
        List of match dictionaries from the API
    """
    if not API_KEY:
        logger.error(
            "FOOTBALL_DATA_API_KEY not set. "
            "Register at https://www.football-data.org/ and add to .env"
        )
        return []

    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {"season": season, "status": "FINISHED"}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()

        matches = response.json().get("matches", [])
        logger.info("Fetched %d matches for %s %d/%d", len(matches), league_code, season, season + 1)

        time.sleep(RATE_LIMIT_DELAY)
        return matches

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            return fetch_matches(league_code, season)
        logger.error("HTTP error fetching %s %d: %s", league_code, season, e)
        return []
    except requests.exceptions.RequestException as e:
        logger.error("Request error fetching %s %d: %s", league_code, season, e)
        return []


def fetch_standings(league_code: str, season: int) -> Dict[int, int]:
    """
    Fetch team standings for a league/season.

    Returns:
        Dictionary mapping team_id → league_position
    """
    url = f"{BASE_URL}/competitions/{league_code}/standings"
    params = {"season": season}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()

        standings = {}
        for table in response.json().get("standings", []):
            if table.get("type") == "TOTAL":
                for entry in table.get("table", []):
                    team_id = entry.get("team", {}).get("id")
                    position = entry.get("position")
                    if team_id and position:
                        standings[team_id] = position

        time.sleep(RATE_LIMIT_DELAY)
        return standings

    except requests.exceptions.RequestException as e:
        logger.error("Error fetching standings for %s %d: %s", league_code, season, e)
        return {}


def process_match_data(matches: List[Dict], standings: Dict[int, int] = None) -> pd.DataFrame:
    """
    Transform raw API match data into a structured DataFrame.

    Args:
        matches: List of match dictionaries from the API
        standings: Optional team_id → league_position mapping

    Returns:
        DataFrame with one row per match
    """
    if standings is None:
        standings = {}

    processed = []
    for match in matches:
        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")
        home_score = match.get("score", {}).get("fullTime", {}).get("home")
        away_score = match.get("score", {}).get("fullTime", {}).get("away")

        if home_score is None or away_score is None:
            continue

        # Determine outcome
        if home_score > away_score:
            outcome = "Home Win"
        elif home_score < away_score:
            outcome = "Away Win"
        else:
            outcome = "Draw"

        row = {
            "match_id": match.get("id"),
            "date": match.get("utcDate"),
            "home_team": match.get("homeTeam", {}).get("name"),
            "away_team": match.get("awayTeam", {}).get("name"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_score": home_score,
            "away_score": away_score,
            "outcome": outcome,
            "matchday": match.get("matchday"),
            "competition": match.get("competition", {}).get("name"),
            "competition_code": match.get("competition", {}).get("code"),
            "season": match.get("season", {}).get("startDate", "").split("-")[0],
            "league_position_home": standings.get(home_id),
            "league_position_away": standings.get(away_id),
        }
        processed.append(row)

    return pd.DataFrame(processed)


def main():
    """Main data collection pipeline."""
    parser = argparse.ArgumentParser(description="Collect match data from football-data.org API.")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024],
                        help="Season start years (e.g., 2022 2023 2024)")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()),
                        help="League codes (e.g., PL BL1 SA PD FL1)")
    args = parser.parse_args()

    all_matches = []

    for league in args.leagues:
        for season in args.seasons:
            logger.info("Collecting %s %d/%d...", LEAGUES.get(league, league), season, season + 1)

            matches = fetch_matches(league, season)
            standings = fetch_standings(league, season)

            if matches:
                df = process_match_data(matches, standings)
                all_matches.append(df)
                logger.info("  → %d matches processed", len(df))

    if not all_matches:
        logger.error("No matches collected. Check your API key and network connection.")
        return

    final_df = pd.concat(all_matches, ignore_index=True)

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)

    # Summary
    logger.info("=" * 50)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 50)
    logger.info("Total matches: %d", len(final_df))
    logger.info("Leagues: %s", final_df["competition"].unique().tolist())
    logger.info("Seasons: %s", sorted(final_df["season"].unique().tolist()))
    logger.info("Outcome distribution:\n%s", final_df["outcome"].value_counts().to_string())
    logger.info("Saved to: %s", OUT_FILE)


if __name__ == "__main__":
    main()
