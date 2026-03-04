"""
Preprocess raw match data into a clean format ready for feature engineering.

Bridges the gap between data collection (data/raw/matches_raw.csv) and
feature engineering (models/nlp_analysis/feature_extractor.py).

Adds rolling statistics:
  - goals_per_game (home/away)
  - goals_conceded_per_game (home/away)
  - home_advantage_score
  - rest days between matches
  - goal_difference (season cumulative)

Usage:
    python data/scrapers/build_features.py
    python data/scrapers/build_features.py --input data/raw/matches_raw.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_goals_per_game(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling goals scored and conceded per game for each team.

    Uses the last N matches played by each team to calculate averages.
    Adds: goals_per_game_home, goals_per_game_away,
          goals_conceded_per_game_home, goals_conceded_per_game_away
    """
    df = df.sort_values("date").reset_index(drop=True)

    team_goals_scored: dict[str, list[int]] = {}
    team_goals_conceded: dict[str, list[int]] = {}

    gpg_home = []
    gpg_away = []
    gcpg_home = []
    gcpg_away = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_score = row.get("home_score", 0) or 0
        a_score = row.get("away_score", 0) or 0

        # Compute averages from BEFORE this match
        h_scored = team_goals_scored.get(home, [])
        h_conceded = team_goals_conceded.get(home, [])
        a_scored = team_goals_scored.get(away, [])
        a_conceded = team_goals_conceded.get(away, [])

        gpg_home.append(np.mean(h_scored[-window:]) if h_scored else 1.3)
        gcpg_home.append(np.mean(h_conceded[-window:]) if h_conceded else 1.1)
        gpg_away.append(np.mean(a_scored[-window:]) if a_scored else 1.1)
        gcpg_away.append(np.mean(a_conceded[-window:]) if a_conceded else 1.3)

        # Update histories
        team_goals_scored.setdefault(home, []).append(int(h_score))
        team_goals_conceded.setdefault(home, []).append(int(a_score))
        team_goals_scored.setdefault(away, []).append(int(a_score))
        team_goals_conceded.setdefault(away, []).append(int(h_score))

    df["goals_per_game_home"] = np.round(gpg_home, 4)
    df["goals_per_game_away"] = np.round(gpg_away, 4)
    df["goals_conceded_per_game_home"] = np.round(gcpg_home, 4)
    df["goals_conceded_per_game_away"] = np.round(gcpg_away, 4)
    return df


def compute_home_advantage(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute home advantage score: rolling home win rate per team.
    Adds: home_advantage_score
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_home_results: dict[str, list[float]] = {}
    scores = []

    for _, row in df.iterrows():
        home = row["home_team"]
        history = team_home_results.get(home, [])
        scores.append(np.mean(history[-window:]) if history else 0.5)

        outcome = row.get("outcome", "Draw")
        if outcome == "Home Win":
            team_home_results.setdefault(home, []).append(1.0)
        elif outcome == "Draw":
            team_home_results.setdefault(home, []).append(0.33)
        else:
            team_home_results.setdefault(home, []).append(0.0)

    df["home_advantage_score"] = np.round(scores, 4)
    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days since last match for each team.
    Adds: days_since_last_match_home, days_since_last_match_away
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_last_match: dict[str, pd.Timestamp] = {}
    rest_home = []
    rest_away = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = pd.Timestamp(row["date"])

        last_home = team_last_match.get(home)
        last_away = team_last_match.get(away)

        if last_home is not None:
            rest_home.append((match_date - last_home).days)
        else:
            rest_home.append(7)  # default

        if last_away is not None:
            rest_away.append((match_date - last_away).days)
        else:
            rest_away.append(7)

        team_last_match[home] = match_date
        team_last_match[away] = match_date

    df["days_since_last_match_home"] = rest_home
    df["days_since_last_match_away"] = rest_away
    return df


def compute_season_goal_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative goal difference per team per season.
    Adds: goal_difference_home, goal_difference_away
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_season_gd: dict[tuple[str, str], int] = {}
    gd_home = []
    gd_away = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        season = str(row.get("season", "unknown"))
        h_score = int(row.get("home_score", 0) or 0)
        a_score = int(row.get("away_score", 0) or 0)

        # Current GD before this match
        gd_home.append(team_season_gd.get((home, season), 0))
        gd_away.append(team_season_gd.get((away, season), 0))

        # Update
        team_season_gd[(home, season)] = team_season_gd.get((home, season), 0) + (h_score - a_score)
        team_season_gd[(away, season)] = team_season_gd.get((away, season), 0) + (a_score - h_score)

    df["goal_difference_home"] = gd_home
    df["goal_difference_away"] = gd_away
    return df


def compute_win_rate(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Compute rolling win rate per team.
    Adds: win_rate_home, win_rate_away
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_results: dict[str, list[float]] = {}
    wr_home = []
    wr_away = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        outcome = row.get("outcome", "Draw")

        h_hist = team_results.get(home, [])
        a_hist = team_results.get(away, [])
        wr_home.append(np.mean(h_hist[-window:]) if h_hist else 0.33)
        wr_away.append(np.mean(a_hist[-window:]) if a_hist else 0.33)

        if outcome == "Home Win":
            team_results.setdefault(home, []).append(1.0)
            team_results.setdefault(away, []).append(0.0)
        elif outcome == "Away Win":
            team_results.setdefault(home, []).append(0.0)
            team_results.setdefault(away, []).append(1.0)
        else:
            team_results.setdefault(home, []).append(0.0)
            team_results.setdefault(away, []).append(0.0)

    df["win_rate_home"] = np.round(wr_home, 4)
    df["win_rate_away"] = np.round(wr_away, 4)
    return df


def main():
    """Preprocess raw matches into a clean, feature-enriched dataset."""
    parser = argparse.ArgumentParser(description="Preprocess raw match data.")
    parser.add_argument(
        "--input", type=Path,
        default=RAW_DIR / "matches_raw.csv",
        help="Path to raw matches CSV"
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Raw matches file not found: %s. Run collect_matches.py first.", args.input)
        return

    df = pd.read_csv(args.input)
    log.info("Loaded %d raw matches from %s", len(df), args.input)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["match_id"], keep="first")
    if len(df) < before:
        log.info("Removed %d duplicate matches", before - len(df))

    # Ensure outcome column
    if "outcome" not in df.columns:
        log.error("Missing 'outcome' column in raw data.")
        return

    log.info("Outcome distribution:\n%s", df["outcome"].value_counts().to_string())

    # Compute rolling features
    log.info("Computing goals per game...")
    df = compute_goals_per_game(df)
    log.info("Computing home advantage score...")
    df = compute_home_advantage(df)
    log.info("Computing rest days...")
    df = compute_rest_days(df)
    log.info("Computing season goal difference...")
    df = compute_season_goal_difference(df)
    log.info("Computing win rates...")
    df = compute_win_rate(df)

    # Save
    out_path = PROCESSED_DIR / "matches.csv"
    df.to_csv(out_path, index=False)

    log.info("=" * 50)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 50)
    log.info("Total matches: %d", len(df))
    log.info("Columns: %s", list(df.columns))
    log.info("Date range: %s — %s", df["date"].min(), df["date"].max())
    log.info("Leagues: %s", df["competition"].unique().tolist() if "competition" in df.columns else "N/A")
    log.info("Saved to: %s", out_path)


if __name__ == "__main__":
    main()
