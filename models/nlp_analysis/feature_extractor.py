"""
Merges NLP features into the main match feature matrix.

Reads data/processed/nlp_features.csv and data/processed/matches.csv,
assigns the match outcome label, and writes data/processed/features_complete.csv.

Usage:
    python models/nlp_analysis/feature_extractor.py
    python models/nlp_analysis/feature_extractor.py --matches data/processed/matches.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_elo(matches: pd.DataFrame, k: int = 20, initial: int = 1500) -> pd.DataFrame:
    """
    Compute running ELO ratings for each team based on match history.
    Returns the input dataframe with added elo_home, elo_away, elo_difference columns.
    """
    elo = {}
    elo_home_list = []
    elo_away_list = []

    matches = matches.sort_values("date").reset_index(drop=True)

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_elo = elo.get(home, initial)
        a_elo = elo.get(away, initial)

        elo_home_list.append(h_elo)
        elo_away_list.append(a_elo)

        # Expected scores
        e_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        e_away = 1 - e_home

        # Actual scores
        outcome = row.get("outcome", "")
        if outcome == "Home Win":
            s_home, s_away = 1.0, 0.0
        elif outcome == "Away Win":
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        elo[home] = h_elo + k * (s_home - e_home)
        elo[away] = a_elo + k * (s_away - e_away)

    matches["elo_home"] = elo_home_list
    matches["elo_away"] = elo_away_list
    matches["elo_difference"] = matches["elo_home"] - matches["elo_away"]
    return matches


def compute_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute rolling form (points per game in last N matches) for each team.
    Adds form_home, form_away, form_difference columns.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    team_results: dict[str, list[float]] = {}

    form_home_list = []
    form_away_list = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        outcome = row.get("outcome", "Draw")

        # Record current form before updating
        home_history = team_results.get(home, [])
        away_history = team_results.get(away, [])

        form_home_list.append(np.mean(home_history[-window:]) if home_history else 1.0)
        form_away_list.append(np.mean(away_history[-window:]) if away_history else 1.0)

        # Update history
        if outcome == "Home Win":
            team_results.setdefault(home, []).append(3.0)
            team_results.setdefault(away, []).append(0.0)
        elif outcome == "Away Win":
            team_results.setdefault(home, []).append(0.0)
            team_results.setdefault(away, []).append(3.0)
        else:
            team_results.setdefault(home, []).append(1.0)
            team_results.setdefault(away, []).append(1.0)

    matches["form_home"] = form_home_list
    matches["form_away"] = form_away_list
    matches["form_difference"] = matches["form_home"] - matches["form_away"]
    return matches


def compute_h2h(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute head-to-head stats for each matchup (last N encounters).
    Adds h2h_home_wins, h2h_draws, h2h_away_wins columns.
    """
    h2h_records: dict[tuple, list[str]] = {}
    h2h_home_wins = []
    h2h_draws = []
    h2h_away_wins = []

    matches = matches.sort_values("date").reset_index(drop=True)

    for _, row in matches.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        history = h2h_records.get(key, [])

        hw = sum(1 for r in history[-window:] if r == row["home_team"])
        aw = sum(1 for r in history[-window:] if r == row["away_team"])
        dr = sum(1 for r in history[-window:] if r == "Draw")

        h2h_home_wins.append(hw)
        h2h_draws.append(dr)
        h2h_away_wins.append(aw)

        outcome = row.get("outcome", "Draw")
        if outcome == "Home Win":
            h2h_records.setdefault(key, []).append(row["home_team"])
        elif outcome == "Away Win":
            h2h_records.setdefault(key, []).append(row["away_team"])
        else:
            h2h_records.setdefault(key, []).append("Draw")

    matches["h2h_home_wins"] = h2h_home_wins
    matches["h2h_draws"] = h2h_draws
    matches["h2h_away_wins"] = h2h_away_wins
    return matches


def engineer_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived match features from raw match data."""
    eps = 1e-6

    # Goals per game (from season stats if available)
    if "goals_home_season" in df.columns and "matches_played_home" in df.columns:
        df["goals_per_game_home"] = df["goals_home_season"] / (df["matches_played_home"] + eps)
    if "goals_away_season" in df.columns and "matches_played_away" in df.columns:
        df["goals_per_game_away"] = df["goals_away_season"] / (df["matches_played_away"] + eps)

    # Strength ratio
    if "elo_home" in df.columns and "elo_away" in df.columns:
        df["strength_ratio"] = df["elo_home"] / (df["elo_away"] + eps)

    # Goal difference delta (from standings)
    if "goal_difference_home" in df.columns and "goal_difference_away" in df.columns:
        df["goal_difference_delta"] = df["goal_difference_home"] - df["goal_difference_away"]

    # Rest days (if available)
    if "rest_days_home" not in df.columns:
        df["rest_days_home"] = 7  # default
    if "rest_days_away" not in df.columns:
        df["rest_days_away"] = 7

    return df


# ── Main merge ────────────────────────────────────────────────────────────────

def build_feature_matrix(matches_path: Path, nlp_path: Path) -> pd.DataFrame:
    log.info("Loading match data from %s", matches_path)
    matches = pd.read_csv(matches_path, low_memory=False)
    log.info("Matches loaded: %d", len(matches))

    # Ensure date is parsed
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        matches = matches.dropna(subset=["date"])

    # Ensure outcome column exists
    if "outcome" not in matches.columns:
        if "score_home" in matches.columns and "score_away" in matches.columns:
            def assign_outcome(row):
                if row["score_home"] > row["score_away"]:
                    return "Home Win"
                elif row["score_home"] < row["score_away"]:
                    return "Away Win"
                return "Draw"
            matches["outcome"] = matches.apply(assign_outcome, axis=1)
        else:
            log.error("Cannot determine match outcomes — missing score columns.")
            return pd.DataFrame()

    log.info("Outcome distribution:\n%s", matches["outcome"].value_counts().to_string())

    # Compute features
    log.info("Computing ELO ratings...")
    matches = compute_elo(matches)
    log.info("Computing form...")
    matches = compute_form(matches)
    log.info("Computing head-to-head stats...")
    matches = compute_h2h(matches)
    matches = engineer_match_features(matches)

    # Merge NLP features
    if nlp_path.exists():
        log.info("Loading NLP features from %s", nlp_path)
        nlp = pd.read_csv(nlp_path)
        if "match_id" in nlp.columns and "match_id" in matches.columns:
            merged = matches.merge(nlp.drop(columns=["home_team", "away_team"], errors="ignore"),
                                   on="match_id", how="left")
        else:
            log.warning("No match_id column for NLP merge. Attempting team-name merge.")
            merged = matches.merge(nlp, on=["home_team", "away_team"], how="left", suffixes=("", "_nlp"))
        nlp_coverage = merged["sentiment_mean_home"].notna().mean() if "sentiment_mean_home" in merged.columns else 0.0
        log.info("NLP feature coverage: %.1f%% of matches", nlp_coverage * 100)
    else:
        log.warning("NLP features file not found at %s. Proceeding without NLP.", nlp_path)
        merged = matches

    out_path = PROCESSED_DIR / "features_complete.csv"
    merged.to_csv(out_path, index=False)
    log.info("Feature matrix saved: %s (%d rows, %d cols)", out_path, len(merged), len(merged.columns))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build final match feature matrix.")
    parser.add_argument("--matches", type=Path, default=None)
    parser.add_argument("--nlp", type=Path, default=PROCESSED_DIR / "nlp_features.csv")
    args = parser.parse_args()

    # Determine matches file: prefer processed, fallback to raw
    if args.matches is not None:
        matches_path = args.matches
    elif (PROCESSED_DIR / "matches.csv").exists():
        matches_path = PROCESSED_DIR / "matches.csv"
    elif (RAW_DIR / "matches_raw.csv").exists():
        matches_path = RAW_DIR / "matches_raw.csv"
        log.warning(
            "Using raw matches file (no preprocessing applied). "
            "For best results, run: python data/scrapers/build_features.py"
        )
    else:
        log.error(
            "Match data not found.\n"
            "Step 1: python data/scrapers/collect_matches.py\n"
            "Step 2: python data/scrapers/build_features.py\n"
            "Step 3: python models/nlp_analysis/feature_extractor.py"
        )
        return

    build_feature_matrix(matches_path, args.nlp)


if __name__ == "__main__":
    main()
