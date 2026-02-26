"""
Data quality validation for the match outcome predictor.

Run after data collection to catch issues early before training.

Usage:
    python data/scrapers/data_validator.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "raw"

REQUIRED_ML_COLUMNS = [
    "home_team", "away_team", "date", "outcome",
    "score_home", "score_away",
]
REQUIRED_NLP_COLUMNS = [
    "match_id", "home_team", "away_team",
    "sentiment_mean_home", "sentiment_mean_away", "sentiment_gap",
]


def check_file_exists(path: Path, label: str) -> bool:
    if not path.exists():
        log.error("Missing file: %s (%s)", path, label)
        return False
    log.info("Found: %s", path.name)
    return True


def check_dataframe(df: pd.DataFrame, required_cols: list[str], label: str) -> bool:
    ok = True
    # Column completeness
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        log.error("[%s] Missing columns: %s", label, missing_cols)
        ok = False

    # Row count
    log.info("[%s] Rows: %d", label, len(df))
    if len(df) < 100:
        log.warning("[%s] Dataset seems small (< 100 rows). Aim for 500+.", label)

    # Missing values
    if required_cols:
        existing = [c for c in required_cols if c in df.columns]
        null_counts = df[existing].isnull().sum()
        high_null = null_counts[null_counts > len(df) * 0.1]
        if not high_null.empty:
            log.warning("[%s] High null rate (>10%%) in columns:\n%s", label, high_null.to_string())

    return ok


def validate_matches() -> bool:
    path = PROCESSED_DIR / "matches.csv"
    if not check_file_exists(path, "matches"):
        return False
    df = pd.read_csv(path)
    ok = check_dataframe(df, REQUIRED_ML_COLUMNS, "matches")

    # Outcome values
    if "outcome" in df.columns:
        valid_outcomes = {"Home Win", "Draw", "Away Win"}
        actual = set(df["outcome"].dropna().unique())
        invalid = actual - valid_outcomes
        if invalid:
            log.error("[matches] Invalid outcome values: %s", invalid)
            ok = False

    # Score sanity
    if "score_home" in df.columns and "score_away" in df.columns:
        neg_scores = ((df["score_home"] < 0) | (df["score_away"] < 0)).sum()
        if neg_scores > 0:
            log.error("[matches] %d rows with negative scores.", neg_scores)
            ok = False
        high_scores = ((df["score_home"] > 15) | (df["score_away"] > 15)).sum()
        if high_scores > 0:
            log.warning("[matches] %d rows with suspiciously high scores (>15).", high_scores)

    # Date range
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        log.info("[matches] Date range: %s to %s", dates.min(), dates.max())

    return ok


def validate_features_complete() -> bool:
    path = PROCESSED_DIR / "features_complete.csv"
    if not check_file_exists(path, "features_complete"):
        return False
    df = pd.read_csv(path)
    ok = check_dataframe(df, REQUIRED_ML_COLUMNS, "features_complete")

    # Check for key engineered features
    expected_features = ["elo_difference", "form_difference", "elo_home", "elo_away"]
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        log.warning("[features_complete] Missing engineered features: %s", missing)

    return ok


def validate_nlp_features() -> bool:
    path = PROCESSED_DIR / "nlp_features.csv"
    if not check_file_exists(path, "nlp_features"):
        return False
    df = pd.read_csv(path)
    ok = check_dataframe(df, REQUIRED_NLP_COLUMNS, "nlp_features")

    # Score range checks
    for col in ["sentiment_mean_home", "sentiment_mean_away", "sentiment_gap"]:
        if col in df.columns:
            vals = df[col].dropna()
            if ((vals < -1) | (vals > 1)).any():
                log.error("[nlp_features] %s values outside [-1, 1]", col)
                ok = False
    if "hype_level" in df.columns:
        vals = df["hype_level"].dropna()
        if ((vals < 0) | (vals > 1)).any():
            log.error("[nlp_features] hype_level outside [0, 1]")
            ok = False
    return ok


def validate_outcome_distribution() -> bool:
    path = PROCESSED_DIR / "features_complete.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        return False
    dist = df["outcome"].value_counts(normalize=True)
    log.info("Outcome distribution:\n%s", dist.to_string())
    # Warn if any class < 10% (matches are roughly 45/27/28 H/D/A)
    if (dist < 0.10).any():
        log.warning("Some outcome classes have < 10%% of samples — check data quality.")
    return True


def main():
    log.info("=== Data Validation ===")
    results = [
        validate_matches(),
        validate_features_complete(),
        validate_nlp_features(),
        validate_outcome_distribution(),
    ]
    if all(results):
        log.info("All validations passed.")
        sys.exit(0)
    else:
        log.error("Some validations failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
