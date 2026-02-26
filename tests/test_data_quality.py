"""
Data quality tests. Run with: pytest tests/test_data_quality.py
"""

import pytest
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def features_df():
    path = PROCESSED_DIR / "features_complete.csv"
    if not path.exists():
        pytest.skip("features_complete.csv not found — run the data pipeline first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nlp_df():
    path = PROCESSED_DIR / "nlp_features.csv"
    if not path.exists():
        pytest.skip("nlp_features.csv not found — run sentiment_analyzer.py first.")
    return pd.read_csv(path)


def test_features_minimum_rows(features_df):
    assert len(features_df) >= 200, f"Expected ≥200 rows, got {len(features_df)}"


def test_features_has_target_column(features_df):
    assert "outcome" in features_df.columns


def test_outcome_values(features_df):
    valid = {"Home Win", "Draw", "Away Win"}
    actual = set(features_df["outcome"].dropna().unique())
    assert actual.issubset(valid), f"Unexpected outcomes: {actual - valid}"


def test_outcome_balance(features_df):
    dist = features_df["outcome"].value_counts(normalize=True)
    for label, pct in dist.items():
        assert pct >= 0.10, f"Outcome '{label}' has only {pct*100:.1f}% — check data quality"


def test_nlp_sentiment_range(nlp_df):
    for col in ["sentiment_mean_home", "sentiment_mean_away"]:
        assert col in nlp_df.columns, f"Missing column: {col}"
        vals = nlp_df[col].dropna()
        assert (vals >= -1).all() and (vals <= 1).all(), f"{col} outside [-1, 1]"


def test_nlp_hype_range(nlp_df):
    col = "hype_level"
    assert col in nlp_df.columns
    vals = nlp_df[col].dropna()
    assert (vals >= 0).all() and (vals <= 1).all(), "Hype level outside [0, 1]"


def test_features_has_team_columns(features_df):
    assert "home_team" in features_df.columns, "Missing home_team column"
    assert "away_team" in features_df.columns, "Missing away_team column"


def test_no_negative_scores(features_df):
    if "score_home" in features_df.columns:
        assert (features_df["score_home"] >= 0).all(), "Negative home scores found"
    if "score_away" in features_df.columns:
        assert (features_df["score_away"] >= 0).all(), "Negative away scores found"
