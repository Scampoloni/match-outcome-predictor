"""
ML pipeline smoke tests. Run with: pytest tests/test_ml_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "ml_classification"))
from train import STATS_FEATURES, NLP_FEATURES, LABEL_ORDER


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dummy_df(n: int = 200, include_nlp: bool = True) -> pd.DataFrame:
    """Create a synthetic match feature dataframe for testing."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "elo_difference": rng.normal(0, 150, n),
        "form_difference": rng.normal(0, 1.5, n),
        "goals_per_game_home": rng.uniform(0.5, 3.0, n),
        "goals_per_game_away": rng.uniform(0.5, 3.0, n),
        "h2h_home_wins": rng.integers(0, 5, n).astype(float),
        "h2h_draws": rng.integers(0, 5, n).astype(float),
        "h2h_away_wins": rng.integers(0, 5, n).astype(float),
        "league_position_home": rng.integers(1, 21, n).astype(float),
        "league_position_away": rng.integers(1, 21, n).astype(float),
        "rest_days_home": rng.integers(2, 14, n).astype(float),
        "rest_days_away": rng.integers(2, 14, n).astype(float),
        "strength_ratio": rng.uniform(0.6, 1.6, n),
        "goal_difference_delta": rng.normal(0, 20, n),
    })
    if include_nlp:
        df["sentiment_gap"] = rng.uniform(-1, 1, n)
        df["confidence_score_home"] = rng.uniform(0, 1, n)
        df["confidence_score_away"] = rng.uniform(0, 1, n)
        df["injury_concern_score_home"] = rng.uniform(0, 1, n)
        df["injury_concern_score_away"] = rng.uniform(0, 1, n)
        df["hype_level"] = rng.uniform(0, 1, n)

    # Assign outcomes based on elo_difference for some signal
    def assign_outcome(row):
        if row["elo_difference"] > 80:
            return "Home Win"
        elif row["elo_difference"] < -80:
            return "Away Win"
        return "Draw"
    df["outcome"] = df.apply(assign_outcome, axis=1)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_feature_columns_present():
    df = make_dummy_df()
    for col in STATS_FEATURES:
        assert col in df.columns, f"Missing stats column: {col}"


def test_label_encoding():
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    encoded = le.transform(LABEL_ORDER)
    assert len(encoded) == 3
    assert set(encoded) == {0, 1, 2}


def test_random_forest_smoke():
    df = make_dummy_df(include_nlp=False)
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df["outcome"])
    X = df[STATS_FEATURES].fillna(0)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.4, f"Smoke test accuracy too low: {acc:.2f}"


def test_nlp_features_improve_accuracy():
    """NLP features should not hurt (on dummy data, difference may be tiny)."""
    df = make_dummy_df(include_nlp=True)
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df["outcome"])

    X_stats = df[STATS_FEATURES].fillna(0)
    X_nlp = df[STATS_FEATURES + NLP_FEATURES].fillna(0)

    clf_stats = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_stats, y)
    clf_nlp = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_nlp, y)

    acc_stats = (clf_stats.predict(X_stats) == y).mean()
    acc_nlp = (clf_nlp.predict(X_nlp) == y).mean()

    # On dummy data both should be decent; NLP shouldn't hurt
    assert acc_nlp >= acc_stats - 0.05, (
        f"NLP features hurt accuracy: stats={acc_stats:.3f}, nlp={acc_nlp:.3f}"
    )


def test_label_order():
    assert LABEL_ORDER == ["Away Win", "Draw", "Home Win"]
