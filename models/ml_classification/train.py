"""
ML training pipeline for match outcome classification.

Trains Logistic Regression, Random Forest, and XGBoost on match features,
with and without NLP features (for the ablation study).

Usage:
    python models/ml_classification/train.py
    python models/ml_classification/train.py --no-nlp   # ablation: stats only
"""

import argparse
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features used for ML (stats-only subset used in ablation)
STATS_FEATURES = [
    "elo_difference",
    "form_difference",
    "goals_per_game_home",
    "goals_per_game_away",
    "goals_conceded_per_game_home",
    "goals_conceded_per_game_away",
    "home_advantage_score",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "league_position_home",
    "league_position_away",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "strength_ratio",
    "goal_difference_delta",
]
NLP_FEATURES = [
    "sentiment_gap",
    "confidence_score_home",
    "confidence_score_away",
    "injury_concern_score_home",
    "injury_concern_score_away",
    "hype_level",
]
TARGET = "outcome"
LABEL_ORDER = ["Away Win", "Draw", "Home Win"]


def load_data(use_nlp: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and labels from processed CSV."""
    path = PROCESSED_DIR / "features_complete.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run the feature engineering pipeline first."
        )
    df = pd.read_csv(path)

    feature_cols = STATS_FEATURES.copy()
    if use_nlp:
        available_nlp = [c for c in NLP_FEATURES if c in df.columns]
        feature_cols.extend(available_nlp)
        log.info("Using NLP features: %s", available_nlp)
    else:
        log.info("Ablation mode: NLP features excluded.")

    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing columns (will be skipped): %s", missing)

    X = df[available].fillna(df[available].median(numeric_only=True))
    y = df[TARGET]
    return X, y


def encode_labels(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    le = LabelEncoder()
    le.fit(LABEL_ORDER)  # Safer than manually setting classes_
    y_enc = le.transform(y)
    return pd.Series(y_enc, name=TARGET), le


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        ),
    }


def train_and_save(use_nlp: bool = True):
    X, y = load_data(use_nlp)
    y_enc, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, stratify=y_enc, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85, stratify=y_train, random_state=42
    )

    log.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    suffix = "with_nlp" if use_nlp else "no_nlp"

    for name, model in build_models().items():
        log.info("Training: %s (%s)", name, suffix)
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_score = model.score(X_val, y_val)
        log.info("  CV F1-macro: %.4f ± %.4f | Val acc: %.4f", cv_scores.mean(), cv_scores.std(), val_score)

        out_path = SAVED_MODELS_DIR / f"{name}_{suffix}.pkl"
        joblib.dump({"model": model, "label_encoder": label_encoder, "features": list(X.columns)}, out_path)
        log.info("  Saved → %s", out_path)

    # Save test split for evaluation
    test_df = X_test.copy()
    test_df["label"] = y_test
    test_df.to_csv(SAVED_MODELS_DIR / f"test_split_{suffix}.csv", index=False)
    log.info("Test split saved.")


def main():
    parser = argparse.ArgumentParser(description="Train match outcome classifiers.")
    parser.add_argument("--no-nlp", action="store_true", help="Exclude NLP features (ablation study)")
    args = parser.parse_args()

    train_and_save(use_nlp=not args.no_nlp)


if __name__ == "__main__":
    main()
