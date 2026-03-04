"""
Shared utilities for the Streamlit app.
Loads models and match data, runs predictions.
"""

from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = ROOT / "models" / "ml_classification" / "saved_models"
PROCESSED_DIR = ROOT / "data" / "processed"

LABEL_ORDER = ["Away Win", "Draw", "Home Win"]
LABEL_COLORS = {
    "Home Win": "#4caf50",
    "Draw": "#ff9800",
    "Away Win": "#f44336",
}


@lru_cache(maxsize=1)
def load_feature_matrix() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "features_complete.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@lru_cache(maxsize=4)
def load_model(name: str = "xgboost", suffix: str = "with_nlp") -> dict | None:
    path = SAVED_MODELS_DIR / f"{name}_{suffix}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def load_match_data(home_team: str, away_team: str) -> pd.Series | None:
    """Return the latest feature row for a home/away matchup."""
    df = load_feature_matrix()
    if df is None or "home_team" not in df.columns:
        return None
    mask = (
        df["home_team"].str.lower().str.contains(home_team.lower(), na=False)
        & df["away_team"].str.lower().str.contains(away_team.lower(), na=False)
    )
    hits = df[mask]
    if hits.empty:
        return None
    # Return the most recent match (last row)
    return hits.iloc[-1]


def predict_match_outcome(match_row: pd.Series, model_name: str = "xgboost", use_nlp: bool = True) -> dict:
    """Run the classifier and return prediction details."""
    suffix = "with_nlp" if use_nlp else "no_nlp"
    bundle = load_model(model_name, suffix)
    if bundle is None:
        return {"label": "Unknown", "probabilities": {}, "error": "Model not loaded"}

    model = bundle["model"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    X = match_row[features].fillna(0).values.reshape(1, -1)
    X = pd.DataFrame(X, columns=features)

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]

    return {
        "label": pred_label,
        "probabilities": dict(zip(LABEL_ORDER, proba.tolist())),
        "confidence": float(proba[pred_idx]),
    }


def get_team_id_mapping() -> dict[str, int]:
    """Returns a mapping of team names to their football-data.org IDs."""

    
    LEAGUES = {"PL": "Premier League", "PD": "La Liga", 
               "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}
    headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
    mapping = {}
    
    for code in LEAGUES.keys():
        url = f"https://api.football-data.org/v4/competitions/{code}/teams"
        try:
            resp = requests.get(url, headers=headers).json()
            for t in resp.get("teams", []):
                mapping[t["name"]] = t["id"]
        except Exception:
            pass
    return mapping

@st.cache_data(ttl=86400)
def get_current_teams() -> dict[str, list[str]]:
    """Returns a dictionary of leagues to lists of current team names."""

    
    LEAGUES = {"PL": "Premier League", "PD": "La Liga", 
               "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}
    headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
    teams = {}
    
    for code, name in LEAGUES.items():
        url = f"https://api.football-data.org/v4/competitions/{code}/teams"
        try:
            resp = requests.get(url, headers=headers).json()
            teams[name] = sorted([t["name"] for t in resp.get("teams", [])])
        except Exception:
            # Fallback to CSV if API fails/rate limits
            df = load_feature_matrix()
            if df is not None:
                t_set = set(df[df["competition"] == name]["home_team"].dropna().unique())
                t_set.update(df[df["competition"] == name]["away_team"].dropna().unique())
                teams[name] = sorted(list(t_set))
            else:
                teams[name] = []
                
    return teams
