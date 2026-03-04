import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from functools import lru_cache
from dateutil.relativedelta import relativedelta

from utils import load_feature_matrix

@st.cache_data(ttl=86400)
def get_live_elo(team_name: str) -> float:
    """Fetch current ELO rating from api.clubelo.com."""
    try:
        # ClubELO expects no spaces in the team name.
        clean_name = team_name.replace(" ", "")
        url = f"http://api.clubelo.com/{clean_name}"
        df = pd.read_csv(url)
        if not df.empty:
            # Drop NaN Elo rows incase
            df = df.dropna(subset=['Elo'])
            # Sort by date just in case and return latest
            df['To'] = pd.to_datetime(df['To'])
            df = df.sort_values('To')
            return float(df.iloc[-1]['Elo'])
    except Exception:
        pass
    return 1500.0  # Safe default


@st.cache_data(ttl=3600)
def fetch_recent_matches(team_id: int) -> list[dict]:
    """Fetch 10 recent finished matches for a team from football-data API."""
    try:
        headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
        url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=10"
        resp = requests.get(url, headers=headers).json()
        return resp.get("matches", [])
    except Exception:
        return []
        
@st.cache_data(ttl=3600)
def fetch_league_standing(team_id: int) -> int:
    """Fetch current league position for a team from football-data API."""
    # We first need to know what competition they are currently in.
    try:
        headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
        url = f"https://api.football-data.org/v4/teams/{team_id}"
        resp_team = requests.get(url, headers=headers).json()
        active_comps = resp_team.get("runningCompetitions", [])
        if active_comps:
            comp_code = active_comps[0]["code"]
            url_standings = f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
            resp_standings = requests.get(url_standings, headers=headers).json()
            standings = resp_standings.get("standings", [])
            if standings:
                for row in standings[0].get("table", []):
                    if row["team"]["id"] == team_id:
                        return row["position"]
    except Exception:
        pass
        
    return 10  # Median position as safe default

def calculate_recent_stats(matches: list[dict], team_id: int) -> tuple[float, float, float, float]:
    """Calculates form, goals_per_game, goals_conceded_per_game, win_rate from recent matches."""
    if not matches:
        return 1.0, 1.0, 1.0, 0.3  # Safe medians
        
    points = 0
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    
    for m in matches[-5:]: # Form based on last 5
        home_team_id = m["homeTeam"]["id"]
        home_score = m["score"]["fullTime"]["home"]
        away_score = m["score"]["fullTime"]["away"]
        
        if home_score is None or away_score is None:
            continue
            
        if home_team_id == team_id:
            # Team is Home
            goals_scored += home_score
            goals_conceded += away_score
            if home_score > away_score:
                points += 3
                wins += 1
            elif home_score == away_score:
                points += 1
        else:
            # Team is Away
            goals_scored += away_score
            goals_conceded += home_score
            if away_score > home_score:
                points += 3
                wins += 1
            elif away_score == home_score:
                points += 1
                
    n = len(matches[-5:])
    if n == 0: return 1.0, 1.0, 1.0, 0.3
    
    avg_pts = float(points / n)
    avg_gf = float(goals_scored / n)
    avg_ga = float(goals_conceded / n)
    win_r = float(wins / n)
    
    return avg_pts, avg_gf, avg_ga, win_r


def build_live_features(home_team_id: int, away_team_id: int, 
                        home_team_name: str, away_team_name: str) -> dict:
    """
    Berechnet aktuelle Feature-Vektoren für das XGBoost-Modell on the fly.
    """
    
    # 1. Fetch API Data
    home_matches = fetch_recent_matches(home_team_id)
    away_matches = fetch_recent_matches(away_team_id)
    
    home_pos = fetch_league_standing(home_team_id)
    away_pos = fetch_league_standing(away_team_id)
    
    # 2. Base stats calculations
    h_form, h_gf, h_ga, h_win_rate = calculate_recent_stats(home_matches, home_team_id)
    a_form, a_gf, a_ga, a_win_rate = calculate_recent_stats(away_matches, away_team_id)
    
    # 3. ELO Ratings calculations
    h_elo = get_live_elo(home_team_name)
    a_elo = get_live_elo(away_team_name)
    
    # 4. Synthesize final dictionary strictly enforcing features_complete.csv names
    features = {
        "league_position_home": float(home_pos),
        "league_position_away": float(away_pos),
        "goals_per_game_home": h_gf,
        "goals_per_game_away": a_gf,
        "goals_conceded_per_game_home": h_ga,
        "goals_conceded_per_game_away": a_ga,
        "home_advantage_score": 1.0,  # Constant fixed
        "days_since_last_match_home": 7.0, # Approximate, could calculate exactly from match date
        "days_since_last_match_away": 7.0,
        "goal_difference_home": h_gf - h_ga,
        "goal_difference_away": a_gf - a_ga,
        "win_rate_home": h_win_rate,
        "win_rate_away": a_win_rate,
        "elo_home": h_elo,
        "elo_away": a_elo,
        "elo_difference": h_elo - a_elo,
        "form_home": h_form,
        "form_away": a_form,
        "form_difference": h_form - a_form,
        "h2h_home_wins": 0, # Difficult to pull historically live, fallback to 0
        "h2h_draws": 0,
        "h2h_away_wins": 0,
        "strength_ratio": h_elo / a_elo if a_elo > 0 else 1.0,
        "goal_difference_delta": (h_gf - h_ga) - (a_gf - a_ga),
        "rest_days_home": 7.0,
        "rest_days_away": 7.0,
        # Default empty NLP logic - will be updated asynchronously if requested 
        "article_count": 0.0,
        "sentiment_mean_home": 0.0,
        "sentiment_mean_away": 0.0,
        "sentiment_std_home": 0.0,
        "sentiment_std_away": 0.0,
        "sentiment_gap": 0.0,
        "confidence_score_home": 0.0,
        "confidence_score_away": 0.0,
        "injury_concern_score_home": 0.0,
        "injury_concern_score_away": 0.0,
        "hype_level": 0.0,
        "pressure_keywords_home": 0.0,
        "pressure_keywords_away": 0.0,
        "morale_keywords_home": 0.0,
        "morale_keywords_away": 0.0,
    }
    
    return features
