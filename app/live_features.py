import requests
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from functools import lru_cache
from dateutil.relativedelta import relativedelta

from utils import load_feature_matrix

log = logging.getLogger(__name__)

# ── ClubELO Name Mapping ─────────────────────────────────────────────────────
# football-data.org team name  →  clubelo.com API name
# Verified against http://api.clubelo.com/{name} on 2026-03-05
CLUBELO_NAME_MAP = {
    # Bundesliga
    "FC Bayern München": "Bayern",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "RB Leipzig": "RBLeipzig",
    "SC Freiburg": "Freiburg",
    "Eintracht Frankfurt": "Frankfurt",
    "VfL Wolfsburg": "Wolfsburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "SV Werder Bremen": "Werder",
    "1. FSV Mainz 05": "Mainz",
    "FC Augsburg": "Augsburg",
    "VfL Bochum 1848": "Bochum",
    "1. FC Union Berlin": "UnionBerlin",
    "1. FC Köln": "Koeln",
    "SV Darmstadt 98": "Darmstadt",
    "1. FC Heidenheim 1846": "Heidenheim",
    "Borussia Mönchengladbach": "Gladbach",
    "FC St. Pauli 1910": "StPauli",
    "Holstein Kiel": "Holstein",
    # Premier League
    "Arsenal FC": "Arsenal",
    "Manchester City FC": "ManCity",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Manchester United FC": "ManUnited",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "West Ham United FC": "WestHam",
    "Aston Villa FC": "AstonVilla",
    "Brighton & Hove Albion FC": "Brighton",
    "Crystal Palace FC": "CrystalPalace",
    "Brentford FC": "Brentford",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "AFC Bournemouth": "Bournemouth",
    "Nottingham Forest FC": "Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "Burnley FC": "Burnley",
    "Sheffield United FC": "SheffieldUnited",
    "Luton Town FC": "Luton",
    "Leicester City FC": "Leicester",
    "Southampton FC": "Southampton",
    "Ipswich Town FC": "Ipswich",
    # La Liga
    "Real Madrid CF": "RealMadrid",
    "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Atletico",
    "Real Sociedad de Fútbol": "Sociedad",
    "Real Betis Balompié": "Betis",
    "Villarreal CF": "Villarreal",
    "Athletic Club": "Bilbao",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Getafe CF": "Getafe",
    "RC Celta de Vigo": "Celta",
    "CA Osasuna": "Osasuna",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "Rayo Vallecano de Madrid": "RayoVallecano",
    "UD Las Palmas": "LasPalmas",
    "Deportivo Alavés": "Alaves",
    "Cádiz CF": "Cadiz",
    "UD Almería": "Almeria",
    "Granada CF": "Granada",
    "CD Leganés": "Leganes",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Real Valladolid CF": "Valladolid",
    # Serie A
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "SSC Napoli": "Napoli",
    "Juventus FC": "Juventus",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "AC Monza": "Monza",
    "Genoa CFC": "Genoa",
    "US Lecce": "Lecce",
    "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli",
    "Hellas Verona FC": "Verona",
    "Udinese Calcio": "Udinese",
    "US Sassuolo Calcio": "Sassuolo",
    "Frosinone Calcio": "Frosinone",
    "US Salernitana 1919": "Salernitana",
    "Parma Calcio 1913": "Parma",
    "Como 1907": "Como",
    "Venezia FC": "Venezia",
    # Ligue 1
    "Paris Saint-Germain FC": "ParisSG",
    "AS Monaco FC": "Monaco",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Lille OSC": "Lille",
    "OGC Nice": "Nice",
    "Stade Rennais FC 1901": "Rennes",
    "Racing Club de Lens": "Lens",
    "RC Strasbourg Alsace": "Strasbourg",
    "FC Nantes": "Nantes",
    "Montpellier HSC": "Montpellier",
    "Toulouse FC": "Toulouse",
    "Stade Brestois 29": "Brest",
    "Stade de Reims": "Reims",
    "Le Havre AC": "LeHavre",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "Clermont Foot 63": "Clermont",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "AS Saint-Étienne": "Saint-Etienne",
}


@st.cache_data(ttl=86400)
def get_live_elo(team_name: str) -> float:
    """Fetch current ELO rating from api.clubelo.com with robust name mapping."""
    try:
        # Use the verified mapping, fallback to simple space-removal
        clean_name = CLUBELO_NAME_MAP.get(team_name)
        if not clean_name:
            # Fallback: strip common suffixes and spaces
            clean_name = team_name.replace(" ", "").replace("FC", "").replace("CF", "").replace("SC", "")
            log.warning("No ClubELO mapping for '%s', trying fallback: '%s'", team_name, clean_name)

        url = f"http://api.clubelo.com/{clean_name}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or len(resp.text) < 50:
            log.warning("ClubELO API returned no data for '%s' (mapped: '%s')", team_name, clean_name)
            return 1500.0

        df = pd.read_csv(url)
        if not df.empty:
            df = df.dropna(subset=['Elo'])
            df['To'] = pd.to_datetime(df['To'])
            df = df.sort_values('To')
            elo = float(df.iloc[-1]['Elo'])
            log.info("ELO for '%s': %.1f", team_name, elo)
            return elo
    except Exception as e:
        log.warning("Failed to fetch ELO for '%s': %s", team_name, e)
    return 1500.0  # Safe default


@st.cache_data(ttl=3600)
def fetch_recent_matches(team_id: int) -> list[dict]:
    """Fetch 10 recent finished matches for a team from football-data API."""
    if team_id == 0:
        log.warning("team_id is 0 — team not found in football-data mapping")
        return []
    try:
        headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
        url = f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=10"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 429:
            log.warning("Rate limited by football-data.org API (429)")
            return []
        data = resp.json()
        return data.get("matches", [])
    except Exception as e:
        log.warning("Failed to fetch recent matches for team %d: %s", team_id, e)
        return []


@st.cache_data(ttl=3600)
def fetch_league_standing(team_id: int) -> int:
    """Fetch current league position for a team from football-data API."""
    if team_id == 0:
        return 10
    try:
        headers = {"X-Auth-Token": st.secrets["FOOTBALL_DATA_API_KEY"]}
        url = f"https://api.football-data.org/v4/teams/{team_id}"
        resp_team = requests.get(url, headers=headers, timeout=15)
        if resp_team.status_code == 429:
            log.warning("Rate limited fetching team info for %d", team_id)
            return 10
        team_data = resp_team.json()
        active_comps = team_data.get("runningCompetitions", [])
        if active_comps:
            comp_code = active_comps[0]["code"]
            url_standings = f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
            resp_standings = requests.get(url_standings, headers=headers, timeout=15)
            if resp_standings.status_code == 429:
                log.warning("Rate limited fetching standings for comp %s", comp_code)
                return 10
            standings = resp_standings.json().get("standings", [])
            if standings:
                for row in standings[0].get("table", []):
                    if row["team"]["id"] == team_id:
                        pos = row["position"]
                        log.info("League position for team %d: %d", team_id, pos)
                        return pos
    except Exception as e:
        log.warning("Failed to fetch league standing for team %d: %s", team_id, e)

    return 10  # Median position as safe default


def calculate_recent_stats(matches: list[dict], team_id: int) -> tuple[float, float, float, float]:
    """Calculates form, goals_per_game, goals_conceded_per_game, win_rate from recent matches."""
    if not matches:
        return 1.0, 1.0, 1.0, 0.3  # Safe medians

    points = 0
    goals_scored = 0
    goals_conceded = 0
    wins = 0

    for m in matches[-5:]:  # Form based on last 5
        home_team_id = m["homeTeam"]["id"]
        home_score = m["score"]["fullTime"]["home"]
        away_score = m["score"]["fullTime"]["away"]

        if home_score is None or away_score is None:
            continue

        if home_team_id == team_id:
            goals_scored += home_score
            goals_conceded += away_score
            if home_score > away_score:
                points += 3
                wins += 1
            elif home_score == away_score:
                points += 1
        else:
            goals_scored += away_score
            goals_conceded += home_score
            if away_score > home_score:
                points += 3
                wins += 1
            elif away_score == home_score:
                points += 1

    n = len(matches[-5:])
    if n == 0:
        return 1.0, 1.0, 1.0, 0.3

    avg_pts = float(points / n)
    avg_gf = float(goals_scored / n)
    avg_ga = float(goals_conceded / n)
    win_r = float(wins / n)

    return avg_pts, avg_gf, avg_ga, win_r


def calculate_days_since_last_match(matches: list[dict]) -> float:
    """Calculate days since the team's most recent match."""
    if not matches:
        return 7.0  # Default
    try:
        last_match = matches[-1]
        match_date = pd.to_datetime(last_match["utcDate"]).tz_localize(None)
        days = (datetime.now() - match_date).days
        return max(1.0, float(days))
    except Exception:
        return 7.0


def get_h2h_from_csv(home_team: str, away_team: str) -> tuple[int, int, int]:
    """
    Look up historical H2H stats from features_complete.csv.
    Returns (h2h_home_wins, h2h_draws, h2h_away_wins) from the most recent row.
    Falls back to reversed match if direct match not found.
    """
    df = load_feature_matrix()
    if df is None:
        return 0, 0, 0

    # Try direct match (home=home, away=away)
    mask = (
        df["home_team"].str.lower().str.contains(home_team.lower().split()[0], na=False)
        & df["away_team"].str.lower().str.contains(away_team.lower().split()[0], na=False)
    )
    hits = df[mask]
    if not hits.empty:
        row = hits.iloc[-1]
        return (
            int(row.get("h2h_home_wins", 0)),
            int(row.get("h2h_draws", 0)),
            int(row.get("h2h_away_wins", 0)),
        )

    # Try reversed (home↔away) — swap h2h_home_wins and h2h_away_wins
    mask_rev = (
        df["home_team"].str.lower().str.contains(away_team.lower().split()[0], na=False)
        & df["away_team"].str.lower().str.contains(home_team.lower().split()[0], na=False)
    )
    hits_rev = df[mask_rev]
    if not hits_rev.empty:
        row = hits_rev.iloc[-1]
        return (
            int(row.get("h2h_away_wins", 0)),
            int(row.get("h2h_draws", 0)),
            int(row.get("h2h_home_wins", 0)),
        )

    log.info("No H2H data found for %s vs %s", home_team, away_team)
    return 0, 0, 0


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

    # 3. ELO Ratings (with robust name mapping)
    h_elo = get_live_elo(home_team_name)
    a_elo = get_live_elo(away_team_name)

    # 4. Days since last match (calculated from actual match dates)
    h_days = calculate_days_since_last_match(home_matches)
    a_days = calculate_days_since_last_match(away_matches)

    # 5. H2H from historical CSV
    h2h_hw, h2h_d, h2h_aw = get_h2h_from_csv(home_team_name, away_team_name)

    # 6. Synthesize final dictionary — feature names MUST match the trained model
    features = {
        "league_position_home": float(home_pos),
        "league_position_away": float(away_pos),
        "goals_per_game_home": h_gf,
        "goals_per_game_away": a_gf,
        "goals_conceded_per_game_home": h_ga,
        "goals_conceded_per_game_away": a_ga,
        "home_advantage_score": 1.0,
        "days_since_last_match_home": h_days,
        "days_since_last_match_away": a_days,
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
        "h2h_home_wins": float(h2h_hw),
        "h2h_draws": float(h2h_d),
        "h2h_away_wins": float(h2h_aw),
        "strength_ratio": h_elo / a_elo if a_elo > 0 else 1.0,
        "goal_difference_delta": (h_gf - h_ga) - (a_gf - a_ga),
        "rest_days_home": h_days,
        "rest_days_away": a_days,
        # Default NLP features — will be updated by live_news.py if available
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
