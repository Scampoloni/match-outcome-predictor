from rapidfuzz import process, fuzz
import streamlit as st
import google.generativeai as genai

from utils import get_current_teams
from live_features import build_live_features
from live_news import calculate_match_sentiment
from utils import predict_match_outcome, get_team_id_mapping

def find_teams_in_query(query: str) -> tuple[str | None, str | None]:
    """
    Finds up to 2 distinct teams in a natural language query using RapidFuzz.
    """
    all_teams_dict = get_current_teams()
    all_teams = []
    for l_teams in all_teams_dict.values():
         all_teams.extend(l_teams)
    all_teams = list(set(all_teams))
    
    # Pre-clean generic words from tokens to avoid grabbing "Real" for "Real Madrid" when talking about generic topics, although fuzzy handles it well
    query_clean = query.lower()
    
    # We want top 2 matches that are above a good cutoff
    extracted = process.extract(query_clean, all_teams, scorer=fuzz.partial_ratio, limit=5)
    
    # Filter distinct matches that hit at least 70
    valid_teams = []
    for match, score, _ in extracted:
        if score >= 70:
            valid_teams.append(match)
            if len(valid_teams) == 2:
                break
                
    if len(valid_teams) >= 2:
        return valid_teams[0], valid_teams[1]
    elif len(valid_teams) == 1:
        return valid_teams[0], None
    return None, None

def build_gemini_context(home_team: str, away_team: str) -> str:
    """
    Builds the system data context prompt enforcing strict boundaries on what Gemini can talk about.
    """
    mapping = get_team_id_mapping()
    h_id = mapping.get(home_team, 0)
    a_id = mapping.get(away_team, 0)
    
    features = build_live_features(h_id, a_id, home_team, away_team)
    news_nlp = calculate_match_sentiment(home_team, away_team)
    
    # Merge NLP features into the feature vector so XGBoost can use them
    if news_nlp["data_available"]:
        for k, v in news_nlp.items():
            if k in features:
                features[k] = v
                
    # Run Prediction
    # Assuming match_row is a series or dictionary
    import pandas as pd
    match_row = pd.Series(features)
    p_res = predict_match_outcome(match_row, "xgboost", True)

    context = f"[SYSTEM DATA CONTEXT]\nMATCH: {home_team} vs {away_team}\n"
    context += f"- League Position: Home #{int(features.get('league_position_home', 0))} | Away #{int(features.get('league_position_away', 0))}\n"
    context += f"- ELO Rating: Home {features.get('elo_home', 0):.0f} | Away {features.get('elo_away', 0):.0f}\n"
    context += f"- Form (Last 5 Games pts/game): Home {features.get('form_home', 0):.2f} | Away {features.get('form_away', 0):.2f}\n"
    
    if news_nlp["data_available"]:
        context += f"- Media Sentiment Score (-1 to 1): Home {news_nlp.get('sentiment_mean_home', 0):.2f} | Away {news_nlp.get('sentiment_mean_away', 0):.2f}\n"
        context += f"- Injury Concerns Mentioned: Home {news_nlp.get('injury_concern_score_home', 0):.2f} | Away {news_nlp.get('injury_concern_score_away', 0):.2f}\n"
    else:
        context += "- Media Sentiment: No current news found.\n"
        
    context += f"- AI Model Prediction: The model predicts a '{p_res.get('label', 'Unknown')}' with {p_res.get('confidence', 0)*100:.1f}% confidence.\n"
    
    context += "\nCRITICAL INSTRUCTION FOR THE AI: Beantworte NUR Fragen die durch die oben stehenden Daten abgedeckt sind. "
    context += "Nenne KEINE Spielernamen, Transfers oder Verletzungen, die nicht in den News-Daten erwaehnt werden. "
    context += "Wenn du keine Information hast, sage das ehrlich und verweise auf die Statistiken oben. Verhalte dich wie ein professioneller Fussball-Analyst."
    
    return context

def call_gemini_api(system_context: str, user_prompt: str, chat_history: list) -> str:
    """Invokes Gemini with context and message history."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_context
        )
        history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
             
        chat = model.start_chat(history=history)
        response = chat.send_message(user_prompt)
        return response.text
    except Exception as e:
        return f"Entschuldigung, es gab ein Problem bei der Verarbeitung: {str(e)}"
