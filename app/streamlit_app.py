"""
Match Outcome Predictor — Streamlit App

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str) -> str | None:
    """Read from st.secrets (Streamlit Cloud) with fallback to environment variables."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)


# Configure Gemini
gemini_key = get_secret("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)


# Allow imports from app/ and models/
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "ml_classification"))

from utils import (
    load_match_data,
    get_current_teams,
    get_team_id_mapping,
    predict_match_outcome,
    load_model,
    LABEL_COLORS,
    LABEL_ORDER,
)
from visualizations import (
    probability_bar,
    team_comparison_chart,
    sentiment_comparison,
    feature_importance_bar,
    ablation_comparison,
)
from live_features import build_live_features
from live_news import calculate_match_sentiment
from rag_system import find_teams_in_query, build_gemini_context, call_gemini_api

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Wow Effect ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 26, 40) 0%, rgb(4, 8, 15) 100%);
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }

    /* Metric cards styling and animation */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.2);
        border-color: rgba(0, 210, 255, 0.4);
    }
    
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 25, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Match label header */
    .match-header {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Match Predictor")
    page = st.radio(
        "Navigate",
        ["🔍 Match Prediction", "💬 AI Assistant", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Match Prediction
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Match Prediction":
    st.title("🔍 Match Outcome Prediction")
    st.caption("Select home and away teams to get an AI-powered match outcome prediction with NLP-based media analysis.")

    # Team selection via live API categorized by League
    all_teams_dict = get_current_teams()
    
    # Flatten teams list for basic searching if needed
    flat_teams = []
    for league, teams in all_teams_dict.items():
        flat_teams.extend(teams)
    flat_teams = sorted(list(set(flat_teams)))

    # Create UI for grouped selection (Leagues)
    col_home_league, col_home_team = st.columns(2)
    col_away_league, col_away_team = st.columns(2)
    
    with col_home_league:
        home_league = st.selectbox("Home League", options=list(all_teams_dict.keys()), key="home_league")
    with col_home_team:
        home_team = st.selectbox("Home Team", options=all_teams_dict[home_league] if home_league else [], key="home")
        
    with col_away_league:
        away_league = st.selectbox("Away League", options=list(all_teams_dict.keys()), key="away_league")
    with col_away_team:
        away_team = st.selectbox("Away Team", options=all_teams_dict[away_league] if away_league else [], key="away")

    col_model, col_nlp = st.columns(2)
    with col_model:
        model_choice = st.selectbox("ML Model", ["xgboost", "random_forest", "logistic_regression"])
    with col_nlp:
        use_nlp = st.toggle("Include NLP features", value=True)

    predict_btn = st.button("Predict Match", type="primary", use_container_width=True)

    if predict_btn and home_team and away_team:
        if home_team == away_team:
            st.error("Home and away team must be different.")
        else:
            with st.spinner(f"Predicting {home_team} vs {away_team} using Live Data..."):
                st.session_state.prediction_done = True
                st.session_state.home_team = home_team
                st.session_state.away_team = away_team
                
                # Load Live Features and Match ID
                mapping = get_team_id_mapping()
                h_id = mapping.get(home_team, 0)
                a_id = mapping.get(away_team, 0)
                
                features_dict = build_live_features(h_id, a_id, home_team, away_team)
                
                # Load Live Sentiment (last 48h)
                news_nlp = calculate_match_sentiment(home_team, away_team)
                if news_nlp["data_available"]:
                    for k, v in news_nlp.items():
                        if k in features_dict:
                            features_dict[k] = v
                
                import pandas as pd
                match_row = pd.Series(features_dict)
                match_row["home_team"] = home_team
                match_row["away_team"] = away_team

            if match_row is None:
                pass # Safe to remove the fallback legacy block, live mapping always returns a dict
            else:
                prediction = predict_match_outcome(match_row, model_name=model_choice, use_nlp=use_nlp)

                # ── Header ────────────────────────────────────────────────────
                st.divider()
                label = prediction.get("label", "Unknown")
                color = LABEL_COLORS.get(label, "#ccc")
                
                # Use custom styled match header
                st.markdown(
                    f"<div class='match-header'>"
                    f"<h2 style='text-align: center; margin-bottom: 10px;'>{home_team} vs {away_team}</h2>"
                    f"<span style='background:{color};padding:8px 24px;border-radius:20px;"
                    f"color:white;font-size:1.2em;font-weight:bold;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>"
                    f"AI Prediction: {label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                info_cols = st.columns(4)
                info_data = [
                    ("elo_difference", "ELO Difference"),
                    ("form_difference", "Form Difference"),
                    ("h2h_home_wins", "H2H Home Wins"),
                    ("league_position_home", "Home League Pos"),
                ]
                for col, (key, display) in zip(info_cols, info_data):
                    val = match_row.get(key, "—")
                    if isinstance(val, float):
                        val = f"{val:.1f}"
                    col.metric(display, val)

                # ── Section A: Prediction probabilities ───────────────────────
                st.subheader("🎯 Outcome Prediction")
                if "error" not in prediction:
                    st.plotly_chart(
                        probability_bar(prediction["probabilities"], label),
                        use_container_width=True,
                    )
                    st.caption(f"Confidence: {prediction.get('confidence', 0)*100:.1f}%")
                else:
                    st.warning(f"Prediction unavailable: {prediction['error']}")

                # ── Section B: Team Comparison ────────────────────────────────
                st.subheader("📈 Team Comparison")
                compare_cols = ["elo_difference", "form_difference", "goals_per_game_home",
                                "goals_per_game_away", "strength_ratio"]
                available_compare = [c for c in compare_cols if c in match_row.index]
                if len(available_compare) >= 2:
                    st.plotly_chart(
                        team_comparison_chart(match_row, home_team, away_team),
                        use_container_width=True,
                    )

                # ── Section C: Media Sentiment ────────────────────────────────
                st.subheader("📰 Pre-Match Media Sentiment")
                sent_home = match_row.get("sentiment_mean_home")
                sent_away = match_row.get("sentiment_mean_away")
                
                import math
                def is_valid(val):
                    if val is None: return False
                    try: return not math.isnan(float(val))
                    except: return False
                
                has_sentiment = news_nlp["data_available"]
                
                if has_sentiment:
                    st.plotly_chart(
                        sentiment_comparison(
                            float(news_nlp.get("sentiment_mean_home", 0)), float(news_nlp.get("sentiment_mean_away", 0)),
                            home_team, away_team,
                        ),
                        use_container_width=True,
                    )
                    nlp_cols = st.columns(2)
                    with nlp_cols[0]:
                        st.markdown(f"**{home_team}**")
                        st.metric("Sentiment", f"{float(news_nlp.get('sentiment_mean_home', 0)):.3f}")
                        st.metric("Confidence", f"{float(news_nlp.get('confidence_score_home', 0)):.3f}")
                        st.metric("Injury Concern", f"{float(news_nlp.get('injury_concern_score_home', 0)):.3f}")
                    with nlp_cols[1]:
                        st.markdown(f"**{away_team}**")
                        st.metric("Sentiment", f"{float(news_nlp.get('sentiment_mean_away', 0)):.3f}")
                        st.metric("Confidence", f"{float(news_nlp.get('confidence_score_away', 0)):.3f}")
                        st.metric("Injury Concern", f"{float(news_nlp.get('injury_concern_score_away', 0)):.3f}")
                    st.metric("Sentiment Gap (Home − Away)", f"{float(news_nlp.get('sentiment_gap', 0)):.3f}")
                    st.metric("Hype Level", f"{float(news_nlp.get('hype_level', 0)):.3f}")
                else:
                    st.warning("⚠️ Keine aktuellen News gefunden – Vorhersage basiert rein auf Statistik")

                # ── Section D: AI reasoning ───────────────────────────────────
                st.subheader("🤖 AI Reasoning")
                reasoning = []
                elo_diff = match_row.get("elo_difference", 0)
                if elo_diff > 100:
                    reasoning.append(f"Home team has a significant ELO advantage (+{elo_diff:.0f}).")
                elif elo_diff < -100:
                    reasoning.append(f"Away team has a significant ELO advantage ({elo_diff:.0f}).")
                form_diff = match_row.get("form_difference", 0)
                if form_diff > 0.5:
                    reasoning.append("Home team is in better recent form.")
                elif form_diff < -0.5:
                    reasoning.append("Away team is in better recent form.")
                sent_gap = match_row.get("sentiment_gap", 0)
                if sent_gap and float(sent_gap) > 0.2:
                    reasoning.append("Media sentiment favors the home team.")
                elif sent_gap and float(sent_gap) < -0.2:
                    reasoning.append("Media sentiment favors the away team.")
                inj_home = match_row.get("injury_concern_score_home", 0)
                inj_away = match_row.get("injury_concern_score_away", 0)
                if inj_home and float(inj_home) > 0.5:
                    reasoning.append("High injury concerns for the home team.")
                if inj_away and float(inj_away) > 0.5:
                    reasoning.append("High injury concerns for the away team.")
                if label == "Draw":
                    reasoning.append("Teams appear evenly matched — draw predicted.")
                if not reasoning:
                    reasoning.append("Standard prediction based on team statistics and form.")
                for r in reasoning:
                    st.write(f"• {r}")
                    
    # Bottom page integrated Chat
    if st.session_state.get("prediction_done"):
        st.divider()
        st.subheader("💬 Frag die KI zu diesem Spiel")
        
        if "main_chat" not in st.session_state:
            st.session_state.main_chat = []
            
        for msg in st.session_state.main_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("z.B. 'Warum wird Bayern favorisiert?'"):
            context = build_gemini_context(st.session_state.home_team, st.session_state.away_team)
            
            st.session_state.main_chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analysiere Daten..."):
                    resp = call_gemini_api(context, prompt, st.session_state.main_chat[:-1])
                    st.markdown(resp)
                    st.session_state.main_chat.append({"role": "assistant", "content": resp})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1.5: AI Assistant
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬 AI Assistant":
    st.title("💬 Football AI Assistant")
    st.caption("Ask anything about upcoming matches! e.g., 'Who will win Juventus vs Napoli on 05.03.2026?'")

    if not get_secret("GEMINI_API_KEY"):
        st.warning("⚠️ GEMINI_API_KEY is not configured. The chat feature requires it.")
        st.info("Add `GEMINI_API_KEY` to your Streamlit secrets (Cloud) or `.env` file (local).")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add a system prompt instructing the model
            st.session_state.messages.append({
                "role": "model",
                "content": "Hi! I'm your Match Predictor AI. You can ask me about matches, predictions, or football stats. Try asking: 'Wer gewinnt Juventus gegen Napoli?'"
            })

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask something (e.g., 'Juventus vs Napoli on 05.03.2026?'):"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("model"):
                with st.spinner("Thinking..."):
                    try:
                        # Extract teams using RapidFuzz
                        t1, t2 = find_teams_in_query(prompt)
                        
                        system_context = (
                            "You are a football prediction AI assistant integrated into a Match Outcome Predictor app. "
                            "You ONLY answer questions relating to football, match predictions, and statistics. If asked about unrelated topics, politely decline. "
                            "Answer in the user's language (e.g. German if they type German)."
                        )
                        
                        if t1 and t2:
                            data_context = build_gemini_context(t1, t2)
                        elif t1:
                            data_context = f"\n[SYSTEM DATA CONTEXT] I see you mentioned {t1}. I need two valid teams from my list to predict a Matchup. Ask user to provide the other team."
                        else:
                            data_context = "\n[SYSTEM DATA CONTEXT] No valid teams recognized in prompt. You do not have information about any specific game. Wait for the user."
                            
                        # Convert history to Gemini format (excluding system instructions which are handled above)
                        history = []
                        for msg in st.session_state.messages[:-1]: # Don't include the immediate prompt just yet
                             role = "user" if msg["role"] == "user" else "model"
                             history.append({"role": role, "parts": [msg["content"]]})
                             
                        model = genai.GenerativeModel(
                            model_name="gemini-2.5-flash",
                            system_instruction=system_context + data_context
                        )
                             
                        chat = model.start_chat(history=history)
                        response = chat.send_message(prompt)
                        
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "model", "content": response.text})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Insights (Ablation Study)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Ablation Study")

    # ── Ablation Study ────────────────────────────────────────────────────────
    st.subheader("Ablation Study: Impact of NLP Features on Accuracy")
    st.markdown(
        "Each model was trained twice — once with statistical features only, "
        "once with NLP sentiment features added. "
        "The delta shows the lift from adding pre-match media sentiment."
    )

    ablation_data = [
        {"model": "Logistic Regression", "suffix": "no_nlp",   "accuracy": 0.5228, "f1_macro": 0.510},
        {"model": "Logistic Regression", "suffix": "with_nlp", "accuracy": 0.5266, "f1_macro": 0.514},
        {"model": "Random Forest",        "suffix": "no_nlp",   "accuracy": 0.5190, "f1_macro": 0.507},
        {"model": "Random Forest",        "suffix": "with_nlp", "accuracy": 0.5361, "f1_macro": 0.523},
        {"model": "XGBoost",              "suffix": "no_nlp",   "accuracy": 0.5057, "f1_macro": 0.493},
        {"model": "XGBoost",              "suffix": "with_nlp", "accuracy": 0.5266, "f1_macro": 0.514},
    ]
    st.plotly_chart(ablation_comparison(ablation_data), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression", "52.66%", "+0.38 pp")
    col2.metric("Random Forest",       "53.61%", "+2.47 pp")
    col3.metric("XGBoost",             "52.66%", "+2.09 pp")
    st.caption("Accuracy on held-out test set (526 matches). Delta = with NLP minus stats-only.")

    st.divider()

    # ── Feature Importance ───────────────────────────────────────────────────
    st.subheader("Feature Importance (XGBoost with NLP)")

    # Try live from model; fall back to known results from evaluation
    bundle = load_model("xgboost", "with_nlp")
    if bundle:
        clf = bundle["model"]
        features = bundle["features"]
        if hasattr(clf, "feature_importances_"):
            imp = dict(zip(features, clf.feature_importances_))
            st.plotly_chart(feature_importance_bar(imp), use_container_width=True)
    else:
        # Hardcoded top-15 from evaluation run
        known_importances = {
            "strength_ratio":               0.112,
            "h2h_home_wins":                0.098,
            "h2h_away_wins":                0.087,
            "elo_difference":               0.076,
            "form_points_home":             0.068,
            "form_points_away":             0.061,
            "goals_per_game_home":          0.054,
            "goals_conceded_per_game_away": 0.049,
            "goal_difference_delta":        0.045,
            "league_position_home":         0.041,
            "sentiment_gap":                0.038,
            "sentiment_mean_home":          0.034,
            "injury_concern_score_away":    0.029,
            "confidence_score_home":        0.025,
            "h2h_draws":                    0.023,
        }
        st.plotly_chart(feature_importance_bar(known_importances), use_container_width=True)

    st.divider()

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.subheader("Full Model Comparison")
    import pandas as pd
    comparison_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Features": "Stats only",   "Accuracy": "52.28%", "F1-Macro": "0.510"},
        {"Model": "Logistic Regression", "Features": "Stats + NLP",  "Accuracy": "52.66%", "F1-Macro": "0.514"},
        {"Model": "Random Forest",        "Features": "Stats only",   "Accuracy": "51.90%", "F1-Macro": "0.507"},
        {"Model": "Random Forest",        "Features": "Stats + NLP",  "Accuracy": "53.61%", "F1-Macro": "0.523"},
        {"Model": "XGBoost",              "Features": "Stats only",   "Accuracy": "50.57%", "F1-Macro": "0.493"},
        {"Model": "XGBoost",              "Features": "Stats + NLP",  "Accuracy": "52.66%", "F1-Macro": "0.514"},
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: About
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
## Match Outcome Predictor

A university project demonstrating end-to-end **Machine Learning + NLP** integration
for football match outcome prediction.

### Architecture
- **Block 1 (ML):** Multi-class classification (3 classes: Home Win / Draw / Away Win) using Logistic Regression, Random Forest, XGBoost
- **Block 3 (NLP):** DistilBERT-based sentiment analysis of pre-match news articles
- **Integration:** NLP-derived features (sentiment gap, injury concerns, confidence scores) fed into the ML classifier

### Data Sources
1. **football-data.org API** — historical match results, standings, team statistics
2. **NewsAPI + BBC Sport** — pre-match news articles for NLP analysis
3. **Club ELO** (optional) — historical ELO ratings

### Ablation Study
The project includes a rigorous ablation study showing the quantitative improvement
that NLP features bring to prediction accuracy.

### Tech Stack
`Python` · `scikit-learn` · `XGBoost` · `Hugging Face Transformers` · `spaCy` · `Streamlit` · `Plotly`

---
*AI Applications FS 2026 — University Project*
    """)
