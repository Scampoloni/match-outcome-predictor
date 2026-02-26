"""
Match Outcome Predictor — Streamlit App

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Allow imports from app/ and models/
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "ml_classification"))

from utils import (
    load_match_data,
    get_all_teams,
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ Match Predictor")
    page = st.radio(
        "Navigate",
        ["🔍 Match Prediction", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Match Prediction
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Match Prediction":
    st.title("🔍 Match Outcome Prediction")
    st.caption("Select home and away teams to get an AI-powered match outcome prediction with NLP-based media analysis.")

    # Team selection
    all_teams = get_all_teams()
    col_home, col_away = st.columns(2)
    with col_home:
        if all_teams:
            home_team = st.selectbox("Home Team", options=[""] + all_teams, key="home",
                                     format_func=lambda x: "Select home team..." if x == "" else x)
        else:
            home_team = st.text_input("Home Team", placeholder="e.g. Arsenal FC")
    with col_away:
        if all_teams:
            away_team = st.selectbox("Away Team", options=[""] + all_teams, key="away",
                                     format_func=lambda x: "Select away team..." if x == "" else x)
        else:
            away_team = st.text_input("Away Team", placeholder="e.g. Chelsea FC")

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
            with st.spinner(f"Predicting {home_team} vs {away_team}..."):
                match_row = load_match_data(home_team, away_team)

            if match_row is None:
                st.error("Match data not found. Run the data pipeline first.")
                st.info("Run: `python data/scrapers/collect_matches.py` then `python models/nlp_analysis/feature_extractor.py`")
            else:
                prediction = predict_match_outcome(match_row, model_name=model_choice, use_nlp=use_nlp)

                # ── Header ────────────────────────────────────────────────────
                st.divider()
                label = prediction.get("label", "Unknown")
                color = LABEL_COLORS.get(label, "#ccc")
                st.markdown(
                    f"## {home_team} vs {away_team} &nbsp; "
                    f"<span style='background:{color};padding:4px 12px;border-radius:12px;"
                    f"color:white;font-size:0.85em'>{label}</span>",
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
                if sent_home is not None and str(sent_home) != "nan":
                    st.plotly_chart(
                        sentiment_comparison(
                            float(sent_home), float(sent_away),
                            home_team, away_team,
                        ),
                        use_container_width=True,
                    )
                    nlp_cols = st.columns(2)
                    with nlp_cols[0]:
                        st.markdown(f"**{home_team}**")
                        st.metric("Sentiment", f"{match_row.get('sentiment_mean_home', 0):.3f}")
                        st.metric("Confidence", f"{match_row.get('confidence_score_home', 0):.3f}")
                        st.metric("Injury Concern", f"{match_row.get('injury_concern_score_home', 0):.3f}")
                    with nlp_cols[1]:
                        st.markdown(f"**{away_team}**")
                        st.metric("Sentiment", f"{match_row.get('sentiment_mean_away', 0):.3f}")
                        st.metric("Confidence", f"{match_row.get('confidence_score_away', 0):.3f}")
                        st.metric("Injury Concern", f"{match_row.get('injury_concern_score_away', 0):.3f}")
                    st.metric("Sentiment Gap (Home − Away)", f"{match_row.get('sentiment_gap', 0):.3f}")
                    st.metric("Hype Level", f"{match_row.get('hype_level', 0):.3f}")
                else:
                    st.info("No NLP features available for this match. Run the sentiment analyzer first.")

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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Insights (Ablation Study)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Ablation Study")
    st.info(
        "This page shows how NLP features contribute to model performance.\n"
        "Run `python models/ml_classification/evaluate.py` and "
        "`python models/ml_classification/model_comparison.py` first."
    )

    # Try to load ablation chart image
    chart_path = Path(__file__).resolve().parents[1] / "models" / "ml_classification" / "saved_models" / "ablation_comparison.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Ablation Study: NLP Feature Impact", use_container_width=True)
    else:
        st.warning("Ablation chart not generated yet.")

    # Feature importance
    st.subheader("Feature Importance (XGBoost)")
    bundle = load_model("xgboost", "with_nlp")
    if bundle:
        clf = bundle["model"]
        features = bundle["features"]
        if hasattr(clf, "feature_importances_"):
            imp = dict(zip(features, clf.feature_importances_))
            st.plotly_chart(feature_importance_bar(imp), use_container_width=True)
    else:
        st.warning("XGBoost model not loaded. Run `train.py` first.")

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
