# INSTRUCTIONS.md - Player Performance Predictor

**⚠️ CRITICAL: READ THIS ENTIRE FILE BEFORE MAKING ANY CHANGES TO THE PROJECT**

This document contains all essential information about this project. Any AI assistant (Claude, Copilot, ChatGPT) must read and understand this file completely before writing code, creating files, or making suggestions.

---

## 📌 PROJECT IDENTITY

**Project Name:** Match Outcome Predictor  
**Repository:** `match-outcome-predictor`  
**Type:** University Project - AI Applications Module (FS 2026)  
**Deadline:** June 7, 2026, 18:00 CET  
**Time Budget:** 90 hours total  
**Grade Target:** 6.0 (Swiss grading: 6 = best, 4 = pass)

---

## 🎯 PROJECT MISSION

Build a machine learning system that predicts football match outcomes (Win/Draw/Loss) by combining:
1. **Team statistical data** (ELO rating, recent form, head-to-head)
2. **Pre-match news sentiment** (NLP analysis of sports articles)
3. **[Optional] Formation analysis** (CV from lineup graphics - only if time permits)

**Key Success Metric:** Demonstrate that NLP features measurably improve match prediction accuracy beyond pure statistics.

---

## 📋 ACADEMIC REQUIREMENTS (NON-NEGOTIABLE)

### Module Structure:
This project MUST satisfy requirements from the "AI Applications" module at a Swiss university:

**Block 1: End-to-End Machine Learning (MANDATORY)**
- Multi-class classification problem
- Feature engineering from multiple sources
- Comparison of ≥3 ML algorithms
- Professional evaluation metrics (accuracy, F1, confusion matrix)
- Reference complexity: Kaggle Titanic level

**Block 3: Natural Language Processing (MANDATORY)**
- Transformer-based models (must use Hugging Face)
- Sentiment analysis on real-world text data
- Feature extraction that feeds into ML model
- Demonstrate quantifiable impact on ML performance

**Block 2: Computer Vision (OPTIONAL BONUS)**
- Only implement if time permits AFTER Blocks 1 & 3 are excellent
- Must add genuine value, not forced integration
- If skipped: project can still achieve grade 6.0

### Integration Requirements:
- ✅ Minimum 2 blocks deeply integrated (ML + NLP)
- ✅ Multiple different data sources (minimum 3)
- ✅ Public deployment with accessible URL
- ✅ GitHub repository with clean structure
- ✅ Ablation study proving NLP improves ML

### Grading Weight Distribution:
- Technical implementation: 40%
- Integration quality: 30%
- Deployment & usability: 15%
- Documentation: 15%

---

## 🏗️ TECHNICAL ARCHITECTURE

### Problem Definition:

**Classification Task:**
```
Input: Two teams + match date
Output: Match outcome prediction with probabilities

Classes (3-class classification):
  1. "Home Win"  - Home team wins
  2. "Draw"      - Match ends in a draw
  3. "Away Win"  - Away team wins

Labeling Strategy:
  - Use actual historical match results as ground truth
  - Minimum 2000+ matches from last 3 seasons
  - Top 5 European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)
```

### Feature Architecture:

**Feature Group 1: Team Statistics (ML Block 1)**
```python
Team Performance Metrics (from football-data.org API):

For Home Team:
  - elo_rating_home: ELO rating (1000-2500)
  - goals_per_game_home: Average goals scored per match
  - goals_conceded_per_game_home: Average goals conceded
  - win_rate_home: Percentage of wins this season
  - form_points_home: Points from last 5 matches (0-15)
  - home_advantage_score: Historical home performance boost
  - league_position_home: Current standing (1-20)
  
For Away Team:
  - elo_rating_away: ELO rating
  - goals_per_game_away: Average goals scored
  - goals_conceded_per_game_away: Average conceded
  - win_rate_away: Win percentage
  - form_points_away: Last 5 matches points
  - away_performance_score: Historical away record
  - league_position_away: Current standing

Match Context:
  - elo_difference: home_elo - away_elo
  - form_difference: home_form - away_form
  - h2h_home_wins: Historical head-to-head home wins
  - h2h_away_wins: Historical head-to-head away wins
  - h2h_draws: Historical draws
  - h2h_goals_avg: Average goals in H2H matches
  - days_since_last_match_home: Rest days for home team
  - days_since_last_match_away: Rest days for away team
  
Derived Features:
  - strength_ratio: elo_home / elo_away
  - form_momentum_home: Trend of last 10 matches
  - form_momentum_away: Trend of last 10 matches
  - goal_difference_delta: (home_gf - home_ga) - (away_gf - away_ga)
```

**Feature Group 2: NLP-Derived (NLP Block 3)**
```python
Pre-Match Media Sentiment (from news articles):

For Home Team:
  - sentiment_mean_home: Average sentiment (-1 to +1)
  - sentiment_std_home: Sentiment volatility
  - confidence_score_home: Media confidence level (0-1)
  - injury_concern_score_home: Injury worry level (0-1)
  - tactical_change_mentioned_home: Boolean (new formation/system)
  - morale_keywords_count_home: "confident", "motivated" mentions
  - pressure_keywords_count_home: "must-win", "crisis" mentions
  
For Away Team:
  - sentiment_mean_away: Average sentiment
  - sentiment_std_away: Sentiment volatility  
  - confidence_score_away: Media confidence
  - injury_concern_score_away: Injury concerns
  - tactical_change_mentioned_away: Boolean
  - morale_keywords_count_away: Positive morale indicators
  - pressure_keywords_count_away: Pressure indicators

Match-Level NLP:
  - sentiment_gap: sentiment_home - sentiment_away
  - hype_level: Total media attention (article count)
  - rivalry_intensity: Historical rivalry mentions
  - referee_controversy: Past referee issues mentioned
  
Text Sources:
  - ESPN, BBC Sport, The Athletic, Sky Sports, Goal.com
  - Minimum 500+ articles covering 100+ matches
  - Time range: Articles published 7 days before each match
  - Language: English only
```

**Feature Group 3: CV-Derived (OPTIONAL - Skip if time limited)**
```python
Formation Analysis (from lineup graphics):

ONLY IMPLEMENT IF:
  ✓ Blocks 1 & 3 are complete and excellent
  ✓ Time budget allows (>20 hours remaining)
  ✓ Clear improvement shown in preliminary tests

If implemented:
  - formation_aggression_home: Offensive formation score (0-1)
  - formation_aggression_away: Offensive formation score (0-1)
  - formation_balance_home: Tactical balance (0-1)
  - formation_balance_away: Tactical balance (0-1)

Alternative (Stats-Based):
  - Calculate formation style from player positions in API data
  - No image processing needed
```

---

## 📊 DATA REQUIREMENTS

### Data Sources (Prioritized):

**Source 1: football-data.org API (PRIMARY - MANDATORY)**
- **Description:** Free API providing comprehensive football data
- **URL:** https://api.football-data.org/v4/
- **Authentication:** Free API key (register at football-data.org)
- **Rate Limit:** 10 calls per minute (sufficient for project)
- **Coverage:** 10+ European leagues, current + historical seasons

**Available Endpoints:**
```python
# Match Results & Fixtures
GET /v4/competitions/{leagueCode}/matches
Parameters:
  - season: 2022, 2023, 2024, 2025
  - status: SCHEDULED, FINISHED, IN_PLAY
  - dateFrom, dateTo: Filter by date range

Response includes:
  - homeTeam, awayTeam (name, ID, crest)
  - score (fullTime, halfTime)
  - utcDate
  - matchday, stage
  - referees

# Team Standings
GET /v4/competitions/{leagueCode}/standings
Response:
  - position, playedGames, won, draw, lost
  - points, goalsFor, goalsAgainst, goalDifference

# Team Details & Matches
GET /v4/teams/{teamId}/matches
Filter matches by:
  - status (FINISHED for historical)
  - venue (HOME, AWAY)
  - limit (max results)

# Head-to-Head
GET /v4/matches/{matchId}/head2head
Parameters:
  - limit: Number of past H2H matches
```

**What You'll Collect:**
- ✅ 2000+ historical matches (last 3 seasons)
- ✅ Team standings and statistics
- ✅ Head-to-head records
- ✅ Match results (ground truth labels)
- ✅ Team lineups (basic formation info)

**Leagues to Use (Priority Order):**
1. Premier League (PL) - Highest quality
2. La Liga (PD)
3. Bundesliga (BL1)
4. Serie A (SA)
5. Ligue 1 (FL1)

**Source 2: Sports News APIs & Scraping (NLP PRIMARY - MANDATORY)**

**Option A: NewsAPI.org (Easiest)**
- **URL:** https://newsapi.org/
- **Free Tier:** 100 requests/day, 1 month archive
- **Use For:** Recent match previews and team news
```python
from newsapi import NewsAPIClient

newsapi = NewsAPIClient(api_key='YOUR_KEY')

# Get pre-match articles
articles = newsapi.get_everything(
    q='Manchester United Liverpool preview',
    language='en',
    sort_by='publishedAt',
    from_param='2026-03-08',  # 7 days before match
    to='2026-03-15',          # Match date
    page_size=10
)
```

**Option B: Web Scraping (More Data)**
- **Sources:** ESPN, BBC Sport, The Athletic, Sky Sports
- **Tools:** BeautifulSoup4, Scrapy
- **Rate Limiting:** 1 request per 2 seconds minimum
- **Storage:** Save as JSON with metadata

**Target Collection:**
- 500+ articles minimum
- Coverage: 100+ matches
- Article types: Match previews, team news, injury reports
- Time window: Published 1-7 days before match

**Source 3: ELO Ratings (OPTIONAL ENHANCEMENT)**

**Option A: Calculate Yourself**
```python
# Implement ELO rating system
# Starting ELO: 1500
# K-factor: 20-40 depending on league
# Update after each match result
```

**Option B: Use Club ELO**
- **URL:** http://clubelo.com/
- **Data:** Pre-calculated ELO ratings for all teams
- **Format:** CSV download available
- **Update Frequency:** Daily

**Source 4: TransferMarkt (OPTIONAL - Metadata)**
- **Use:** Team market values, squad depth
- **Library:** `pip install transfermarkt-api`
- **Not critical:** Skip if time limited

### Data Quality Standards:
```python
Minimum Requirements:
  - Complete dataset: ≥2000 matches
  - Feature completeness: <15% missing values per feature
  - NLP coverage: ≥60% of matches have ≥3 articles
  - Label balance: No class <15% of dataset (use oversampling if needed)
  - Temporal split: Train on seasons 2022-2024, validate on early 2025, test on late 2025/2026
  - No data leakage: Test matches must be chronologically AFTER training matches

Data Validation Checks:
  ✓ No duplicate match entries
  ✓ Scores are valid (non-negative integers)
  ✓ Dates are properly formatted and in range
  ✓ Team IDs consistent across datasets
  ✓ Text data properly encoded (UTF-8)
  ✓ Sentiment scores in valid range [-1, 1]
  ✓ ELO ratings in realistic range [1000-2500]
```

---

## 🔧 IMPLEMENTATION STANDARDS

### Code Quality Requirements:

**Python Style:**
```python
# MANDATORY:
- Follow PEP 8 style guide
- Type hints for all functions
- Docstrings (Google style) for all public methods
- Maximum line length: 100 characters
- Use descriptive variable names (no single letters except i, j, k in loops)

# Example:
def calculate_sentiment_score(
    article_text: str,
    model: transformers.Pipeline
) -> Dict[str, float]:
    """
    Calculate sentiment score for a news article.
    
    Args:
        article_text: Raw article text (already cleaned)
        model: Hugging Face sentiment analysis pipeline
        
    Returns:
        Dictionary with 'score' (float -1 to 1) and 'confidence' (0 to 1)
    """
    # Implementation here
```

**Error Handling:**
```python
# ALWAYS handle errors gracefully:
try:
    prediction = model.predict(features)
except ValueError as e:
    logger.error(f"Invalid features: {e}")
    return {"error": "Invalid input data", "details": str(e)}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"error": "Prediction failed", "details": str(e)}
```

**Logging:**
```python
# Use Python logging (not print statements):
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting data collection...")
logger.warning("Missing sentiment data for 5 players")
logger.error("Failed to load model checkpoint")
```

### File Organization Standards:

**Naming Conventions:**
```
Python files: lowercase_with_underscores.py
Notebooks: 01_descriptive_name.ipynb (numbered for sequence)
Data files: descriptive_name_YYYY-MM-DD.csv
Models: model_name_v1.pkl (versioned)
Constants: UPPERCASE_WITH_UNDERSCORES
Classes: CapitalizedWords
Functions/variables: lowercase_with_underscores
```

**Import Organization:**
```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Local application imports
from models.ml_classification import train
from utils.data_processing import clean_data
```

---

## 🧪 ML PIPELINE SPECIFICATIONS

### Model Development Process:

**Phase 1: Baseline Model (Week 1-2)**
```python
Step 1: Simple Logistic Regression
  - Use only top 10 team statistical features
  - No hyperparameter tuning
  - Establishes minimum performance threshold
  - Expected accuracy: 45-50% (3-class is harder than binary)

Step 2: Document baseline metrics
  - Confusion matrix
  - Per-class precision/recall
  - Feature importance (coefficients)
```

**Phase 2: Advanced Models (Week 3-4)**
```python
Model 2: Random Forest
  - n_estimators: 100-500
  - max_depth: 10-30
  - Use all statistical features
  - Expected accuracy: 50-55%

Model 3: XGBoost/LightGBM
  - learning_rate: 0.01-0.1
  - n_estimators: 100-1000
  - max_depth: 5-15
  - Use all features including NLP
  - Expected accuracy: 55-60%

Model 4: Neural Network (optional)
  - Only if time permits
  - Simple feedforward (2-3 layers)
  - Use as comparison point
```

**Phase 3: Integration with NLP (Week 4-5)**
```python
Critical Ablation Study:
  1. Train model WITHOUT NLP features → Record accuracy
  2. Train model WITH NLP features → Record accuracy
  3. Calculate improvement: (with_nlp - without_nlp) / without_nlp
  4. Statistical significance test (t-test or bootstrap)
  
Target: ≥3% accuracy improvement from NLP features
Minimum: ≥2% improvement for passing grade
```

### Hyperparameter Tuning:
```python
# Use GridSearchCV or RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=20,  # Don't overdo it (time constraint)
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Document best parameters in evaluation_results.md
```

### Evaluation Metrics:

**Required Metrics (ALL must be reported):**
```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

Required Outputs:
  1. Overall accuracy (test set)
  2. Per-class precision, recall, F1-score
  3. Confusion matrix (visualized)
  4. ROC curves (if applicable)
  5. Feature importance ranking (top 20)
  6. Cross-validation scores (5-fold)
  
Bonus Outputs:
  - Learning curves (training vs validation)
  - Precision-Recall curves
  - Calibration plots
```

---

## 📰 NLP PIPELINE SPECIFICATIONS

### Sentiment Analysis Implementation:

**Model Selection:**
```python
Primary Model (MUST USE):
  Model: distilbert-base-uncased-finetuned-sst-2-english
  Source: Hugging Face Transformers
  Reason: Fast, accurate, well-documented
  
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0  # Use GPU if available
)

Alternative (if primary doesn't work):
  Model: cardiffnlp/twitter-roberta-base-sentiment
  Better for informal text (social media)
```

**Text Preprocessing Pipeline:**
```python
def preprocess_article(raw_text: str) -> str:
    """
    Clean and prepare article text for NLP analysis.
    
    Steps:
        1. Remove HTML tags
        2. Fix encoding issues
        3. Remove URLs
        4. Normalize whitespace
        5. Keep punctuation (important for sentiment)
        6. Lowercase only for specific tasks
    
    DO NOT:
        - Remove stopwords (context matters)
        - Aggressive stemming (loses meaning)
        - Remove all punctuation
    """
    # Implementation
    text = remove_html_tags(raw_text)
    text = fix_encoding(text)
    text = remove_urls(text)
    text = normalize_whitespace(text)
    return text
```

**Feature Extraction Process:**
```python
def extract_nlp_features(home_team: str, away_team: str, articles: List[str]) -> Dict:
    """
    Extract comprehensive NLP features for a match.
    
    Returns:
        {
            'sentiment_mean_home': float,     # -1 to +1
            'sentiment_mean_away': float,     # -1 to +1
            'sentiment_std_home': float,      # Volatility
            'sentiment_std_away': float,      # Volatility
            'confidence_score_home': float,   # Media confidence (0-1)
            'confidence_score_away': float,   # Media confidence (0-1)
            'injury_concern_score_home': float,  # Injury concern (0-1)
            'injury_concern_score_away': float,  # Injury concern (0-1)
            'sentiment_gap': float,           # home - away sentiment
            'hype_level': float,              # Total media attention
            'article_count': int,             # Coverage frequency
        }
    """
    # Process each article
    home_sentiments = []
    away_sentiments = []
    for article in articles:
        result = sentiment_analyzer(article[:512])  # Truncate to model max
        score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
        
        # Attribute to team based on mention frequency
        content_lower = article.lower()
        if content_lower.count(home_team.lower()) > content_lower.count(away_team.lower()):
            home_sentiments.append(score)
        else:
            away_sentiments.append(score)
    
    # Aggregate
    features = {
        'sentiment_mean_home': np.mean(home_sentiments) if home_sentiments else 0.0,
        'sentiment_mean_away': np.mean(away_sentiments) if away_sentiments else 0.0,
        'sentiment_gap': np.mean(home_sentiments or [0]) - np.mean(away_sentiments or [0]),
        # ... extract other features
    }
    
    return features
```

**Entity Extraction (Advanced):**
```python
# Use spaCy for entity recognition
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_injury_concerns(text: str, team_name: str) -> float:
    """
    Extract injury-related information from text for a specific team.
    
    Keywords: injury, injured, sidelined, doubt, surgery, recovery, ruled out
    
    Returns:
        Score from 0 (no concerns) to 1 (major concerns)
    """
    doc = nlp(text)
    
    injury_keywords = ['injury', 'injured', 'sidelined', 'doubt', 'surgery', 'ruled out']
    severity_modifiers = ['major', 'serious', 'long-term', 'season-ending']
    
    # Count and weight mentions near team name
    # Implementation
```

---

## 🎨 DEPLOYMENT SPECIFICATIONS

### Streamlit Application Structure:

**File: app/streamlit_app.py**
```python
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List

# CONFIGURATION
st.set_page_config(
    page_title="⚽ Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MANDATORY SECTIONS:

def main():
    """
    Main application entry point.
    
    Structure:
        1. Header & Introduction
        2. Sidebar Navigation
        3. Main Content Area (dynamic based on selection)
        4. Footer with credits
    """
    
    # Sidebar
    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Predict Match", "📊 Model Insights", "ℹ️ About"]
        )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Predict Match":
        show_prediction_page()
    elif page == "📊 Model Insights":
        show_model_insights()
    elif page == "ℹ️ About":
        show_about_page()

# REQUIRED PAGES:

def show_prediction_page():
    """
    Main prediction interface.
    
    Components:
        - Home team / Away team selection
        - Match date input
        - Optional: News URL input for custom NLP
        - Prediction button
        - Results display (comprehensive report)
    """
    st.title("🔍 Match Prediction")
    
    # Team inputs
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input(
            "Home Team",
            placeholder="e.g., Juventus"
        )
    with col2:
        away_team = st.text_input(
            "Away Team",
            placeholder="e.g., Roma"
        )
    
    match_date = st.date_input("Match Date")
    
    # Optional enhancements
    with st.expander("📰 Advanced: Add Custom News Articles"):
        news_url = st.text_area("Paste article URLs (one per line)")
    
    if st.button("⚡ Predict Match Outcome", type="primary"):
        with st.spinner("Running AI analysis..."):
            display_match_report(home_team, away_team, str(match_date), news_url)

def display_match_report(home_team: str, away_team: str, match_date: str, custom_news: str = None):
    """
    Display comprehensive match prediction report.
    
    Sections (in order):
        1. Match Overview (teams, date)
        2. Outcome Prediction (probabilities with confidence)
        3. Team Comparison (stats side-by-side)
        4. Media Sentiment Analysis (NLP results per team)
        5. Feature Importance (what drove this prediction)
        6. AI Reasoning (natural language explanation)
    """
    # Implementation
    
    # Section 1: Overview
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### {home_team}")
        st.metric("ELO Rating", "1,847")
        st.metric("League Position", "2nd")
        st.metric("Form", "W-W-D-W-W")
    with col2:
        st.markdown("### VS")
    with col3:
        st.markdown(f"### {away_team}")
        st.metric("ELO Rating", "1,723")
        st.metric("League Position", "6th")
        st.metric("Form", "W-L-W-D-L")
    
    # Section 2: Prediction
    st.subheader("🎯 Outcome Probabilities")
    home_prob = 0.62
    draw_prob = 0.23
    away_prob = 0.15
    
    fig = create_probability_chart(...)
    st.plotly_chart(fig, use_container_width=True)
    
    # Section 3: Team Stats
    st.subheader("📊 Team Comparison")
    # Side-by-side stats
    
    # Section 4: NLP
    st.subheader("📰 Media Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} Sentiment: Positive** (+0.74)")
        st.write("- ✅ Full squad available")
        st.write("- ✅ Confidence high after UCL win")
    with col2:
        st.write(f"**{away_team} Sentiment: Neutral** (+0.12)")
        st.write("- ⚠️ Star striker doubtful")
        st.write("- ⚠️ Poor away record")
    
    # Section 5: Feature Importance
    st.subheader("🔬 What Drives This Prediction?")
    fig_importance = create_feature_importance_chart(...)
    st.plotly_chart(fig_importance)
    
    # Section 6: Reasoning
    st.subheader("🧠 AI Reasoning")
    st.info(f"""
    **{home_team}** is favored to win based on:
    - Significant ELO advantage (+124 points)
    - Strong home record (78% win rate at home)
    - Excellent recent form vs. {away_team}'s inconsistency
    - Positive media sentiment indicates high team confidence
    - Historical dominance in head-to-head matchups
    """)

def show_model_insights():
    """
    Display model evaluation and ablation study.
    
    CRITICAL SECTION - This proves NLP value!
    
    Components:
        - Model comparison table
        - Ablation study visualization
        - Feature importance across models
        - Performance metrics breakdown
    """
    st.title("📊 Model Performance Insights")
    
    # Ablation Study (MANDATORY)
    st.header("🔬 Ablation Study: Impact of NLP Features")
    
    ablation_data = {
        'Model': ['Baseline (Stats Only)', '+ NLP Features'],
        'Accuracy': [0.523, 0.568],
        'F1-Score': [0.511, 0.555]
    }
    
    df_ablation = pd.DataFrame(ablation_data)
    
    fig = px.bar(
        df_ablation,
        x='Model',
        y='Accuracy',
        title='NLP Features Improve Match Prediction Accuracy by 4.5%',
        text='Accuracy'
    )
    st.plotly_chart(fig)
    
    st.success("""
    **Key Finding:** Adding NLP-derived features (pre-match sentiment, 
    injury concerns, team confidence) improves match prediction accuracy 
    by 4.5 percentage points. This demonstrates that media narratives 
    contain predictive signals beyond raw team statistics.
    """)
```

### Deployment Requirements:

**Platform:** Streamlit Community Cloud (free)
- URL format: `https://[username]-match-outcome-predictor.streamlit.app`
- Automatic deployment from GitHub
- No credit card required

**Configuration File:** `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

**Secrets Management (if using APIs):**
```toml
# .streamlit/secrets.toml (DO NOT COMMIT)
NEWS_API_KEY = "your_api_key_here"
HUGGINGFACE_TOKEN = "your_token_here"
```

**Performance Requirements:**
- Initial load time: <10 seconds
- Prediction response time: <5 seconds
- Support concurrent users: 10+
- Mobile responsive: Yes

---

## 🧪 TESTING STANDARDS

### Required Tests:

**Data Validation Tests:**
```python
# tests/test_data_quality.py

def test_no_missing_labels():
    """Ensure all matches have valid outcome labels."""
    df = load_processed_data()
    assert df['outcome'].isna().sum() == 0

def test_feature_ranges():
    """Validate feature values are in expected ranges."""
    df = load_processed_data()
    assert (df['elo_difference'] >= -1500).all() and (df['elo_difference'] <= 1500).all()
    assert (df['sentiment_gap'] >= -2).all() and (df['sentiment_gap'] <= 2).all()

def test_no_data_leakage():
    """Ensure test set is chronologically after training set."""
    train_dates = load_train_dates()
    test_dates = load_test_dates()
    assert train_dates.max() < test_dates.min()
```

**Model Tests:**
```python
# tests/test_ml_pipeline.py

def test_model_predictions_valid():
    """Ensure model outputs valid class predictions."""
    model = load_trained_model()
    test_features = load_test_features()
    predictions = model.predict(test_features)
    
    valid_classes = ['Home Win', 'Draw', 'Away Win']
    assert all(pred in valid_classes for pred in predictions)

def test_model_reproducibility():
    """Ensure predictions are deterministic."""
    model = load_trained_model()
    test_features = load_test_features()
    
    preds1 = model.predict(test_features)
    preds2 = model.predict(test_features)
    
    assert (preds1 == preds2).all()
```

**NLP Tests:**
```python
# tests/test_nlp_pipeline.py

def test_sentiment_analyzer_loads():
    """Ensure Hugging Face model loads correctly."""
    from models.nlp_analysis.sentiment_analyzer import load_model
    model = load_model()
    assert model is not None

def test_sentiment_output_range():
    """Validate sentiment scores are in [-1, 1]."""
    from models.nlp_analysis.sentiment_analyzer import analyze_sentiment
    
    test_text = "This player is absolutely amazing!"
    score = analyze_sentiment(test_text)
    
    assert -1 <= score <= 1
```

---

## 📚 DOCUMENTATION REQUIREMENTS

### README.md Structure:
```markdown
# Required Sections (in order):

1. Project Title & Brief Description (2-3 sentences)
2. Live Demo Link (Streamlit URL)
3. Key Results Summary (accuracy, NLP impact)
4. Features & Capabilities
5. Tech Stack
6. Installation & Setup (for local development)
7. Usage Examples
8. Project Structure (brief)
9. Methodology (link to docs/methodology.md)
10. Results & Evaluation (link to docs/evaluation_results.md)
11. Future Improvements
12. Credits & License
13. Contact Information
```

### docs/methodology.md:

Must explain:
- Problem formulation
- Data collection process
- Feature engineering decisions
- Model selection rationale
- Evaluation strategy
- Deployment architecture

### docs/evaluation_results.md:

Must include:
- All model comparison metrics
- Ablation study results (WITH CHARTS)
- Confusion matrices
- Feature importance rankings
- Performance on test set
- Error analysis (what the model gets wrong)

---

## ⚠️ COMMON PITFALLS TO AVOID

### Data Issues:

❌ **DON'T:**
- Use data from before 2020 (too outdated)
- Mix different seasons without normalization
- Ignore missing data (impute or drop systematically)
- Use test data during feature engineering

✅ **DO:**
- Document all data cleaning decisions
- Version your datasets (add dates to filenames)
- Keep raw data separate from processed data
- Create data quality reports

### ML Issues:

❌ **DON'T:**
- Train on entire dataset (no test set)
- Use accuracy alone for imbalanced classes
- Skip cross-validation
- Overfit with too many features

✅ **DO:**
- Stratified train/test split
- Use F1-score for imbalanced data
- 5-fold cross-validation minimum
- Feature selection based on importance

### NLP Issues:

❌ **DON'T:**
- Assume all sentiment is reliable
- Mix languages without translation
- Ignore sarcasm/irony in text
- Use outdated news (>12 months old)

✅ **DO:**
- Filter for English-only articles
- Document sentiment model limitations
- Weight recent news higher
- Validate sentiment on sample manually

### Deployment Issues:

❌ **DON'T:**
- Hardcode file paths
- Store API keys in code
- Ignore error handling in UI
- Deploy without testing on different devices

✅ **DO:**
- Use environment variables
- Store secrets in .streamlit/secrets.toml
- Add try-catch blocks everywhere
- Test on mobile before final submission

---

## 🚀 WORKFLOW GUIDELINES

### Git Commit Standards:
```bash
# Format: : 

Types:
  feat:     New feature
  fix:      Bug fix
  docs:     Documentation changes
  style:    Code formatting (no logic change)
  refactor: Code restructuring
  test:     Adding/updating tests
  chore:    Maintenance tasks

Examples:
  git commit -m "feat: add sentiment analysis pipeline"
  git commit -m "fix: handle missing player data gracefully"
  git commit -m "docs: update methodology with NLP details"
  git commit -m "refactor: optimize feature extraction loop"
```

### Branch Strategy:
```
main              - Stable, deployable code only
├── develop       - Integration branch
    ├── feature/data-collection
    ├── feature/ml-pipeline
    ├── feature/nlp-analysis
    └── feature/streamlit-ui
```

### Development Cycle:

**Week 1-2: Foundation**
- Setup project structure
- Collect and explore data
- Create baseline ML model
- Document initial findings

**Week 3-4: Core Development**
- Implement NLP pipeline
- Integrate NLP features into ML
- Train advanced models
- Run ablation study

**Week 5-6: Integration & Deployment**
- Build Streamlit interface
- Connect all components
- Deploy to Streamlit Cloud
- User testing and bug fixes

**Week 7: Finalization**
- Polish documentation
- Final testing
- Create submission materials
- Backup everything

---

## 📞 SUPPORT & RESOURCES

### When Stuck:

1. **Check this file first** (you might have missed something)
2. **Review module slides** (in /mnt/project/)
3. **Search existing issues** on GitHub
4. **Consult official docs:**
   - Scikit-learn: https://scikit-learn.org/stable/
   - Hugging Face: https://huggingface.co/docs
   - Streamlit: https://docs.streamlit.io
5. **Ask AI assistant** (provide context from this file)

### External Resources:

**Datasets & APIs:**
- football-data.org API: https://api.football-data.org/v4/ (free tier: 10 calls/min)
- Club ELO: http://clubelo.com/ (pre-calculated ELO ratings)
- NewsAPI: https://newsapi.org/ (free tier: 100 requests/day)

**Pre-trained Models:**
- Hugging Face Models: https://huggingface.co/models?pipeline_tag=sentiment-analysis
- Recommended: distilbert-base-uncased-finetuned-sst-2-english

**Tools:**
- Label Studio (for manual labeling if needed): https://labelstud.io/
- Weights & Biases (experiment tracking): https://wandb.ai/

---

## ✅ PRE-SUBMISSION CHECKLIST

**2 Weeks Before Deadline:**
- [ ] All code is committed and pushed to GitHub
- [ ] Streamlit app is deployed and accessible
- [ ] README.md is complete and professional
- [ ] All required documentation exists (methodology, evaluation)
- [ ] Model achieves ≥55% accuracy on test set
- [ ] Ablation study shows ≥2% improvement from NLP
- [ ] No hardcoded paths or API keys in code
- [ ] .gitignore prevents sensitive data commits

**1 Week Before Deadline:**
- [ ] UI tested on mobile and desktop
- [ ] All notebooks run without errors
- [ ] Test predictions on 20+ different matches
- [ ] Documentation reviewed for clarity
- [ ] GitHub README looks professional
- [ ] Deployment URL is stable
- [ ] Code is commented and clean

**1 Day Before Deadline:**
- [ ] Final test of entire system
- [ ] Screenshot of working app (backup)
- [ ] Export environment (requirements.txt verified)
- [ ] Backup entire project locally
- [ ] Submit GitHub URL via official channel
- [ ] Verify submission confirmation received

---

## 🎓 GRADE 6.0 REQUIREMENTS SUMMARY

To achieve the highest grade, this project must demonstrate:

**Technical Excellence (40%):**
- ✅ Clean, well-documented code
- ✅ Proper ML pipeline (preprocessing, training, evaluation)
- ✅ Advanced NLP implementation (Transformers, not simple VADER)
- ✅ Efficient data processing
- ✅ Professional Git usage

**Integration Quality (30%):**
- ✅ NLP features genuinely improve ML performance (proven quantitatively)
- ✅ Multiple data sources combined seamlessly
- ✅ Ablation study with clear visualizations
- ✅ Feature engineering shows creativity

**Deployment & Usability (15%):**
- ✅ Working, stable Streamlit app
- ✅ Professional UI/UX
- ✅ Fast response times (<5 sec)
- ✅ Mobile-friendly design
- ✅ Clear user guidance

**Documentation (15%):**
- ✅ Comprehensive README
- ✅ Detailed methodology documentation
- ✅ Evaluation results with interpretations
- ✅ Code comments explain non-obvious decisions
- ✅ Reproducible setup instructions

---

## 🔒 FINAL NOTES

**Philosophy:**
This project should demonstrate **practical AI skills**, not just theoretical knowledge. Every component should work together to solve a real problem: predicting match outcomes more accurately than statistics alone.

**Quality > Quantity:**
Two excellent blocks (ML + NLP) are better than three mediocre blocks. If Computer Vision doesn't add clear value, skip it confidently.

**Document Decisions:**
When you make a choice (model selection, feature engineering, etc.), document WHY. This shows critical thinking.

**Test Early, Test Often:**
Don't wait until the end to test integration. Test each component as you build it.

**Ask for Help:**
If stuck for >2 hours, ask for help (AI assistant, classmates, instructor). Time is limited.

---

**Last Updated:** [Current Date]  
**Version:** 1.0  
**Maintainer:** [Your Name]

---

**⚠️ REMEMBER: Any AI assistant should read this ENTIRE file before making suggestions or writing code. Ignoring these instructions may result in wasted effort or incompatible implementations.**

---

END OF INSTRUCTIONS.MD