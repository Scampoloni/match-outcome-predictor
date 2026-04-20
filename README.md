# Match Outcome Predictor

Predicts football match outcomes (Home Win / Draw / Away Win) by combining team statistics, ELO ratings, and pre-match news sentiment — served through an interactive Streamlit dashboard with a built-in AI chat assistant.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://match-outcome-predictor.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

Given any two teams from Europe's top 5 leagues, the system:

1. Fetches **live team statistics** — ELO ratings, league position, recent form, goals scored/conceded
2. Scrapes **pre-match news articles** and runs **DistilBERT sentiment analysis** on them
3. Combines both into a **45+ feature vector** fed into a trained XGBoost classifier
4. Returns outcome probabilities with a breakdown of the key drivers
5. Optionally lets you query the **Gemini-powered RAG chatbot** for natural-language analysis

---

## Key Results

| Model | Stats Only | + NLP Features | Delta |
|-------|-----------|----------------|-------|
| Logistic Regression | 52.28% | 52.66% | +0.38% |
| Random Forest | 51.90% | 53.61% | **+2.47%** |
| XGBoost | 50.57% | 52.66% | **+2.09%** |

- **Dataset:** 3,503 matches across 5 leagues (2022–2025) + 802 news articles
- **Baseline (always predict Home Win):** ~45% → model beats it by 7–9 pp
- **Best model:** Random Forest with NLP at 53.61%
- **Top features:** `strength_ratio`, `h2h_home_wins`, `h2h_away_wins`, `form_points_home`

> The ablation study confirms that NLP-derived sentiment features provide a measurable and consistent lift, particularly for XGBoost (+2.09%) and Random Forest (+2.47%).

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn, XGBoost |
| NLP | Hugging Face Transformers (DistilBERT), spaCy |
| Deep Learning | PyTorch |
| Data collection | football-data.org API, ClubELO, NewsAPI, RSS feeds |
| UI | Streamlit, Plotly |
| AI Chat | Google Gemini 2.5 Flash (RAG) |
| Utilities | pandas, numpy, python-dotenv, rapidfuzz, feedparser |

---

## Project Structure

```
match-outcome-predictor/
│
├── app/                        # Streamlit web application
│   ├── streamlit_app.py        # Main dashboard
│   ├── utils.py                # Model loading & inference
│   ├── visualizations.py       # Plotly chart factories
│   ├── live_features.py        # Real-time feature extraction from APIs
│   ├── live_news.py            # Live news scraping & DistilBERT sentiment
│   └── rag_system.py           # Gemini RAG chatbot with fuzzy team matching
│
├── data/
│   ├── raw/                    # Raw API/scraped data (not tracked)
│   ├── processed/              # Feature-engineered datasets (not tracked)
│   └── scrapers/               # Data collection & feature engineering scripts
│       ├── collect_matches.py  # Fetch historical matches (football-data.org)
│       ├── collect_news.py     # Scrape news articles (NewsAPI + RSS)
│       ├── build_features.py   # Compute rolling stats, ELO, H2H features
│       └── data_validator.py   # Dataset quality checks
│
├── models/
│   ├── ml_classification/      # ML training & evaluation
│   │   ├── train.py            # Train LR / RF / XGBoost (with & without NLP)
│   │   ├── evaluate.py         # Metrics, confusion matrices, feature importance
│   │   ├── model_comparison.py # Side-by-side ablation results
│   │   └── saved_models/       # Trained .pkl files (not tracked)
│   └── nlp_analysis/           # NLP pipeline
│       ├── sentiment_analyzer.py   # DistilBERT inference on articles
│       ├── compare_models.py       # DistilBERT vs RoBERTa comparison
│       ├── feature_extractor.py    # Merge NLP features into ML dataset
│       └── text_preprocessor.py   # HTML stripping, spaCy lemmatization
│
├── notebooks/                  # Exploration & analysis (numbered sequence)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_nlp_pipeline.ipynb
│   ├── 04_ml_baseline.ipynb
│   ├── 05_model_optimization.ipynb
│   └── 06_ablation_study.ipynb
│
├── docs/
│   ├── methodology.md          # Problem formulation & design decisions
│   └── evaluation_results.md   # Full metrics & ablation results
│
├── tests/
│   ├── test_data_quality.py
│   ├── test_ml_pipeline.py
│   └── test_nlp_pipeline.py
│
├── .env.example                # API key template
├── requirements.txt
└── setup.py
```

---

## Local Setup

**Prerequisites:** Python 3.10+, ~4 GB disk space for models

```bash
# 1. Clone the repository
git clone https://github.com/Scampoloni/match-outcome-predictor.git
cd match-outcome-predictor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Configure API keys
cp .env.example .env
# Edit .env — required keys: FOOTBALL_DATA_API_KEY, NEWS_API_KEY
# Optional: GNEWS_API_KEY, GEMINI_API_KEY (for the RAG chatbot)
```

### Reproduce the full pipeline (optional)

If you want to retrain from scratch rather than use pre-built models:

```bash
# Collect data
python data/scrapers/collect_matches.py
python data/scrapers/collect_news.py
python data/scrapers/build_features.py

# Run NLP pipeline
python models/nlp_analysis/sentiment_analyzer.py
python models/nlp_analysis/feature_extractor.py

# Train models (with and without NLP for ablation)
python models/ml_classification/train.py
python models/ml_classification/train.py --no-nlp

# Evaluate
python models/ml_classification/evaluate.py
python models/ml_classification/model_comparison.py
```

### Launch the app

```bash
streamlit run app/streamlit_app.py
```

The app fetches live data on demand — no pre-downloaded dataset required to run predictions.

---

## How It Works

### 1. Feature Engineering

For each match, 45+ features are computed across two groups:

**Statistical features** (from football-data.org + ClubELO):
- ELO ratings and ELO difference between teams
- Rolling form points (last 5 matches)
- Goals scored / conceded per game
- Head-to-head record (wins, draws, losses)
- Current league position
- Days of rest since last match
- Derived: `strength_ratio` (ELO home / ELO away), `goal_difference_delta`

**NLP features** (from DistilBERT on pre-match articles):
- `sentiment_mean_home` / `sentiment_mean_away` — average polarity score [-1, +1]
- `sentiment_gap` — home minus away sentiment
- `confidence_score` — media confidence tone
- `injury_concern_score` — injury keyword density near team name

### 2. Models

Three classifiers are trained using stratified 5-fold cross-validation on an 85/15 train/test split. Each is trained twice — once with full features and once without NLP features — to produce the ablation study.

### 3. RAG Chatbot

The Gemini 2.5 Flash chatbot receives a system prompt containing live backend data (stats + sentiment) for the selected teams. It cannot hallucinate statistics because they are injected directly. Fuzzy team name matching via `rapidfuzz` handles typos and name variants.

---

## Documentation

- [Methodology](docs/methodology.md) — problem formulation, feature design, model selection
- [Evaluation Results](docs/evaluation_results.md) — full metrics, confusion matrices, ablation study

---

## License

MIT — see [LICENSE](LICENSE) for details.
