# ⚽ Match Outcome Predictor

> AI-powered football match prediction using Machine Learning and Natural Language Processing.
> University Project — AI Applications FS 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
![Status](https://img.shields.io/badge/status-in_development-yellow.svg)

---

## 🎯 Project Overview

Predict football match outcomes (**Home Win / Draw / Away Win**) by combining:
- **Statistical Analysis:** Team form, ELO ratings, head-to-head records
- **Sentiment Analysis:** Pre-match news and media coverage (NLP)
- **[Optional] Formation Analysis:** Tactical setup recognition (Computer Vision)

**Use Case:** Sports analytics, betting insights, fantasy football optimization

---

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│   Data Sources       │    │   NLP Pipeline       │
│  ─ football-data.org │    │  ─ News scraping     │
│  ─ Club ELO          │───▶│  ─ DistilBERT        │
│  ─ News articles     │    │  ─ Sentiment scores  │
└─────────────────────┘    └────────┬────────────┘
                                    │
                           ┌────────▼────────────┐
                           │   Feature Matrix     │
                           │  (team stats + NLP)  │
                           └────────┬────────────┘
                                    │
                           ┌────────▼────────────┐
                           │   ML Classifier      │
                           │  ─ Logistic Reg.     │
                           │  ─ Random Forest     │
                           │  ─ XGBoost (best)    │
                           └────────┬────────────┘
                                    │
                           ┌────────▼────────────┐
                           │  Streamlit App       │
                           │  (live demo)         │
                           └─────────────────────┘
```

---

## 🏆 Key Results

*[To be updated after model training]*

| Model | Without NLP | With NLP | Improvement |
|-------|-------------|----------|-------------|
| Logistic Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | **+X%** |

- Dataset: 2000+ matches, 500+ news articles analyzed

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/match-outcome-predictor.git
cd match-outcome-predictor
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Collect match data
python data/scrapers/collect_matches.py

# 4. Collect news articles
python data/scrapers/collect_news.py

# 5. Run NLP pipeline
python models/nlp_analysis/sentiment_analyzer.py

# 6. Train ML models
python models/ml_classification/train.py

# 7. Launch Streamlit app
streamlit run app/streamlit_app.py
```

---

## 💡 Usage Example

```python
from models.ml_classification.predict import MatchPredictor

predictor = MatchPredictor()

result = predictor.predict(
    home_team="Juventus",
    away_team="Roma",
    match_date="2026-03-15"
)

print(result)
# {
#   'home_win_prob': 0.62,
#   'draw_prob': 0.23,
#   'away_win_prob': 0.15,
#   'prediction': 'Home Win',
#   'confidence': 0.62,
#   'key_factors': ['ELO difference', 'Home advantage', 'Positive sentiment']
# }
```

---

## 📁 Project Structure

```
match-outcome-predictor/
├── data/
│   ├── raw/                          # Raw API/scraped data
│   ├── processed/                    # Cleaned datasets
│   └── scrapers/                     # Data collection scripts
├── models/
│   ├── ml_classification/            # ML training & prediction
│   ├── nlp_analysis/                 # Sentiment analysis
│   └── [optional] computer_vision/   # Formation recognition
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_nlp_pipeline.ipynb
│   ├── 04_ml_baseline.ipynb
│   ├── 05_model_optimization.ipynb
│   └── 06_ablation_study.ipynb
├── app/
│   ├── streamlit_app.py              # Main web app
│   ├── utils.py                      # Helper functions
│   └── visualizations.py             # Charts & graphs
├── docs/
│   ├── methodology.md
│   └── evaluation_results.md
├── tests/
│   ├── test_data_quality.py
│   ├── test_ml_pipeline.py
│   └── test_nlp_pipeline.py
├── INSTRUCTIONS.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── setup.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost |
| NLP | Hugging Face Transformers, spaCy |
| Data | pandas, BeautifulSoup, football-data.org API |
| UI | Streamlit, Plotly |
| Language | Python 3.10+ |

---

## 📊 Methodology

Detailed methodology in [`docs/methodology.md`](docs/methodology.md)

1. **Data Collection:** 2000+ historical matches + 500+ news articles
2. **Feature Engineering:** 30+ statistical features + 15+ NLP-derived features
3. **Model Training:** Compare Logistic Regression, Random Forest, XGBoost
4. **Ablation Study:** Prove NLP features improve accuracy
5. **Deployment:** Streamlit chat interface with real-time predictions

---

## 📈 Evaluation

- **Accuracy, Precision, Recall, F1-Score** per class
- **Confusion matrix** visualization
- **Feature importance** (XGBoost)
- **Ablation study**: NLP vs. stats-only performance

Full results in [`docs/evaluation_results.md`](docs/evaluation_results.md)

---

## 🔮 Future Improvements

- [ ] Add live match prediction (real-time data)
- [ ] Incorporate weather data
- [ ] Player-level analysis (key player availability)
- [ ] Betting odds integration (market consensus)
- [ ] Multi-language news support

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**⚠️ Disclaimer:** This project is for educational purposes only. Predictions should not be used for real-money betting decisions.

---

*University Project — AI Applications FS 2026 | Deadline: June 7, 2026*
