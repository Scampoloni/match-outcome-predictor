# Methodology

## Target Variable

Matches are assigned to one of three outcome classes based on the final score:

| Label | Condition |
|-------|-----------|
| Home Win | Home score > Away score |
| Draw | Home score = Away score |
| Away Win | Home score < Away score |

Typical distribution: ~45% Home Win, ~27% Draw, ~28% Away Win (varies by league).

## Feature Engineering

### Statistical Features (from football-data.org + ELO computation)
- ELO ratings (home, away, difference) — computed from match history
- Form (rolling points per game, last 5 matches)
- Goals per game (home, away)
- Head-to-head record (last 5 encounters)
- League positions (home, away)
- Rest days (home, away)
- Strength ratio (ELO home / ELO away)
- Goal difference delta (season goal difference home − away)

### NLP Features (from DistilBERT sentiment analysis of pre-match news)
- `sentiment_mean_home` / `sentiment_mean_away` — mean sentiment for each team (-1 to +1)
- `sentiment_gap` — difference (home − away)
- `confidence_score_home` / `confidence_score_away` — derived from sentiment (0 to 1)
- `injury_concern_score_home` / `injury_concern_score_away` — fraction of articles mentioning injuries
- `hype_level` — normalized article count (0 to 1)

## Models

Three classifiers are compared:

1. **Logistic Regression** (baseline, interpretable)
2. **Random Forest** (ensemble, feature importance)
3. **XGBoost** (gradient boosting, best expected performance)

All models use stratified 5-fold cross-validation. The final test set (15%) is held out until final evaluation. Temporal split ensures no future data leaks.

## Ablation Study

The core contribution of NLP features is quantified by training each model twice:
1. **Stats-only** — using only match statistics and ELO features
2. **Stats + NLP** — adding all NLP-derived features

The delta in accuracy and F1-macro is reported as the "NLP contribution."

## Data Collection

- **football-data.org API** — historical match results and standings (2,000+ matches across multiple seasons)
- **NewsAPI + BBC Sport** — pre-match news articles for NLP (scraped/fetched per match)
- **Club ELO** (optional) — historical ELO ratings for validation
