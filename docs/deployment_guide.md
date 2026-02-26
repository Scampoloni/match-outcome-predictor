# Deployment Guide

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (football-data.org, NewsAPI)

# Run Streamlit locally
streamlit run app/streamlit_app.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub (must be public or connected to Streamlit Cloud)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch `main`, and entry point `app/streamlit_app.py`
5. Click **Deploy**

The app will be available at `https://your-username-match-outcome-predictor-app-streamlit-app-XXXXX.streamlit.app`

## Required Files for Cloud Deployment

Streamlit Cloud will not have access to `data/` (gitignored). Options:

**Option A — Embed small dataset**
Add a small `data/processed/features_complete.csv` (200 matches) to the repo (unignore it for demo purposes).

**Option B — Use Streamlit Secrets + cloud storage**
Upload the full CSV to S3/GCS and load it at runtime via URL.

**Option C — Pre-built demo mode**
Serve cached predictions for a fixed list of known matchups.

Recommended for submission: **Option A** with a curated 200-match subset.

## GitHub Actions CI

The workflow at `.github/workflows/deploy.yml` runs `pytest` on every push to `main`.
Streamlit Cloud auto-deploys when tests pass.
