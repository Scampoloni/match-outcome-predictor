import feedparser
import requests
import streamlit as st
from transformers import pipeline

# Load DistilBERT model efficiently (caching prevents reloading on every query)
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=None)

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.skysports.com/rss/12040",
    "https://www.theguardian.com/football/rss",
    "https://www.goal.com/feeds/en/news",
]

def fetch_articles_for_match(home_team: str, away_team: str) -> list[str]:
    """
    1. Search RSS-Feeds (no API Key needed)
    2. GNews API as a supplementary source
    Returns a list of article texts (title + description) matching the teams.
    """
    articles = []
    # Team name variations for better matching
    h_tokens = set(home_team.lower().replace('fc', '').replace('cf', '').split())
    a_tokens = set(away_team.lower().replace('fc', '').replace('cf', '').split())
    
    # 1. RSS feeds
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                text = f"{entry.title} {entry.get('description', '')}"
                text_lower = text.lower()
                # Simple heuristic: at least part of home or away team in the string
                if any(t in text_lower for t in h_tokens if len(t) > 3) or \
                   any(t in text_lower for t in a_tokens if len(t) > 3):
                    articles.append(text)
        except Exception:
            pass

    # 2. GNews API
    try:
        api_key = st.secrets["GNEWS_API_KEY"]
        # Basic query looking for both teams
        query = f'"{home_team.split()[0]}" OR "{away_team.split()[0]}"'
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=10&apikey={api_key}"
        resp = requests.get(url).json()
        for art in resp.get("articles", []):
            text = f"{art['title']} {art['description']}"
            articles.append(text)
    except Exception:
        pass
        
    # Deduplicate
    return list(set(articles))

def calculate_keyword_score(texts: list[str], keywords: list[str], team_name: str) -> float:
    """Helper to count how often keywords appear near a team name."""
    score = 0
    t_name = team_name.lower().split()[0]
    for text in texts:
        t = text.lower()
        if t_name in t:
            for kw in keywords:
                if kw in t:
                    score += 1
    # Normalize heavily simplified between 0 and 1
    return min(1.0, score / max(1, len(texts)))

def calculate_match_sentiment(home_team: str, away_team: str) -> dict:
    """
    Executes live sentiment analysis using DistilBERT on scraped text 
    about the two specific teams.
    """
    texts = fetch_articles_for_match(home_team, away_team)
    
    res = {
        "sentiment_mean_home": 0.0,
        "sentiment_mean_away": 0.0,
        "sentiment_gap": 0.0,
        "injury_concern_score_home": 0.0,
        "injury_concern_score_away": 0.0,
        "confidence_score_home": 0.0,
        "confidence_score_away": 0.0,
        "hype_level": len(texts) / 20.0, # Normalizing assumption
        "articles_found": len(texts),
        "data_available": len(texts) > 0
    }
    
    if not res["data_available"]:
        return res
        
    sentiment_model = load_sentiment_pipeline()
    
    h_sentiments = []
    a_sentiments = []
    
    h_key = home_team.split()[0].lower()
    a_key = away_team.split()[0].lower()
    
    for text in texts:
        # Avoid extremely long texts breaking BERT
        text_trunc = text[:512] 
        try:
            preds = sentiment_model(text_trunc)[0]
            # Convert DISTILBERT output to a -1 to 1 score
            score = 0
            for p in preds:
                if p['label'] == 'POSITIVE':
                    score += p['score']
                elif p['label'] == 'NEGATIVE':
                    score -= p['score']
            
            tl = text_trunc.lower()
            if h_key in tl:
                h_sentiments.append(score)
            if a_key in tl:
                a_sentiments.append(score)
        except:
            pass
            
    if h_sentiments:
        res["sentiment_mean_home"] = sum(h_sentiments) / len(h_sentiments)
    if a_sentiments:
        res["sentiment_mean_away"] = sum(a_sentiments) / len(a_sentiments)
        
    res["sentiment_gap"] = res["sentiment_mean_home"] - res["sentiment_mean_away"]
    
    # Keyword based scores mirroring the original dataset structure
    injury_words = ["injury", "injured", "ruled out", "misses", "blow", "doubt"]
    conf_words = ["confident", "sure", "dominating", "ready", "favourite"]
    
    res["injury_concern_score_home"] = calculate_keyword_score(texts, injury_words, home_team)
    res["injury_concern_score_away"] = calculate_keyword_score(texts, injury_words, away_team)
    res["confidence_score_home"] = calculate_keyword_score(texts, conf_words, home_team)
    res["confidence_score_away"] = calculate_keyword_score(texts, conf_words, away_team)
    
    return res
