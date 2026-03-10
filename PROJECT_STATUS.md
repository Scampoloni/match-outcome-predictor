# ⚽ Match Outcome Predictor - Project Status & Roadmap

Dieses Dokument fasst den aktuellen Entwicklungsstand des "Match Outcome Predictor" zusammen und bietet einen Leitfaden für mögliche zukünftige Erweiterungen.

---

## 🟢 Aktueller Status: Was bereits implementiert ist

Das Projekt verfügt über eine vollständige End-to-End-Pipeline, die statistisches Machine Learning mit Natural Language Processing (NLP) und einer LLM-gestützten Benutzeroberfläche kombiniert. 

### 1. Daten & Feature Engineering (Backend)
- **Historische Daten-Pipeline:** Automatisierter Abruf von über 3500 Matches der Top 5 Ligen via `football-data.org`.
- **ELO-Integration:** Dynamisches Merging von historischen ELO-Ratings via `clubelo.com`.
- **Feature Engineering:** Komplexe Ableitungen wie Formkurven (letzte 5 Spiele), Torverhältnisse, ELO-Differenzen und Strength-Ratios (`build_features.py`).
- **Live-Daten-Abruf:** Anbindung an Live-APIs, um aktuelle Tabellenstände, Formkurven und ELO-Werte für "On-the-Fly"-Vorhersagen zu generieren (`live_features.py`).

### 2. Natural Language Processing (NLP)
- **Sentiment-Analyse:** Nutzung von Hugging Face Transformer-Modellen (primär `DistilBERT`), um die Tonalität von englischsprachigen Fußballnachrichten zu bewerten.
- **Live-News-Scraping:** Extrahierung von rezenten Artikeln über ausgewählte Teams aus kostenfreien RSS-Feeds (BBC, SkySports, Guardian) und der GNews API (`live_news.py`).
- **Text-Features:** Berechnung von `sentiment_mean`, `sentiment_gap` (Stimmungsunterschied zwischen den Teams), `injury_concern_score` (Verletzungssorgen) und `confidence_score`.

### 3. Machine Learning Modelle
- **Modell-Vergleich:** Trainierte Modelle für Logistic Regression, Random Forest und XGBoost zur Vorhersage von (Home Win, Draw, Away Win).
- **Ablation Study:** Wissenschaftlicher Nachweis der Effektivität von NLP in der Vorhersage durch den Vergleich von Modellen mit und ohne Text-Features (`compare_models.py`).
- **Produktionsmodell:** Ein getuntes XGBoost-Modell, das Wahrscheinlichkeiten und Confidence-Scores für alle Ausgänge liefert.

### 4. User Interface & RAG-Chatbot (Frontend)
- **Streamlit Dashboard:** Ein modernes "Glassmorphism" UI im Dark-Mode für ansprechende Datenvisualisierungen.
- **Interaktive Charts:** Plotly-Integration für Wahrscheinlichkeitsverteilungen, statistische Team-Vergleiche und Sentiment-Gaps.
- **Erklärbare KI (XAI):** Ein regelbasiertes Reasoning-System, das dem Nutzer textuell erklärt, warum das Modell eine bestimmte Vorhersage trifft (basierend auf ELO, Form, Sentiment).
- **RAG-Chat-System (Retrieval Augmented Generation):** 
  - Eine eingebettete Google Gemini KI (`gemini-2.5-flash`), die durch `rapidfuzz` Tippfehler in Teamnamen erkennt (`rag_system.py`).
  - Die KI erhält über einen versteckten System-Prompt die exakten harten Daten (Live-Stats & Live-Sentiment) aus unserem Backend und nutzt diese als Wissensbasis.
  - Dadurch werden Halluzinationen eliminiert – der Bot agiert als fachkundiger Analyst über *unsere* Daten.

---

## 🚀 Future Roadmap: Was man noch erweitern könnte

Hier sind Ideen, um das Projekt auf ein noch höheres industrielles Level zu heben (z. B. für eine Weiterführung als Masterarbeit oder MLOps-Projekt):

### A. Daten & Feature-Erweiterungen (Data Science)
1. **Spieler-Statistiken (xG & xA):** 
   - Einbindung von "Expected Goals" (xG) oder "Expected Assists" (xA) der Schlüsselspieler über Scraper für fbref.com oder understat.
2. **Kaderwert & Verletzungs-APIs:**
   - Echtzeit-Verletzungsdatenbank-Anbindung oder Transfermarkt.de-Marktwert-Vergleich der Startaufstellungen, um den "Strength-Ratio" zu verbessern.
3. **Wetter- und Stadiondaten:**
   - Lokales Wetter am Spieltag (Regen vs. trockener Rasen) oder Auslastung/Stimmung im Stadion als weitere Features.

### B. NLP & Sentiment (Deep Learning)
1. **Multilinguale Sentiment-Analyse:**
   - Aktuell werden primär englische Quellen gelesen. Nutzung von `XLM-RoBERTa`, um italienische (Gazzetta), spanische (Marca) oder deutsche (Kicker) Lokalnachrichten in die Stimmung mit einfließen zu lassen.
2. **Named Entity Recognition (NER) für Verletzungen:**
   - Statt nur nach dem Wort "injury" zu suchen, ein Spacy-Modell trainieren, das erkennt *welcher* Spieler (z. B. Starspieler vs. Ersatzspieler) verletzt ist.
3. **Twitter/X-Sentiment:**
   - Einbindung von Social-Media-Stimmungen kurz vor Anpfiff, um die emotionale Aufgeladenheit von Derbys besser zu erfassen.

### C. Systemarchitektur & Deployment (MLOps)
1. **Cloud Deployment (Docker):**
   - Containerisierung der gesamten App mit Docker und automatisiertes Deployment auf AWS, Google Cloud Run oder Render.
2. **Automatisches Model-Retraining:**
   - Eine Airflow- oder GitHub Action Pipeline aufsetzen, die das XGBoost-Modell jeden Montagabend automatisch mit dem vergangenen Spieltagswochenende neu trainiert und updatet (Continuous Training).
3. **Datenbank-Anbindung:**
   - Weg von lokalen `.csv`-Dateien hin zu einer echten PostgreSQL- oder MongoDB-Datenbank für historische Daten und gecachte Live-Feeds.

### D. Frontend & UX (Benutzeroberfläche)
1. **Live-Wettquoten Vergleich:**
   - Anbindung einer Odds-API, um die Vorhersagewahrscheinlichkeiten unseres Modells visuell mit den Quoten echter Buchmacher (Value Betting) zu vergleichen.
2. **Mobile App:**
   - Das Dashboard via React Native oder Flutter in eine native App für iOS/Android verpacken.
