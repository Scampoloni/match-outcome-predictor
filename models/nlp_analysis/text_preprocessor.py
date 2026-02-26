"""
Text preprocessing utilities for football news articles.

Cleans raw HTML-scraped text before feeding it to the sentiment model or spaCy.
"""

import re
import string

import spacy

# Lazy-loaded spaCy model (en_core_web_sm)
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser"])
    return _NLP


def clean_text(text: str) -> str:
    """Remove HTML artifacts, excess whitespace, and non-ASCII junk."""
    # Strip HTML tags that might have slipped through
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove special characters but keep sentence punctuation
    text = re.sub(r"[^\w\s.,!?'\"-]", " ", text)
    return text.strip()


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Return lemmatized tokens (no stopwords, no punctuation)."""
    nlp = _get_nlp()
    doc = nlp(clean_text(text))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
    ]
    return tokens


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text using spaCy NER."""
    nlp = _get_nlp()
    doc = nlp(clean_text(text))
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities


def chunk_text(text: str, max_chars: int = 512) -> list[str]:
    """Split text into chunks suitable for transformer input (by sentence)."""
    nlp = _get_nlp()
    # Add sentencizer if not present
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    doc = nlp(clean_text(text))
    chunks, current = [], ""
    for sent in doc.sents:
        if len(current) + len(sent.text) < max_chars:
            current += " " + sent.text
        else:
            if current:
                chunks.append(current.strip())
            current = sent.text
    if current:
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]
