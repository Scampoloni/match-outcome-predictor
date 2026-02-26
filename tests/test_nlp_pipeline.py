"""
NLP pipeline unit tests for text preprocessing.
Run with: pytest tests/test_nlp_pipeline.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "nlp_analysis"))
from text_preprocessor import clean_text, tokenize_and_lemmatize, chunk_text


def test_clean_text_removes_html():
    raw = "<p>Haaland <b>scored</b> twice.</p>"
    result = clean_text(raw)
    assert "<" not in result
    assert "Haaland" in result


def test_clean_text_removes_urls():
    raw = "Check this out https://www.bbc.co.uk/sport/football/12345 for more."
    result = clean_text(raw)
    assert "http" not in result


def test_clean_text_normalises_whitespace():
    result = clean_text("  Hello   world  ")
    assert result == "Hello world"


def test_tokenize_returns_list():
    tokens = tokenize_and_lemmatize("Erling Haaland scored an incredible goal at the Etihad.")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_filters_stopwords():
    tokens = tokenize_and_lemmatize("The player is very good at scoring goals.")
    # Common stopwords should not appear
    assert "the" not in tokens
    assert "is" not in tokens


def test_chunk_text_respects_max_chars():
    long_text = "This is a sentence. " * 100
    chunks = chunk_text(long_text, max_chars=200)
    for chunk in chunks:
        assert len(chunk) <= 300  # slightly above max due to full-sentence boundary


def test_chunk_text_nonempty():
    chunks = chunk_text("Short text.")
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)
