import os
from pathlib import Path

# Direcrories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Dataset Path
WIKI_CSV_PATH = RAW_DATA_DIR / "people_wiki.csv"

# Global Parameters
RANDOM_SEED = 42

# TFIDF Vectorizer Parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.7
NEWSGROUPS_MIN_WORDS = 25
WIKI_MAX_WORDS_PERCENTILE = 99
NEWSGROUPS_MAX_WORDS_PERCENTILE = 99