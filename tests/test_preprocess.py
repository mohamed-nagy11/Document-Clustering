import pandas as pd
from src.preprocess import TextCleaner, LinguisticProcessor

def test_text_cleaner():
    """Tests if URLs, numbers, punctuation, and contractions are cleaned correctly."""
    cleaner = TextCleaner()
    raw_data = pd.Series([
        "Check this out: https://example.com!!!",
        "I can't believe it's 2026."
    ])
    
    cleaned_data = cleaner.transform(raw_data)
    
    assert "https://example.com" not in cleaned_data.iloc[0], "TextCleaner failed to remove URL"
    assert "!" not in cleaned_data.iloc[0], "TextCleaner failed to remove punctuation"
    assert "cannot" in cleaned_data.iloc[1], "TextCleaner failed to expand contractions"
    assert "2026" not in cleaned_data.iloc[1], "TextCleaner failed to remove numbers"

def test_linguistic_processor():
    """Tests if stopwords are removed and words are properly lemmatized."""
    processor = LinguisticProcessor()
    
    # Providing pre-cleaned text exactly as it would arrive from the TextCleaner
    cleaned_data = pd.Series([
        "the medical doctors are conducting clinical trials"
    ])
    
    processed_data = processor.transform(cleaned_data)
    
    assert "the" not in processed_data.iloc[0], "LinguisticProcessor failed to remove stopword"
    assert "are" not in processed_data.iloc[0], "LinguisticProcessor failed to remove stopword"
    assert "doctor" in processed_data.iloc[0], "LinguisticProcessor failed to lemmatize 'doctors'"
    assert "conduct" in processed_data.iloc[0], "LinguisticProcessor failed to lemmatize 'conducting'"