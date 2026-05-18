import pytest
import pandas as pd
import numpy as np
from src.features import TfidfLsaVectorizer, TransformerVectorizer

@pytest.fixture
def sample_data():
    """Provides a tiny sample dataset for testing vectorizers."""
    return pd.Series([
        "machine learning algorithms analyze data",
        "deep neural networks require large datasets",
        "artificial intelligence is transforming industries"
    ])

def test_tfidf_lsa_vectorizer(sample_data):
    """Tests if TF-IDF + LSA produces a dense matrix of the expected shape."""
    # We use very small constraints just for the tiny test dataset
    vectorizer = TfidfLsaVectorizer(max_features=10, n_components=2)
    vectorizer.tfidf.min_df = 1 
    
    embeddings = vectorizer.fit_transform(sample_data)
    
    assert isinstance(embeddings, np.ndarray), "Output must be a numpy array"
    assert embeddings.shape[0] == 3, "Output should have 3 rows"
    assert embeddings.shape[1] == 2, "Output should have 2 dimensions (n_components)"

def test_transformer_vectorizer(sample_data):
    """Tests if the SentenceTransformer produces a dense matrix of the expected shape."""
    vectorizer = TransformerVectorizer(model_name='all-MiniLM-L6-v2')
    
    embeddings = vectorizer.fit_transform(sample_data)
    
    assert isinstance(embeddings, np.ndarray), "Output must be a numpy array"
    assert embeddings.shape[0] == 3, "Output should have 3 rows"
    assert embeddings.shape[1] == 384, "all-MiniLM-L6-v2 should output 384 dimensions"