import pytest
import numpy as np
from src.cluster import get_kmeans, get_gmm, get_hierarchical

@pytest.fixture
def dummy_embeddings():
    """Generates a random dense matrix simulating LSA or Transformer output.
    
    Returns 50 documents, each with 50 dimensions.
    """
    np.random.seed(42)
    return np.random.rand(50, 50)

def test_kmeans_factory(dummy_embeddings):
    """Tests if K-Means factory produces a working model."""
    model = get_kmeans(k=3)
    labels = model.fit_predict(dummy_embeddings)
    
    assert len(labels) == 50, "Should output exactly 50 labels"
    assert len(set(labels)) == 3, "Should create exactly 3 unique clusters"

def test_gmm_factory(dummy_embeddings):
    """Tests if GMM factory produces a working model."""
    model = get_gmm(k=3)
    labels = model.fit_predict(dummy_embeddings)
    
    assert len(labels) == 50, "Should output exactly 50 labels"
    # GMM may collapse an empty cluster because of soft assignments, so we check for <= 3 clusters
    assert len(set(labels)) <= 3, "Should create up to 3 unique clusters"

def test_hierarchical_factory(dummy_embeddings):
    """Tests if Hierarchical factory produces a working model."""
    model = get_hierarchical(k=3)
    labels = model.fit_predict(dummy_embeddings)
    
    assert len(labels) == 50, "Should output exactly 50 labels"
    assert len(set(labels)) == 3, "Should create exactly 3 unique clusters"