import numpy as np
from src.evaluate import evaluate_clusters

def test_evaluate_clusters_valid_output():
    """Tests if the evaluation correctly scores perfectly separated clusters."""
    # Create two perfectly separated groups of points
    X = np.array([
        [1.0, 1.0], [1.1, 1.1], [0.9, 0.9],    # Cluster 0
        [10.0, 10.0], [10.1, 10.1], [9.9, 9.9] # Cluster 1
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    
    metrics = evaluate_clusters(X, labels)
    
    assert isinstance(metrics, dict), "Should return a dictionary"
    assert "silhouette_score" in metrics, "Missing silhouette_score key"
    
    # A perfectly separated dataset should have a score close to 1.0
    assert metrics["silhouette_score"] > 0.8, "Score should be high for well-separated data"

def test_evaluate_clusters_invalid_clusters():
    """Tests if the function handles model failure gracefully."""
    X = np.array([[1.0, 1.0], [1.1, 1.1], [0.9, 0.9]])
    
    # Model collapsed and assigned everything to 1 single cluster
    labels = np.array([0, 0, 0]) 
    
    metrics = evaluate_clusters(X, labels)
    
    # The function should catch the error and return -1.0 instead of crashing the pipeline
    assert metrics["silhouette_score"] == -1.0, "Should return -1.0 for a single cluster"