import numpy as np
from sklearn.metrics import silhouette_score

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_clusters(X: np.ndarray, labels: np.ndarray, max_samples: int = 10000) -> dict:
    """Calculates evaluation metrics for the clustering output.
    
    Samples the dataset for the Silhouette score if the number of documents 
    exceeds max_samples to prevent out-of-memory errors and long computation times.

    Args:
        X (np.ndarray): The dense feature matrix (embeddings).
        labels (np.ndarray): The cluster assignments for each document.
        max_samples (int, optional): Maximum rows to use for Silhouette calculation. Defaults to 10000.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    n_samples = X.shape[0]
    n_clusters = len(set(labels))
    
    if n_clusters < 2 or n_clusters >= n_samples:
        logger.warning(f"Invalid number of clusters ({n_clusters}) for Silhouette score. Returning -1.0")
        return {"silhouette_score": -1.0}

    sample_size = max_samples if n_samples > max_samples else None
    
    if sample_size:
        logger.info(f"Calculating Silhouette score using a random sample of {sample_size} documents...")
    else:
        logger.info("Calculating Silhouette score on the full dataset...")

    score = silhouette_score(
        X, 
        labels, 
        sample_size=sample_size, 
        random_state=config.RANDOM_SEED
    )
    
    logger.info(f"Evaluation complete. Silhouette Score: {score:.4f}")
    
    return {
        "silhouette_score": float(score)
    }