import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

def get_kmeans(k: int = 20) -> KMeans:
    """Creates a pre-configured K-Means clustering model.

    Args:
        k (int, optional): The number of clusters. Defaults to 20.

    Returns:
        KMeans: An initialized Scikit-Learn KMeans instance.
    """
    logger.info(f"Initializing K-Means model with k={k}")
    return KMeans(
        n_clusters=k, 
        init='k-means++', 
        n_init=10, 
        random_state=config.RANDOM_SEED
    )

def get_gmm(k: int = 20, covariance_type: str = 'spherical') -> GaussianMixture:
    """Creates a pre-configured Gaussian Mixture Model.

    Args:
        k (int, optional): The number of mixture components. Defaults to 20.
        covariance_type (str, optional): Type of covariance parameters to use. Defaults to 'spherical'.

    Returns:
        GaussianMixture: An initialized Scikit-Learn GaussianMixture instance.
    """
    logger.info(f"Initializing GMM with k={k} and covariance='{covariance_type}'")
    return GaussianMixture(
        n_components=k, 
        covariance_type=covariance_type, 
        random_state=config.RANDOM_SEED
    )

def get_hierarchical(k: int = 20, metric: str = 'cosine', linkage: str = 'complete') -> AgglomerativeClustering:
    """Creates a pre-configured Hierarchical (Agglomerative) clustering model.

    Args:
        k (int, optional): The number of clusters to find. Defaults to 20.
        metric (str, optional): Metric used to compute the linkage. Defaults to 'cosine'.
        linkage (str, optional): Which linkage criterion to use. Defaults to 'complete'.

    Returns:
        AgglomerativeClustering: An initialized Scikit-Learn AgglomerativeClustering instance.
    """
    logger.info(f"Initializing Hierarchical Clustering with k={k}, metric='{metric}'")
    return AgglomerativeClustering(
        n_clusters=k, 
        metric=metric, 
        linkage=linkage
    )