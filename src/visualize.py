import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

from src.logger import get_logger

logger = get_logger(__name__)

def get_2d_cluster_plot(X_matrix: np.ndarray, cluster_labels: np.ndarray, original_texts: pd.Series, title: str = "2D Semantic Cluster Map") -> px.scatter:
    """Compresses high-dimensional text embeddings to 2D using SVD and t-SNE, and returns an interactive Plotly figure.

    Args:
        X_matrix (np.ndarray): The high-dimensional feature matrix (e.g., TF-IDF or Transformer embeddings).
        cluster_labels (np.ndarray): A 1D array of cluster assignments for each document.
        original_texts (pd.Series): A Pandas Series containing the original raw text documents, used to generate hover snippets.
        title (str, optional): The title of the generated plot. Defaults to "2D Semantic Cluster Map".

    Returns:
        plotly.graph_objs._figure.Figure: An interactive Plotly scatter plot object ready for rendering or MLflow logging.
    """
    logger.info("Starting dimensionality reduction (SVD -> t-SNE) for visualization...")
    
    # Linear compression (Sparse to 50D Dense)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_matrix)
    
    # Non-linear compression (50D to 2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_svd)
    
    df_plot = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'Cluster': [f"Cluster {c}" for c in cluster_labels],
        'Text_Snippet': original_texts.astype(str).str[:100] + "..."
    })
    
    fig = px.scatter(
        df_plot, x='x', y='y', color='Cluster',
        hover_data={'Text_Snippet': True, 'x': False, 'y': False},
        title=title,
        template="plotly_dark",
        width=1000, height=700
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    
    return fig


def get_dendrogram_plot(X_matrix: np.ndarray, method: str = 'complete', metric: str = 'cosine', title: str = "Hierarchical Clustering Dendrogram") -> plt.figure:
    """Generates and returns a static Matplotlib dendrogram figure for hierarchical clustering.

    Args:
        X_matrix (np.ndarray): The high-dimensional feature matrix.
        method (str, optional): The linkage algorithm to use for computing distances between clusters. Defaults to 'complete'.
        metric (str, optional): The distance metric to use (e.g., 'cosine', 'euclidean'). Defaults to 'cosine'.
        title (str, optional): The title of the generated plot. Defaults to "Hierarchical Clustering Dendrogram".

    Returns:
        matplotlib.figure.Figure: A static Matplotlib figure object containing the dendrogram tree.
    """
    logger.info(f"Generating dendrogram using {method} linkage and {metric} distance...")
    
    # Linear compression (Sparse to 50D Dense) to compute the linkage matrix
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_dense = svd.fit_transform(X_matrix)
    
    linkage_matrix = sch.linkage(X_dense, method=method, metric=metric)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title(title)
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel(f"Distance ({method.capitalize()} Linkage / {metric.capitalize()})")
    
    sch.dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=20,
        leaf_rotation=45.,
        leaf_font_size=12.,
        show_contracted=True,
        ax=ax
    )
    plt.tight_layout()
    
    return fig