import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

class TfidfLsaVectorizer(BaseEstimator, TransformerMixin):
    """Converts text to TF-IDF features and reduces dimensionality using LSA.
    
    Attributes:
        max_features (int): Maximum vocabulary size for TF-IDF.
        n_components (int): Target dimensionality for TruncatedSVD.
    """
    
    def __init__(self, max_features: int = config.TFIDF_MAX_FEATURES, n_components: int = 50):
        self.max_features = max_features
        self.n_components = n_components
        
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features, 
            min_df=config.TFIDF_MIN_DF, 
            max_df=config.TFIDF_MAX_DF
        )
        self.svd = TruncatedSVD(
            n_components=self.n_components, 
            random_state=config.RANDOM_SEED
        )
        
    def fit(self, X: pd.Series, y=None) -> 'TfidfLsaVectorizer':
        """Fits the TF-IDF and SVD models to the text data.

        Args:
            X (pd.Series): The preprocessed text data.
            y (None, optional): Ignored.

        Returns:
            TfidfLsaVectorizer: The fitted transformer instance.
        """
        logger.info(f"Fitting TF-IDF (max_features={self.max_features}) and LSA (n_components={self.n_components})...")
        tfidf_matrix = self.tfidf.fit_transform(X)
        self.svd.fit(tfidf_matrix)
        return self
        
    def transform(self, X: pd.Series, y=None) -> np.ndarray:
        """Transforms text into dense LSA embeddings.

        Args:
            X (pd.Series): The preprocessed text data.
            y (None, optional): Ignored.

        Returns:
            np.ndarray: A dense numpy array of shape (n_samples, n_components).
        """
        tfidf_matrix = self.tfidf.transform(X)
        dense_matrix = self.svd.transform(tfidf_matrix)
        return dense_matrix


class TransformerVectorizer(BaseEstimator, TransformerMixin):
    """Converts text to dense semantic embeddings using a pre-trained Transformer.
    
    Utilizes Hugging Face's sentence-transformers with lazy-loading.
    
    Attributes:
        model_name (str): The Hugging Face model string.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None  # Lazy loading flag
        
    def fit(self, X: pd.Series, y=None) -> 'TransformerVectorizer':
        """Fits the transformer (no-op as the model is pre-trained).

        Args:
            X (pd.Series): The preprocessed text data.
            y (None, optional): Ignored.

        Returns:
            TransformerVectorizer: The transformer instance.
        """
        return self
        
    def transform(self, X: pd.Series, y=None) -> np.ndarray:
        """Transforms text into deep semantic embeddings.

        Args:
            X (pd.Series): The preprocessed text data.
            y (None, optional): Ignored.

        Returns:
            np.ndarray: A dense numpy array of embeddings.
        """
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
        logger.info("Encoding text with SentenceTransformer...")
        
        # SentenceTransformers expects a list of strings
        text_list = X.tolist() if isinstance(X, pd.Series) else list(X)
        
        embeddings = self.model.encode(text_list, show_progress_bar=False)
        return embeddings