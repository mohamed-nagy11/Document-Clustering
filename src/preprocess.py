import re
import contractions
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

# 'parser' and 'ner' are disabled to speed up processing since we only need POS/lemmas.
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    logger.info("spaCy 'en_core_web_sm' model loaded successfully.")
except OSError:
    logger.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
    raise


class TextCleaner(BaseEstimator, TransformerMixin):
    """Cleans raw text by removing URLs, contractions, and special characters."""
    
    def fit(self, X: pd.Series, y=None) -> 'TextCleaner':
        """Fits the transformer (no-op).

        Args:
            X (pd.Series): The input text data.
            y (None, optional): Ignored.

        Returns:
            TextCleaner: The fitted transformer instance.
        """
        return self
        
    def transform(self, X: pd.Series, y=None) -> pd.Series:
        """Applies text cleaning transformations.

        Args:
            X (pd.Series): The input text data.
            y (None, optional): Ignored.

        Returns:
            pd.Series: A Pandas Series containing the cleaned text.
        """
        clean_docs = []
        for text in X:
            # Normalization (Lowercasing)
            text = str(text).lower()

            # Remove newlines and tabs
            text = text.replace('\n', ' ').replace('\t', ' ')
            
            # Remove URLs
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Expand contractions
            text = contractions.fix(text)
            
            # Remove numbers and punctuations (Keep letters and spaces)
            text = re.sub(r'[^a-z\s]', '', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            clean_docs.append(text)
        
        # Return a Pandas Series preserving the original index
        if isinstance(X, pd.Series):
            return pd.Series(clean_docs, index=X.index)
        return pd.Series(clean_docs)


class LinguisticProcessor(BaseEstimator, TransformerMixin):
    """Processes text by tokenizing, lemmatizing, and removing stopwords."""
    
    def fit(self, X: pd.Series, y=None) -> 'LinguisticProcessor':
        """Fits the transformer (no-op).

        Args:
            X (pd.Series): The input text data.
            y (None, optional): Ignored.

        Returns:
            LinguisticProcessor: The fitted transformer instance.
        """
        return self
        
    def transform(self, X: pd.Series, y=None) -> pd.Series:
        """Applies linguistic transformations.

        Args:
            X (pd.Series): The input text data.
            y (None, optional): Ignored.

        Returns:
            pd.Series: A Pandas Series containing the processed text.
        """
        processed_docs = []
        
        for doc in nlp.pipe(X, batch_size=200):
            # Extract lemma if it's not a stopword and length > 1
            clean_words = [
                token.lemma_ for token in doc 
                if not token.is_stop and token.is_alpha and len(token.text) > 1
            ]
            processed_docs.append(' '.join(clean_words))
            
        if isinstance(X, pd.Series):
            return pd.Series(processed_docs, index=X.index)
        return pd.Series(processed_docs)