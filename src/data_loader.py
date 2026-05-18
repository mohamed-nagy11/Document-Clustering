import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

def load_wikipedia_data(filepath: str = config.WIKI_CSV_PATH) -> pd.DataFrame:
    """
    Loads the Wikipedia biographies dataset.

    Args:
        filepath (str): The path to the Wikipedia data.

    Returns:
        pd.DataFrame: The Wikipedia dataframe.
    
    Raises:
        FileNotFoundError: If the Wikipedia data is not found.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Wikipedia data loaded successfully: {df.shape[0]} rows.")
        return df
    except FileNotFoundError:
        logger.error(f"Wikipedia data not found at {filepath}.")
        raise FileNotFoundError(f"Wikipedia data not found at {filepath}.")

def load_newsgroups_data() -> pd.DataFrame:
    """
    Fetches the 20 Newsgroups dataset and strips non-body text.

    Removes headers, footers, and quotes from the raw forum posts to 
    ensure only conversational text is passed to the pipeline.

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'target' columns.
    """
    logger.info("Fetching 20 Newsgroups dataset.")
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        random_state=config.RANDOM_SEED
    )
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target
    })
    logger.info(f"20 Newsgroups data loaded successfully: {df.shape[0]} rows.")
    return df
    