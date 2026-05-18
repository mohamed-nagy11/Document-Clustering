import os
import shutil
import pandas as pd
import kagglehub
from sklearn.datasets import fetch_20newsgroups

import src.config as config
from src.logger import get_logger

logger = get_logger(__name__)

def download_wikipedia_from_kaggle():
    """
    Downloads the Wikipedia dataset using the modern kagglehub library.
    
    Raises:
        Exception: If the dataset failed to download.
    """
    if config.WIKI_CSV_PATH.exists():
        logger.info("Wikipedia dataset already exists locally. Skipping download.")
        return

    logger.info("Wikipedia dataset not found. Downloading via kagglehub...")
    try:
        cache_dir = kagglehub.dataset_download("sameersmahajan/people-wikipedia-data")
        cached_file_path = os.path.join(cache_dir, "people_wiki.csv")
        
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file_path, config.WIKI_CSV_PATH)
        
        logger.info(f"Successfully downloaded and moved data to {config.WIKI_CSV_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to download using kagglehub. Error: {e}")
        raise

def load_wikipedia_data(filepath: str = config.WIKI_CSV_PATH) -> pd.DataFrame:
    """
    Loads the Wikipedia biographies dataset.

    Args:
        filepath (str, optional): The path to the Wikipedia CSV file. 
            Defaults to config.WIKI_CSV_PATH.

    Returns:
        pd.DataFrame: A DataFrame containing the Wikipedia data.

    Raises:
        FileNotFoundError: If the CSV file cannot be found after download attempt.
    """
    # Ensure the data exists before trying to read it
    download_wikipedia_from_kaggle()
    
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