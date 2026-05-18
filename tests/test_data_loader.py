import pytest
from src.data_loader import load_wikipedia_data, load_newsgroups_data

def test_wikipedia_data_loads_correctly():
    """Tests if the Wikipedia dataset loads and contains the required columns."""
    df_wiki = load_wikipedia_data()
    assert df_wiki is not None, "Wikipedia DataFrame should not be None"
    assert 'text' in df_wiki.columns, "Wikipedia DataFrame is missing the 'text' column"
    assert len(df_wiki) > 0, "Wikipedia DataFrame is empty"

def test_newsgroups_data_loads_correctly():
    """Tests if the Newsgroups dataset loads and contains the required columns."""
    df_news = load_newsgroups_data()
    assert df_news is not None, "Newsgroups DataFrame should not be None"
    assert 'text' in df_news.columns, "Newsgroups DataFrame is missing the 'text' column"
    assert 'target' in df_news.columns, "Newsgroups DataFrame is missing the 'target' column"
    assert len(df_news) > 0, "Newsgroups DataFrame is empty"