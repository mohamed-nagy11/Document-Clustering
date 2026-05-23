import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import mlflow

# Page Configuration & Setup
st.set_page_config(page_title="Document Clustering Dashboard", layout="wide")

# Hide the buggy Streamlit "Running" status indicator
st.markdown(
    """
    <style>
    [data-testid="stStatusWidget"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #0d6efd;'>Document Clustering Analysis</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #a0a0a0; margin-bottom: 40px;'>An end-to-end pipeline for text clustering</p>", 
    unsafe_allow_html=True
)

# Load Assets & MLflow Metadata
ASSETS_DIR = Path(__file__).parent / "assets"
try:
    with open(ASSETS_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    metadata = {}

TRACKING_URI = "sqlite:///../mlflow.db" 
EXPERIMENT_NAME = "Document_Clustering_Project"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)

if exp:
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    df_scores = runs_df[['params.dataset', 'params.model', 'params.vectorizer', 'metrics.silhouette_score']].copy()
    df_scores.columns = ['Dataset', 'Algorithm', 'Vectorizer', 'Score']
    df_scores = df_scores.dropna()
else:
    df_scores = pd.DataFrame(columns=['Dataset', 'Algorithm', 'Vectorizer', 'Score'])

def get_score(dataset: str, algorithm: str, vectorizer: str):
    match = df_scores[(df_scores["Dataset"] == dataset) & 
                      (df_scores["Algorithm"] == algorithm) & 
                      (df_scores["Vectorizer"] == vectorizer)]
    if not match.empty:
        return round(match["Score"].values[0], 4)
    return "N/A"

# Sidebar Controls
st.sidebar.header("Pipeline Configurations")

dataset = st.sidebar.selectbox(
    "Dataset", 
    ["wikipedia", "newsgroups"], 
    format_func=lambda x: "20 Newsgroups" if x == "newsgroups" else "Wikipedia Biographies"
)

model = st.sidebar.selectbox(
    "Algorithm", 
    ["kmeans", "gmm", "hierarchical"], 
    format_func=lambda x: {"kmeans": "K-Means", "gmm": "Gaussian Mixture", "hierarchical": "Hierarchical"}[x]
)

vectorizer = st.sidebar.selectbox(
    "Vectorizer", 
    ["tfidf", "transformer"], 
    format_func=lambda x: "TF-IDF (Statistical)" if x == "tfidf" else "Transformer (Semantic)"
)

# Metric Card
run_key = f"{dataset}_{model}_{vectorizer}"
score = get_score(dataset, model, vectorizer)

st.markdown(f"""
<div style="background-color: #1e1e1e; padding: 20px; border-radius: 8px; border: 1px solid #333; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <p style="color: #a0a0a0; font-size: 16px; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Current Pipeline Silhouette Score</p>
    <h2 style="color: #0d6efd; font-size: 48px; margin: 10px 0 0 0;">{score}</h2>
</div>
""", unsafe_allow_html=True)

st.divider()

# Plotting Area
st.subheader("Cluster Visualization")
if model == "hierarchical":
    img_path = ASSETS_DIR / f"{run_key}.png"
    if img_path.exists():
        st.image(str(img_path), width=900)
else:
    html_path = ASSETS_DIR / f"{run_key}.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # We must use components.html so the Plotly JavaScript actually executes
        components.html(source_code, width=1100, height=750, scrolling=True)

st.divider()

# Keywords Area
st.subheader("Cluster Centroid Keywords")
run_meta = metadata.get(run_key, {}).get("keywords", {})

if run_meta:
    for cluster, words in run_meta.items():
        st.markdown(f"**{cluster}:** {words}")
else:
    if vectorizer == "transformer":
        st.info("Keyword extraction is unsupported for Transformer embeddings (dense semantic space without direct dictionary mapping).")
    elif model == "hierarchical":
        st.info("Keyword extraction is unsupported for Hierarchical clustering (no explicitly calculated mathematical centroids).")
    else:
        st.warning("No keyword data found for this run.")

st.divider()

# Model Comparison
st.subheader("Model Comparison")

if not df_scores.empty:
    clean_names = {"kmeans": "K-Means", "gmm": "GMM", "hierarchical": "Hierarchical", "tfidf": "TF-IDF", "transformer": "Transformer"}
    df_plot = df_scores.copy()
    df_plot["Algorithm"] = df_plot["Algorithm"].map(clean_names)
    df_plot["Vectorizer"] = df_plot["Vectorizer"].map(clean_names)

    fig_wiki = px.bar(
        df_plot[df_plot["Dataset"] == "wikipedia"], 
        x="Algorithm", y="Score", color="Vectorizer", barmode="group",
        title="Wikipedia Silhouette Scores", text_auto='.4f'
    )
    
    fig_news = px.bar(
        df_plot[df_plot["Dataset"] == "newsgroups"], 
        x="Algorithm", y="Score", color="Vectorizer", barmode="group",
        title="20 Newsgroups Silhouette Scores", text_auto='.4f'
    )

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.plotly_chart(fig_wiki, width="stretch")
    with plot_col2:
        st.plotly_chart(fig_news, width="stretch")