# Unsupervised Document Clustering & Analysis Pipeline


An end-to-end, production-grade unsupervised machine learning pipeline designed to cluster high-dimensional text data. This project systematically processes, vectorizes, and clusters documents from the **20 Newsgroups** and **People Wikipedia** datasets, wrapped in a beautiful, interactive Streamlit dashboard.

---

## Live Demo

Experience the interactive dashboard deployed on Hugging Face Spaces:
[Document Clustering Dashboard](https://huggingface.co/spaces/mohamednagy11/Document_Clustering)
---

## Highlighted Skills & What I Built

This project showcases a complete lifecycle of an ML project, from data ingestion to containerized deployment.

*   **Machine Learning & NLP:** Designed and implemented multiple vectorization pipelines, comparing statistical methods (**TF-IDF + Truncated SVD/LSA**) against deep semantic embeddings (**Hugging Face Sentence Transformers `all-MiniLM-L6-v2`**). Applied and tuned various clustering algorithms including **K-Means**, **Gaussian Mixture Models (GMM)**, and **Hierarchical Clustering**.
*   **Software Engineering (SWE):** Ensured strict separation of concerns, modularity, and high maintainability across data loading, text preprocessing, feature engineering, and model training.
*   **MLOps & Experiment Tracking:** Integrated **MLflow** with SQLite persistence to automatically log hyperparameters, metric scores (Silhouette score), and visual artifacts (interactive t-SNE maps and dendrograms) for every pipeline run.
*   **Data Engineering:** Built robust data loaders for the Newsgroups and Wikipedia corpora, alongside a sophisticated multi-step linguistic preprocessing engine using `spaCy` to handle lemmatization, stop-word removal, and noise reduction.
*   **Deployment & Containerization:** Developed an interactive **Streamlit** dashboard to visualize the clustering results and interactively compare model performances. Containerized the entire application using **Docker** for seamless deployment to Hugging Face Spaces.

---

## Technologies Used

*   **Language:** Python
*   **Machine Learning:** Scikit-Learn, NumPy, Pandas, SciPy
*   **Natural Language Processing (NLP):** spaCy, Sentence Transformers (Hugging Face)
*   **MLOps & Tracking:** MLflow
*   **Dashboard & Visualization:** Streamlit, Plotly, Matplotlib
*   **Deployment:** Docker, Hugging Face Spaces

---

## Project Architecture

The system is built with strict separation of concerns, ensuring high maintainability and scalability:

*   **Ingestion:** Robust data loaders for fetching and standardizing the Newsgroups and Wikipedia corpora.
*   **Preprocessing:** A multi-step linguistic engine utilizing `spaCy` (`en_core_web_sm`) for context-aware lemmatization, stop-word removal, and noise reduction (stripping HTML, URLs, and numeric artifacts).
*   **Feature Engineering:** Provides two distinct vectorization pathways:
    *   *Sparse/Statistical:* `TfidfLsaVectorizer` (TF-IDF followed by Truncated SVD/LSA dimensionality reduction).
    *   *Dense/Semantic:* `TransformerVectorizer` utilizing Hugging Face's Sentence Transformers (`all-MiniLM-L6-v2`) for deep semantic embeddings.
*   **Clustering Models:** Configurable implementations of K-Means, Gaussian Mixture Models (GMM), and Agglomerative Hierarchical Clustering.
*   **Observability:** Fully integrated MLflow tracking for logging hyperparameters, Silhouette scores, and visual artifacts (2D t-SNE maps and dendrograms).

## Repository Structure

```text
├── dashboard/                  # Streamlit dashboard and deployment assets
│   ├── app.py                  # Main Streamlit application
│   ├── assets/                 # Pre-generated visualization assets & metadata
│   └── requirements.txt        # Dashboard specific dependencies
├── notebooks/
│   └── playground.ipynb        # Exploratory Data Analysis, k-optimization, and grid searches
├── src/
│   ├── cluster.py              # Clustering algorithm initializers
│   ├── config.py               # Global configurations and optimal hyperparameter dictionaries
│   ├── data_loader.py          # Dataset fetching and parsing
│   ├── evaluate.py             # Metric calculations (Silhouette scoring)
│   ├── features.py             # Custom Scikit-Learn transformers for TF-IDF and SentenceTransformer
│   ├── logger.py               # Standardized logging configuration
│   ├── preprocess.py           # Text cleaning and linguistic processing classes
│   ├── train.py                # Main orchestration script and MLflow integration
│   └── visualize.py            # t-SNE projections and dendrogram generation
├── tests/                      # Unit tests for core pipeline modules
├── requirements.txt            # Project dependencies
├── Dockerfile                  # Containerization for deployment
└── README.md                   # Project documentation
```

---

## Setup and Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mohamed-nagy11/Document-Clustering.git
cd Document-Clustering
```

2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Training the Pipeline
The pipeline is executed via the `src/train.py` script. It accepts command-line arguments to dictate which dataset, model, and vectorizer to use for a given run.

**Command Syntax:**
```bash
python -m src.train --dataset [newsgroups|wikipedia] --model [kmeans|gmm|hierarchical] --vectorizer [tfidf|transformer]
```

**Execution Examples:**
```bash
# Run K-Means clustering on the Newsgroups dataset using TF-IDF features
python -m src.train --dataset newsgroups --model kmeans --vectorizer tfidf

# Run Hierarchical clustering on the Wikipedia dataset using Transformer embeddings
python -m src.train --dataset wikipedia --model hierarchical --vectorizer transformer
```

### 2. Experiment Tracking (MLflow)
Every pipeline execution automatically logs its configuration, metrics, and generated artifacts to a local database. To view the results locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` in your browser.

### 3. Running the Dashboard Locally
You can run the interactive Streamlit dashboard locally to visualize the clustering results:

```bash
cd dashboard
streamlit run app.py
```
Navigate to `http://localhost:8501` to interact with the visualizer.