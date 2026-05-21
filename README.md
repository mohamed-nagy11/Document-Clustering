# Unsupervised Document Clustering Pipeline

This repository contains an end-to-end, production-grade unsupervised machine learning pipeline designed to cluster high-dimensional text data. The project leverages an object-oriented, modular architecture to systematically process, vectorize, and cluster documents from the 20 Newsgroups and Wikipedia Biographies datasets.

The core pipeline supports multiple vectorization strategies (statistical and semantic) and clustering algorithms, all orchestrated and tracked via MLflow.

## Project Architecture

The system is built with strict separation of concerns, ensuring high maintainability and scalability:

* **Ingestion:** Robust data loaders for fetching and standardizing the Newsgroups and Wikipedia corpora.
* **Preprocessing:** A multi-step linguistic engine utilizing `spaCy` (`en_core_web_sm`) for context-aware lemmatization, stop-word removal, and noise reduction (stripping HTML, URLs, and numeric artifacts).
* **Feature Engineering:** Provides two distinct vectorization pathways:
* *Sparse/Statistical:* `TfidfLsaVectorizer` (TF-IDF followed by Truncated SVD/LSA dimensionality reduction).
* *Dense/Semantic:* `TransformerVectorizer` utilizing Hugging Face's Sentence Transformers (`all-MiniLM-L6-v2`) for deep semantic embeddings.


* **Clustering Models:** Configurable implementations of K-Means, Gaussian Mixture Models (GMM), and Agglomerative Hierarchical Clustering.
* **Observability:** Fully integrated MLflow tracking with SQLite persistence for logging hyperparameters, Silhouette scores, and visual artifacts (2D t-SNE maps and dendrograms).

## Repository Structure

```text
├── notebooks/
│   └── playground.ipynb        # Exploratory Data Analysis, k-optimization, and grid searches
├── src/
│   ├── cluster.py              # Clustering algorithm initializers
│   ├── config.py               # Global configurations and optimal hyperparameter dictionaries
│   ├── data_loader.py          # Dataset fetching and parsing
│   ├── evaluate.py             # Metric calculations (Silhouette scoring)
│   ├── features.py             # Custom Scikit-Learn transformers for TF-IDF and SBERT
│   ├── logger.py               # Standardized logging configuration
│   ├── preprocess.py           # Text cleaning and linguistic processing classes
│   ├── train.py                # Main orchestration script and MLflow integration
│   └── visualize.py            # t-SNE projections and dendrogram generation
├── tests/                      # Unit tests for core pipeline modules
├── requirements.txt            # Project dependencies
└── README.md

```

## Setup and Installation

1. **Clone the repository and navigate to the root directory.**
2. **Create and activate a virtual environment.**
3. **Install the required dependencies:**
```bash
pip install -r requirements.txt

```


## Usage

The pipeline is executed via the `src/train.py` script. It accepts command-line arguments to dictate which dataset, model, and vectorizer to use for a given run.

**Command Syntax:**

```bash
python -m src.train --dataset [newsgroups|wikipedia] --model [kmeans|gmm|hierarchical] --vectorizer [tfidf|transformer]

```

**Execution Examples:**

To run K-Means clustering on the Newsgroups dataset using TF-IDF features:

```bash
python -m src.train --dataset newsgroups --model kmeans --vectorizer tfidf

```

To run Hierarchical clustering on the Wikipedia dataset using Transformer embeddings:

```bash
python -m src.train --dataset wikipedia --model hierarchical --vectorizer transformer

```

*Note: The pipeline includes a dynamic guardrail that automatically detects and drops "zero-vectors" (documents reduced to empty strings during preprocessing) to prevent distance calculation errors during training.*

## Experiment Tracking (MLflow)

Every pipeline execution automatically logs its configuration, metrics, and generated artifacts to a local database.

To view the results, start the MLflow UI from the root directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db

```

Navigate to `http://127.0.0.1:5000` in your browser. Under the **Document_Clustering_Project** experiment, you can review the Silhouette scores, compare runs, and access the generated HTML t-SNE maps and PNG dendrograms stored in the artifacts section.

---