import argparse
import time
import mlflow
import mlflow.sklearn

import src.config as config
from src.logger import get_logger
from src.data_loader import load_wikipedia_data, load_newsgroups_data
from src.preprocess import TextCleaner, LinguisticProcessor
from src.features import TfidfLsaVectorizer, TransformerVectorizer
from src.cluster import get_kmeans, get_gmm, get_hierarchical
from src.evaluate import evaluate_clusters

logger = get_logger(__name__)

def run_pipeline(dataset_name: str, model_name: str, vectorizer_name: str):
    """Executes the end-to-end clustering pipeline and logs to MLflow."""
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Document_Clustering_Project")
    
    with mlflow.start_run(run_name=f"{dataset_name}_{model_name}_{vectorizer_name}"):
        
        logger.info(f"=== Starting Pipeline: {dataset_name} | {model_name} | {vectorizer_name} ===")
        
        mlflow.log_params({
            "dataset": dataset_name,
            "model": model_name,
            "vectorizer": vectorizer_name,
            "random_seed": config.RANDOM_SEED
        })

        if dataset_name == "newsgroups":
            df = load_newsgroups_data()
            best_params = config.NEWSGROUPS_PARAMS
        else:
            df = load_wikipedia_data()
            best_params = config.WIKI_PARAMS
            
        # For testing the pipeline quickly
        # df = df.head(2000)

        logger.info("Starting text cleaning...")
        cleaner = TextCleaner()
        clean_text = cleaner.transform(df['text'])
        
        logger.info("Starting linguistic processing...")
        processor = LinguisticProcessor()
        processed_text = processor.transform(clean_text)

        if vectorizer_name == "tfidf":
            vectorizer = TfidfLsaVectorizer()
        else:
            vectorizer = TransformerVectorizer()
            
        embeddings = vectorizer.fit_transform(processed_text)

        model_kwargs = best_params.get(model_name, {})
        mlflow.log_params(model_kwargs)
        
        if model_name == "kmeans":
            model = get_kmeans(**model_kwargs)
            labels = model.fit_predict(embeddings)
        elif model_name == "gmm":
            model = get_gmm(**model_kwargs)
            labels = model.fit_predict(embeddings)
        elif model_name == "hierarchical":
            model = get_hierarchical(**model_kwargs)
            labels = model.fit_predict(embeddings)

        logger.info("Evaluating clusters...")
        metrics = evaluate_clusters(embeddings, labels)
        mlflow.log_metrics(metrics)
        
        if vectorizer_name == "tfidf":
            mlflow.sklearn.log_model(vectorizer, "vectorizer")
        if model_name in ["kmeans", "gmm"]:
            mlflow.sklearn.log_model(model, "clustering_model")

        logger.info("=== Pipeline Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Document Clustering Pipeline")
    parser.add_argument("--dataset", type=str, choices=["newsgroups", "wikipedia"], required=True)
    parser.add_argument("--model", type=str, choices=["kmeans", "gmm", "hierarchical"], required=True)
    parser.add_argument("--vectorizer", type=str, choices=["tfidf", "transformer"], default="tfidf")
    
    args = parser.parse_args()
    
    start_time = time.time()
    run_pipeline(args.dataset, args.model, args.vectorizer)
    logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")