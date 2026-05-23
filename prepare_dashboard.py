import mlflow
import pickle
import json
import shutil
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Document_Clustering_Project"
ASSETS_DIR = Path("dashboard/assets")

def auto_prepare():
    """Extracts MLflow artifacts and metadata to populate the Dash assets directory."""
    logger.info("Initializing automated dashboard asset preparation...")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        logger.error(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow database.")
        return

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    
    metadata = {}
    logger.info(f"Found {len(runs)} total runs in MLflow. Processing metadata and artifacts...")

    for run in runs:
        params = run.data.params
        dataset = params.get("dataset")
        model_name = params.get("model")
        vectorizer_name = params.get("vectorizer")
        
        if not all([dataset, model_name, vectorizer_name]):
            continue

        run_key = f"{dataset}_{model_name}_{vectorizer_name}"
        metadata[run_key] = {"keywords": {}}

        # Extract Keywords (Only from TF-IDF runs for speed and vocabulary access)
        if vectorizer_name == "tfidf" and model_name != "hierarchical":
            try:
                logger.info(f"Extracting TF-IDF keywords for {run_key}...")
                
                vec_path = client.download_artifacts(run.info.run_id, "vectorizer/model.pkl")
                mod_path = client.download_artifacts(run.info.run_id, "clustering_model/model.pkl")
                
                with open(vec_path, "rb") as f: vec = pickle.load(f)
                with open(mod_path, "rb") as f: mod = pickle.load(f)
                
                # Grab the vocabulary from the internal tfidf object
                vocab = vec.tfidf.get_feature_names_out()
                
                # Get the 50-dimensional SVD centroids
                centers_lsa = mod.cluster_centers_ if hasattr(mod, 'cluster_centers_') else mod.means_
                
                # Project the centroids backwards into the original word space!
                centers_tfidf = vec.svd.inverse_transform(centers_lsa)
                
                # Map the highest weighted words
                for i, center in enumerate(centers_tfidf):
                    top_idx = center.argsort()[-5:][::-1]
                    metadata[run_key]["keywords"][f"Cluster {i}"] = ", ".join([vocab[idx] for idx in top_idx])
                    
            except Exception as e:
                logger.warning(f"Failed to extract keywords for {run_key}: {e}")

        artifact_uri = run.info.artifact_uri.replace("file:///", "")
        viz_dir = Path(artifact_uri) / "visualizations"
        
        if viz_dir.exists():
            for viz_file in viz_dir.iterdir():
                ext = viz_file.suffix
                dest_name = f"{run_key}{ext}"
                try:
                    shutil.copy(viz_file, ASSETS_DIR / dest_name)
                    logger.info(f"Successfully copied visual artifact: {dest_name}")
                except Exception as e:
                    logger.error(f"Failed to copy visual artifact {dest_name}: {e}")
        else:
            logger.warning(f"No visualizations folder found for run: {run_key}")

    metadata_path = ASSETS_DIR / "metadata.json"
    try:
        with open(metadata_path, "wb") as f:
            f.write(json.dumps(metadata, indent=4).encode("utf-8"))
        logger.info(f"Successfully saved automated cluster labels to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata.json: {e}")
        
    logger.info("--- Dashboard Asset Preparation Complete ---")

if __name__ == "__main__":
    auto_prepare()