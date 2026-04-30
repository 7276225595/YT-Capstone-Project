import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vaibhavlokhande1798"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/vaibhavlokhande1798/YT-Capstone-Project.mlflow')
dagshub.init(repo_owner='vaibhavlokhande1798', repo_name='YT-Capstone-Project', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry with alias."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Create client
        client = mlflow.tracking.MlflowClient()
        
        # Add alias to the model version (NEW API - fixes INTERNAL_ERROR)
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=str(model_version.version)  # Must be string!
        )
        
        # Optional: Add description
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Model registered from run {model_info['run_id']}"
        )
        
        logging.debug(f'Model {model_name} version {model_version.version} registered with alias "Production".')
        print(f"✅ Model registered successfully! Version: {model_version.version}, Alias: Production")
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def load_model_by_alias(model_name: str, alias: str = "Production"):
    """Load a model using its alias."""
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        logging.debug(f'Model {model_name} loaded with alias {alias}')
        print(f"✅ Model loaded successfully using alias '{alias}'")
        return model
    except Exception as e:
        logging.error('Error loading model by alias: %s', e)
        raise


def list_model_aliases(model_name: str):
    """List all aliases for a model."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        aliases = {}
        for version in model_versions:
            if version.aliases:
                aliases[version.version] = version.aliases
        return aliases
    except Exception as e:
        logging.error('Error listing model aliases: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        
        # Register model with alias
        register_model(model_name, model_info)
        
        # List all aliases to verify
        aliases = list_model_aliases(model_name)
        print(f"Aliases for {model_name}: {aliases}")
        
        # Load the model back to verify
        model = load_model_by_alias(model_name, "Production")
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()