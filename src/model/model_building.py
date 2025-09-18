# # model_building.py
# # This script handles model training for the ML pipeline.
# # It loads hyperparameters from params.yaml, reads processed training data,
# # trains a Gradient Boosting model, and saves the trained model as a pickle file.

# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.ensemble import GradientBoostingClassifier
# import yaml
# import logging


# # Logging configuration

# logger = logging.getLogger('model_building')
# logger.setLevel('DEBUG')

# # Console logs (for debugging)
# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# # File logs (only errors go into this file)
# file_handler = logging.FileHandler('model_building_errors.log')
# file_handler.setLevel('ERROR')

# # Format for logs
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# # Attach handlers to logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


# # Utility Functions

# def load_params(params_path: str) -> dict:
#     """
#     Load parameters from a YAML file.

#     Args:
#         params_path (str): Path to params.yaml file.

#     Returns:
#         dict: Dictionary containing parameters.
#     """
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error: %s', e)
#         raise


# def load_data(file_path: str) -> pd.DataFrame:
#     """
#     Load data from a CSV file.

#     Args:
#         file_path (str): Path to CSV file.

#     Returns:
#         pd.DataFrame: Loaded dataset as a DataFrame.
#     """
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise


# def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
#     """
#     Train the Gradient Boosting model.

#     Args:
#         X_train (np.ndarray): Training features.
#         y_train (np.ndarray): Training labels.
#         params (dict): Model hyperparameters.

#     Returns:
#         GradientBoostingClassifier: Trained model.
#     """
#     try:
#         clf = GradientBoostingClassifier(
#             n_estimators=params['n_estimators'],
#             learning_rate=params['learning_rate']
#         )
#         clf.fit(X_train, y_train)
#         logger.debug('Model training completed')
#         return clf
#     except Exception as e:
#         logger.error('Error during model training: %s', e)
#         raise


# def save_model(model, file_path: str) -> None:
#     """
#     Save the trained model to a pickle file.

#     Args:
#         model: Trained machine learning model.
#         file_path (str): Destination path for saving the model.
#     """
#     try:
#         with open(file_path, 'wb') as file:
#             pickle.dump(model, file)
#         logger.debug('Model saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model: %s', e)
#         raise


# # Main Pipeline Step

# def main():
#     """Main function to execute the model building pipeline step."""
#     try:
#         # Load model hyperparameters from params.yaml
#         params = load_params('params.yaml')['model_building']

#         # Load processed training data
#         train_data = load_data('./data/processed/train_tfidf.csv')
#         X_train = train_data.iloc[:, :-1].values  # Features
#         y_train = train_data.iloc[:, -1].values   # Labels

#         # Train the model
#         clf = train_model(X_train, y_train, params)
        
#         # Save the trained model
#         save_model(clf, 'models/model.pkl')
#     except Exception as e:
#         logger.error('Failed to complete the model building process: %s', e)
#         print(f"Error: {e}")


# if __name__ == '__main__':
#     main()

# model_building.py
# This script handles model training for the ML pipeline.
# It loads hyperparameters from params.yaml, reads processed training data
# (Bag of Words features), trains a Gradient Boosting model, and saves the trained model.

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging


# ------------------ Logging Configuration ------------------

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Console logs (for debugging)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File logs (only errors go into this file)
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

# Format for logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------ Utility Functions ------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train the Gradient Boosting model using Bag of Words features."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        logger.debug('Model training completed successfully')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a pickle file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


# ------------------ Main Pipeline Step ------------------

def main():
    """Main function to execute the model building pipeline step."""
    try:
        # Load model hyperparameters from params.yaml
        params = load_params('params.yaml')['model_building']

        # Load processed training data (from Bag of Words features)
        train_data = load_data('./data/processed/train_bow.csv')  # 👈 Changed from train_tfidf.csv to train_bow.csv
        X_train = train_data.iloc[:, :-1].values  # Features
        y_train = train_data.iloc[:, -1].values   # Labels

        # Train the model
        clf = train_model(X_train, y_train, params)
        
        # Save the trained model
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
