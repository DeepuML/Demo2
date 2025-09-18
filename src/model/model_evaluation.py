# Model Building Script

# This script handles the model training phase of a machine 
# learning pipeline. It loads the processed data, retrieves 
# model parameters from a YAML configuration file, trains a 
# Gradient Boosting Classifier, and saves the trained model 
# for later use in evaluation or deployment.


import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging
import os 



# Logging Configuration
# 
# We use Python's logging module to track the execution flow.
# DEBUG logs help in development & debugging, while ERROR logs
# are saved into a file for error traceability.

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Console handler -> for displaying logs in the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler -> logs errors into 'model_building_errors.log'
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

# Formatting logs to include timestamp, module, and message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load Parameters

def load_params(params_path: str) -> dict:
    """
    Load model parameters from a YAML configuration file.
    
    Args:
        params_path (str): Path to the params.yaml file.

    Returns:
        dict: Dictionary of parameters.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error while reading parameters: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# Load Data

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed TF-IDF data from a CSV file.
    
    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


# Train Model

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting Classifier model.
    
    Args:
        X_train (np.ndarray): Training feature vectors.
        y_train (np.ndarray): Training labels.
        params (dict): Model hyperparameters (n_estimators, learning_rate).

    Returns:
        GradientBoostingClassifier: Trained classifier.
    """
    try:
        # Initialize model with parameters from YAML
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        # Train model
        clf.fit(X_train, y_train)
        logger.debug('Model training completed successfully')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


# Save Model

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a pickle file.
    
    Args:
        model: Trained ML model.
        file_path (str): File path to save the model.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error while saving model: %s', e)
        raise

    

# Main Function

def main():
    """
    Main execution function for model building.
    Steps:
    1. Load model parameters from params.yaml.
    2. Load processed TF-IDF training data.
    3. Train a Gradient Boosting Classifier.
    4. Save the trained model to the models/ directory.
    """
    try:
        # Load hyperparameters for model building
        params = load_params('params.yaml')['model_building']

        # Load training data (features & labels)
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values  # all columns except last → features
        y_train = np.array(train_data.iloc[:, -1].values)   # last column → labels

        # Train Gradient Boosting model
        clf = train_model(X_train, y_train, params)
        
        # Save trained model
        save_model(clf, 'models/model.pkl')

        # Evaluate model on test set and save metrics
        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = np.array(test_data.iloc[:, -1].values)
        y_pred = clf.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        from src.model.metrics_utils import save_metrics
        os.makedirs('reports', exist_ok=True)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Model building process failed: %s', e)
        print(f"Error: {e}")


# Script Execution Entry Point

if __name__ == '__main__':
    main()
