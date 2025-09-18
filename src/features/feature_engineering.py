# # Feature Engineering Script



# # Import required libraries
# import numpy as np                                # For numerical operations
# import pandas as pd                               # For handling CSV datasets and DataFrames
# import os                                         # For file system path handling
# from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF feature extraction
# import yaml                                       # For reading YAML configuration files
# import logging                                    # For logging debugging messages and errors


# # Logging Configuration Setup



# # Create logger instance with the name "feature_engineering"
# logger = logging.getLogger('feature_engineering')
# logger.setLevel('DEBUG')  # Capture all logs of level DEBUG and above

# # Console handler - sends logs to terminal
# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# # File handler - writes only ERROR logs to file
# file_handler = logging.FileHandler('feature_engineering_errors.log')
# file_handler.setLevel('ERROR')

# # Define consistent log message format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # Attach formatter to handlers
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# # Register handlers to logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)



# # Function: Load Parameters



# def load_params(params_path: str) -> dict:
#     """
#     Load configuration parameters from a YAML file.
    
#     Args:
#         params_path (str): Path to the params.yaml file.

#     Returns:
#         dict: Dictionary of configuration parameters.
#     """
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)  # Load YAML contents
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



# # Function: Load Data


# def load_data(file_path: str) -> pd.DataFrame:
#     """
#     Load dataset from CSV and handle missing values.
    
#     Args:
#         file_path (str): Path to the CSV file.

#     Returns:
#         pd.DataFrame: Loaded DataFrame with NaNs replaced by empty strings.
#     """
#     try:
#         df = pd.read_csv(file_path)     # Load CSV file into DataFrame
#         df.fillna('', inplace=True)     # Replace NaN values with empty strings to avoid errors
#         logger.debug('Data loaded and NaNs filled from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise


# # Function: Apply Bag of Words (TF-IDF)


# def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
#     """
#     Apply TF-IDF vectorization to transform text into numerical features.
    
#     Args:
#         train_data (pd.DataFrame): Training dataset containing text and sentiment labels.
#         test_data (pd.DataFrame): Test dataset containing text and sentiment labels.
#         max_features (int): Maximum number of features (vocabulary size) to keep.

#     Returns:
#         tuple: Transformed (train_df, test_df) DataFrames with TF-IDF features + labels.
#     """
#     try:
#         # Initialize the TF-IDF Vectorizer with a limited vocabulary size
#         vectorizer = TfidfVectorizer(max_features=max_features)

#         # Extract features (content column) and labels (sentiment column) from train/test sets
#         X_train = train_data['content'].values
#         y_train = train_data['sentiment'].values
#         X_test = test_data['content'].values
#         y_test = test_data['sentiment'].values

#         # Fit TF-IDF on training data and transform both train and test sets
#         X_train_bow = vectorizer.fit_transform(X_train)
#         X_test_bow = vectorizer.transform(X_test)

#         # Convert sparse matrices into dense DataFrames for saving/processing
#         train_df = pd.DataFrame(X_train_bow.toarray())
#         train_df['label'] = y_train   # Append sentiment labels to training features

#         test_df = pd.DataFrame(X_test_bow.toarray())
#         test_df['label'] = y_test     # Append sentiment labels to testing features

#         logger.debug('Bag of Words applied and data transformed')
#         return train_df, test_df
#     except Exception as e:
#         logger.error('Error during Bag of Words transformation: %s', e)
#         raise


# # Function: Save Data


# def save_data(df: pd.DataFrame, file_path: str) -> None:
#     """
#     Save DataFrame into CSV file at the specified path.
    
#     Args:
#         df (pd.DataFrame): DataFrame to save.
#         file_path (str): Destination path for the CSV file.
#     """
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure folder exists
#         df.to_csv(file_path, index=False)  # Save DataFrame without index column
#         logger.debug('Data saved to %s', file_path)
#     except Exception as e:
#         logger.error('Unexpected error occurred while saving the data: %s', e)
#         raise



# # Main Execution Function


# def main():
#     """
#     Main function to run feature engineering pipeline:
#     1. Load parameters from params.yaml
#     2. Load preprocessed train/test data
#     3. Apply TF-IDF vectorization
#     4. Save transformed data into ./data/processed
#     """
#     try:
#         # Step 1: Load parameters (like max_features for TF-IDF)
#         params = load_params('params.yaml')
#         max_features = params['feature_engineering']['max_features']

#         # Step 2: Load preprocessed data (from previous stage)
#         train_data = load_data('./data/interim/train_processed.csv')
#         test_data = load_data('./data/interim/test_processed.csv')

#         # Step 3: Apply TF-IDF transformation
#         train_df, test_df = apply_bow(train_data, test_data, max_features)

#         # Step 4: Save the final feature-engineered data
#         save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
#         save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
#     except Exception as e:
#         logger.error('Failed to complete the feature engineering process: %s', e)
#         print(f"Error: {e}")


# # Script Entry Point

# if __name__ == '__main__':
#     main()  # Run pipeline when script is executed

# Feature Engineering Script (Bag of Words)

# Import required libraries
import numpy as np                                # For numerical operations
import pandas as pd                               # For handling CSV datasets and DataFrames
import os                                         # For file system path handling
from sklearn.feature_extraction.text import CountVectorizer  # ✅ For Bag of Words feature extraction
import yaml                                       # For reading YAML configuration files
import logging                                    # For logging debugging messages and errors


# Logging Configuration Setup
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function: Load Parameters
def load_params(params_path: str) -> dict:
    """Load configuration parameters from a YAML file."""
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


# Function: Load Data
def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV and handle missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Replace NaNs with empty strings
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# Function: Apply Bag of Words (CountVectorizer)
def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Bag of Words vectorization to transform text into numerical features."""
    try:
        # ✅ Initialize the CountVectorizer (BoW)
        vectorizer = CountVectorizer(max_features=max_features)

        # Extract text and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Fit on training data and transform both train/test
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse matrix → DataFrame and append labels
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('Bag of Words applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


# Function: Save Data
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame into CSV file at the specified path."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# Main Execution Function
def main():
    try:
        # Step 1: Load params
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        # Step 2: Load preprocessed data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Step 3: Apply BoW
        train_df, test_df = apply_bow(train_data, test_data, max_features)

        # Step 4: Save outputs
        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
