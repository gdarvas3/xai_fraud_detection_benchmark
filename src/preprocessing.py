# src/preprocessing.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List

# Sklearn imports for pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Local imports
import utils

# --- Data Configuration ---
TARGET_COLUMN = "isFraud" 
RAW_DATA_FILENAME = "ieee_fraud.csv" 
ID_COLUMN = 'TransactionID'
TIMESTAMP_COLUMN = 'TransactionDT'

# --- Preprocessing Pipeline Helpers ---

def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates meaningful time-based features from the 'TransactionDT' column
    and removes the original raw column.
    """
    # 1. Convert 'TransactionDT' (seconds) to a datetime object
    # We assume an arbitrary start date, as only the cyclical patterns matter.
    start_date = pd.to_datetime('2020-01-01 00:00:00')
    df['TransactionDateTime'] = start_date + pd.to_timedelta(df['TransactionDT'], unit='s')
    
    # 2. Extract cyclical and contextual features
    df['Hour_of_Day'] = df['TransactionDateTime'].dt.hour
    df['Day_of_Week'] = df['TransactionDateTime'].dt.dayofweek  # Monday=0, Sunday=6
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    
    # 3. Drop the original/intermediate columns
    df = df.drop(columns=['TransactionDT', 'TransactionDateTime'])
    
    logging.info("Time features (Hour_of_Day, Day_of_Week, Is_Weekend) created from TransactionDT.")
    return df

def _get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automata...
    """
    features_df = df.copy()
    if TARGET_COLUMN in features_df.columns:
        logging.warning(f"Target column '{TARGET_COLUMN}' found in feature data. Dropping it.")
        features_df = features_df.drop(columns=[TARGET_COLUMN])

    numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    logging.info(f"Identified {len(numeric_features)} numeric features.")
    logging.info(f"Identified {len(categorical_features)} categorical features.")
    
    return numeric_features, categorical_features 

def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # Define numeric pipeline (Impute median -> Scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define categorical pipeline (Impute most frequent -> OneHotEncode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Pass through any columns not specified
    )
    
    return preprocessor

def _get_processed_column_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    # Extract feature names after transformation (especially from OneHotEncoder)
    try:
        cat_transformer = preprocessor.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features).tolist()
    except Exception as e:
        logging.warning(f"Could not extract OHE names: {e}. Using generic names.")
        cat_feature_names = [f"cat_{i}" for i in range(len(categorical_features))] 

    # Combine numeric names (which don't change) and new categorical names
    return numeric_features + cat_feature_names


# --- Main Data Processing Function ---

def load_and_preprocess_data(
    raw_data_path: Path,
    processed_data_path: Path,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main function for the data processing workflow.
    """
    logging.info("Starting data preprocessing pipeline...")
    
    # 1. Load raw data
    try:
        data_file = raw_data_path / RAW_DATA_FILENAME
        df = pd.read_csv(data_file)
        logging.info(f"Raw data loaded successfully from {data_file}")
    except FileNotFoundError:
        logging.error(f"Critical Error: Raw data file not found at {data_file}")
        raise
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        raise

    # 1.1. Validate target column
    if TARGET_COLUMN not in df.columns:
        logging.error(f"Critical Error: Target column '{TARGET_COLUMN}' not in DataFrame.")
        raise ValueError(f"Target column '{TARGET_COLUMN}' does not exist.")

    # 2. Feature Engineering

    # 2.1. Drop useless ID columns
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        logging.info(f"Dropped '{ID_COLUMN}' column.")

    # 2.2. Create time-based features
    if TIMESTAMP_COLUMN in df.columns:
        df = _create_time_features(df) # Ez a függvény már eldobja a TIMESTAMP_COLUMN-t
        logging.info(f"Created time features from '{TIMESTAMP_COLUMN}'.")

    # 2.3. Separate features (X) and target (y)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # 3. Split data into train and test sets (before any transformation)
    # This is the most important step to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y # Ensure class distribution is preserved
    )
    logging.info(f"Data split into train/test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Build the preprocessing pipeline
    numeric_features, categorical_features = _get_feature_types(X_train)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    # 5. Fit pipeline on training data and transform both sets
    logging.info("Fitting preprocessor on training data...")
    X_train_processed_np = preprocessor.fit_transform(X_train)
    
    logging.info("Transforming test data using fitted preprocessor...")
    X_test_processed_np = preprocessor.transform(X_test)

    # 6. Save the fitted preprocessor object for future inference
    preprocessor_path = processed_data_path / "preprocessor.pkl"
    utils.save_pickle(preprocessor, preprocessor_path)
    logging.info(f"Preprocessor object saved to: {preprocessor_path}")

    # 7. Convert processed NumPy arrays back to DataFrames
    # This is crucial for readability and for passing feature names to SHAP
    new_column_names = _get_processed_column_names(preprocessor, numeric_features, categorical_features)
    
    X_train_processed_df = pd.DataFrame(
        X_train_processed_np, 
        columns=new_column_names,
        index=X_train.index
    )
    X_test_processed_df = pd.DataFrame(
        X_test_processed_np, 
        columns=new_column_names,
        index=X_test.index
    )
    
    # 8. Save processed data sets to 'processed' folder
    X_train_processed_df.to_csv(processed_data_path / "train_features.csv", index=False)
    X_test_processed_df.to_csv(processed_data_path / "test_features.csv", index=False)
    y_train.to_csv(processed_data_path / "train_target.csv", index=False)
    y_test.to_csv(processed_data_path / "test_target.csv", index=False)
    logging.info(f"Processed data sets saved to '{processed_data_path}'")
    
    # 9. Return processed data for use in the main training pipeline
    return X_train_processed_df, X_test_processed_df, y_train, y_test