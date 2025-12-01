# src/preprocessing.py

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
from imblearn.over_sampling import SMOTE

# Sklearn imports for pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Local imports
import utils
import config

# --- Data Configuration ---
TARGET_COLUMN = config.TARGET_COLUMN
RAW_DATA_FILENAME = f"{config.DATASET_ID}.csv"
TIMESTAMP_COLUMN = config.TIMESTAMP_COLUMN

# --- Preprocessing Pipeline Helpers ---

def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles timestamp processing:
    A) Numeric offset -> Convert to datetime relative to reference.
    B) String datetime -> Parse directly.
    """
    col = TIMESTAMP_COLUMN
    
    if col not in df.columns:
        logging.error(f"ERROR: Column '{col}' not found.")
        return df

    logging.info(f"Processing timestamp ('{col}')...")

    # --- CASE 1: Already datetime object ---
    if is_datetime64_any_dtype(df[col]):
        df['TransactionDateTime'] = df[col]
        logging.info(" -> Type: Datetime object (OK)")

    # --- CASE 2: Numeric values (offsets) ---
    elif is_numeric_dtype(df[col]):
        logging.info(" -> Type: Numeric. Converting relative to reference...")
        start_date = pd.to_datetime('2020-01-01 00:00:00')
        # Unit='s' (seconds). Adjust if data uses different unit (e.g., 'm', 'h', 'D')
        df['TransactionDateTime'] = start_date + pd.to_timedelta(df[col], unit='s')

    # --- CASE 3: String/Object ---
    else:
        logging.info(" -> Type: String/Object. Parsing datetime...")
        df['TransactionDateTime'] = pd.to_datetime(df[col], errors='coerce')

    # --- CHECK ---
    if df['TransactionDateTime'].isnull().any():
        null_count = df['TransactionDateTime'].isnull().sum()
        logging.warning(f"WARNING: Failed to parse time for {null_count} rows (NaT)!")

    # --- Feature Generation ---
    df['Hour_of_Day'] = df['TransactionDateTime'].dt.hour
    df['Day_of_Week'] = df['TransactionDateTime'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

    # Fill missing values to prevent pipeline errors
    df['Hour_of_Day'] = df['Hour_of_Day'].fillna(-1) 
    df['Day_of_Week'] = df['Day_of_Week'].fillna(-1)
    df['Is_Weekend'] = df['Is_Weekend'].fillna(0)

    # Cleanup: drop original and temp cols
    cols_to_drop = [col, 'TransactionDateTime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logging.info("Time features (Hour, Day, Weekend) created successfully.")
    return df

def _get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifies feature types: Numeric, Categorical, and Binary.
    Prevents binary features from being treated as high-cardinality categorical.
    """
    features_df = df.copy()
    if TARGET_COLUMN in features_df.columns:
        logging.warning(f"Target column '{TARGET_COLUMN}' found in feature data. Dropping it.")
        features_df = features_df.drop(columns=[TARGET_COLUMN])

    numeric_features = []
    categorical_features = []
    binary_features = []
    dropped_features = []

    for col in features_df.columns:
        unique_count = features_df[col].nunique()
        dtype = features_df[col].dtype

        # Check for Binary (2 unique values)
        if unique_count == 2:
            # Treat as binary to avoid OHE splitting
            binary_features.append(col)
            continue

        # Check for Numeric
        if is_numeric_dtype(dtype):
            numeric_features.append(col)
            continue

        # Check for Categorical (Cardinality Check)
        if unique_count <= config.CARDINALITY_THRESHOLD:
            categorical_features.append(col)
        else:
            # Too many unique values (likely ID or raw text)
            dropped_features.append(col)
            logging.warning(f"EXCLUDED: '{col}' has too many unique values ({unique_count}). Likely ID or text.")
    
    logging.info(f"Identified {len(numeric_features)} numeric features.")
    logging.info(f"Identified {len(binary_features)} binary features.")
    logging.info(f"Identified {len(categorical_features)} categorical features.")
    
    return numeric_features, categorical_features, binary_features

def _build_preprocessor(numeric_features: List[str], categorical_features: List[str], binary_features: List[str]) -> ColumnTransformer:
    # 1. Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical pipeline (OHE)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            handle_unknown='infrequent_if_exist', 
            sparse_output=False,
            min_frequency=0.05 # Merge rare categories (<5%)
        ))
    ])

    # 3. Binary pipeline (Impute only, no OHE)
    # Binary features are kept as single columns (0/1)
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # OrdinalEncoder ensures string binaries (Yes/No) become 0/1
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    # 4. Combine pipelines
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    if binary_features:
        transformers.append(('bin', binary_transformer, binary_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    return preprocessor

# --- Main Data Processing Function ---

def load_and_preprocess_data(
    raw_data_path: Path,
    processed_data_path: Path,
    test_size: float,
    calibration_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    logging.info("Starting data preprocessing pipeline...")
    
    # 1. Load raw data
    data_file = raw_data_path / RAW_DATA_FILENAME
    df = pd.read_csv(data_file)

    if(config.DATASET_ID == 'bot_accounts'):
        unique_targets = df[TARGET_COLUMN].unique()
        logging.info(f"Original target values: {unique_targets}")

        # Manual mapping to ensure Bot is class 1
        label_mapping = {
            'human': 0,  # Normal
            'bot': 1     # Anomaly/Fraud
        }

        if df[TARGET_COLUMN].dtype == 'object':
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(label_mapping)

    if TARGET_COLUMN not in df.columns: raise ValueError("No target")
    
    # 2. Initial Cleanup
    df = df.drop(columns=config.COLS_TO_DROP, errors='ignore')
    if TIMESTAMP_COLUMN in df.columns: 
        df = _create_time_features(df)
    
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # 3. Split data (STRATIFIED)
    X_remain, X_test, y_remain, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    relative_calib_size = calibration_size / (1 - test_size)
    
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_remain, y_remain, 
        test_size=relative_calib_size, 
        random_state=random_state, 
        stratify=y_remain
    )

    logging.info(f"Data Split Summary:")
    logging.info(f"  Train: {X_train.shape} (Used for Model Training)")
    logging.info(f"  Calib: {X_calib.shape} (Used for MAPIE Calibration)")
    logging.info(f"  Test:  {X_test.shape} (Used for Evaluation)")

    # 4. Build & Fit Preprocessor
    numeric_features, categorical_features, binary_features = _get_feature_types(X_train)

    # Ensure numeric columns are strictly numeric (coerce errors to NaN)
    for col in numeric_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_calib[col] = pd.to_numeric(X_calib[col], errors='coerce')
        X_test[col]  = pd.to_numeric(X_test[col], errors='coerce')

    preprocessor = _build_preprocessor(numeric_features, categorical_features, binary_features)

    logging.info("Fitting preprocessor on ORIGINAL training data...")
    preprocessor.fit(X_train)

    # Transform data
    X_train_proc = preprocessor.transform(X_train)
    X_calib_proc = preprocessor.transform(X_calib)
    X_test_proc = preprocessor.transform(X_test)

    # --- DataFrame Reconstruction ---
    
    # 1. Extract feature names
    try:
        new_cols = preprocessor.get_feature_names_out()
        logging.info(f"Feature names extracted successfully: {len(new_cols)} columns.")
    except Exception as e:
        logging.error(f"Error getting feature names: {e}. Using generic indices.")
        new_cols = [f"feat_{i}" for i in range(X_train_proc.shape[1])]

    # 2. Handle sparse matrix output
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_calib_proc = X_calib_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    # 3. Create DataFrames
    X_train_df = pd.DataFrame(X_train_proc, columns=new_cols)
    X_calib_df = pd.DataFrame(X_calib_proc, columns=new_cols)
    X_test_df = pd.DataFrame(X_test_proc, columns=new_cols)
    
    # Reset indices to match arrays
    y_train = y_train.reset_index(drop=True)
    y_calib = y_calib.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    logging.info(f"DataFrames reconstructed. Shape: {X_train_df.shape}")

    # -------------------------------------------------------------------------
    # 5. SAVE DATASETS
    # -------------------------------------------------------------------------

    # A) TRAIN SET (Balanced via SMOTE)
    logging.info("Applying SMOTE to Training set...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)
    
    X_train_bal.to_csv(processed_data_path / "train_features_balanced.csv", index=False)
    y_train_bal.to_csv(processed_data_path / "train_target_balanced.csv", index=False)
    
    # B) CALIBRATION SET (Original Distribution)
    logging.info("Saving Calibration set (Original Distribution)...")
    X_calib_df.to_csv(processed_data_path / "calib_features.csv", index=False)
    y_calib.to_csv(processed_data_path / "calib_target.csv", index=False)

    # C) TEST SET
    logging.info("Saving Test set...")
    X_test_df.to_csv(processed_data_path / "test_features.csv", index=False)
    y_test.to_csv(processed_data_path / "test_target.csv", index=False)
    
    # D) UNSUPERVISED TRAIN (Normal class only)
    # Combine normal samples from Train and Calib for density estimation
    X_train_normal_part = X_train_df[y_train == 0]
    X_calib_normal_part = X_calib_df[y_calib == 0]
    X_unsupervised_full = pd.concat([X_train_normal_part, X_calib_normal_part], axis=0)
    X_unsupervised_full.to_csv(processed_data_path / "train_features_normal.csv", index=False)

    return X_train_bal, X_calib_df, X_test_df, X_unsupervised_full, y_train_bal, y_calib, y_test