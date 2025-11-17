# src/preprocessing.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
from imblearn.over_sampling import SMOTE

# Sklearn imports for pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Local imports
import utils
import config

# --- Data Configuration ---
TARGET_COLUMN = config.TARGET_COLUMN
RAW_DATA_FILENAME = f"{config.DATASET_ID}.csv"
ID_COLUMN = config.ID_COLUMN
TIMESTAMP_COLUMN = config.TIMESTAMP_COLUMN

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
    
    logging.info("Starting data preprocessing pipeline...")
    
    # 1. Load raw data & 2. Feature Engineering (Ugyanaz, mint nálad)
    # ... (Kódod 1-es és 2-es pontja változatlan) ...
    # Feltételezzük, hogy itt már megvan az X és y
    
    # --- KÓDOD ELEJE ---
    data_file = raw_data_path / RAW_DATA_FILENAME
    df = pd.read_csv(data_file)
    if TARGET_COLUMN not in df.columns: raise ValueError("No target")
    if ID_COLUMN in df.columns: df = df.drop(columns=[ID_COLUMN])
    if TIMESTAMP_COLUMN in df.columns: df = _create_time_features(df)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    # --- KÓDOD VÉGE ---

    # 3. Split data (STRATIFIED - Nagyon fontos!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Build & Fit Preprocessor
    numeric_features, categorical_features = _get_feature_types(X_train)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    logging.info("Fitting preprocessor on ORIGINAL training data...")
    X_train_processed_np = preprocessor.fit_transform(X_train)
    X_test_processed_np = preprocessor.transform(X_test)
    
    # Save preprocessor
    utils.save_pickle(preprocessor, processed_data_path / "preprocessor.pkl")

    # Reconstruct DataFrames (Oszlopnevek visszanyerése)
    new_cols = _get_processed_column_names(preprocessor, numeric_features, categorical_features)
    
    X_train_df = pd.DataFrame(X_train_processed_np, columns=new_cols) # Index reset miatt nem kell index=...
    X_test_df = pd.DataFrame(X_test_processed_np, columns=new_cols) # Itt sem, mert numpy array lett
    
    # y indexek igazítása (Mivel a numpy array eldobta az indexet, reseteljük az y-t is)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 5. STRATÉGIÁK MEGVALÓSÍTÁSA
    # -------------------------------------------------------------------------

    # A) STANDARD MENTÉS (Baseline)
    logging.info("Saving STANDARD datasets...")
    X_train_df.to_csv(processed_data_path / "train_features.csv", index=False)
    y_train.to_csv(processed_data_path / "train_target.csv", index=False)

    # B) SUPERVISED STRATÉGIA: SMOTE (Kiegyensúlyozás)
    logging.info("Applying SMOTE for balanced dataset...")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
    
    X_train_resampled.to_csv(processed_data_path / "train_features_balanced.csv", index=False)
    y_train_resampled.to_csv(processed_data_path / "train_target_balanced.csv", index=False)
    logging.info(f"Balanced dataset saved. Size: {X_train_resampled.shape}")

    # C) UNSUPERVISED STRATÉGIA: Csak a normális adatok
    logging.info("Filtering for Normal-Only dataset (Unsupervised)...")
    # Csak a 0-ás címkéjű sorokat tartjuk meg
    X_train_normal = X_train_df[y_train == 0]
    
    X_train_normal.to_csv(processed_data_path / "train_features_normal.csv", index=False)
    # Unsupervisedhez elvileg nem kell target, de validációhoz elmenthetjük, bár mind 0 lesz.
    logging.info(f"Normal-only dataset saved. Size: {X_train_normal.shape}")

    # 6. A TESZT HALMAZ MENTÉSE (KÖZÖS PONT!)
    logging.info("Saving COMMON TEST datasets...")
    X_test_df.to_csv(processed_data_path / "test_features.csv", index=False)
    y_test.to_csv(processed_data_path / "test_target.csv", index=False)

    return X_train_resampled, X_train_normal, X_test, y_train_resampled, y_test