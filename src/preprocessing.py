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
TIMESTAMP_COLUMN = config.TIMESTAMP_COLUMN

# --- Preprocessing Pipeline Helpers ---

def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kezeli mindkét esetet:
    A) Numerikus offset (pl. 0, 1, 500) -> Dátummá konvertálja referenciához képest.
    B) String dátum (pl. "2020-01-01 10:00") -> Dátummá konvertálja közvetlenül.
    """
    
    col = TIMESTAMP_COLUMN  # Rövidítés a kód átláthatóságáért
    
    if col not in df.columns:
        logging.error(f"HIBA: A '{col}' oszlop nem található.")
        return df

    logging.info(f"Időbélyeg feldolgozása ('{col}')...")

    # --- 1. ESET: Már eleve dátum objektum ---
    if is_datetime64_any_dtype(df[col]):
        df['TransactionDateTime'] = df[col]
        logging.info(" -> Típus: Dátum objektum (OK)")

    # --- 2. ESET (A opció): Numerikus értékek (0, 1, 2...) ---
    elif is_numeric_dtype(df[col]):
        logging.info(" -> Típus: Numerikus (A opció). Konvertálás referenciához képest...")
        start_date = pd.to_datetime('2020-01-01 00:00:00')
        # Unit='s' (másodperc). Ha az adataid nem másodpercek, itt írd át (pl. 'm', 'h', 'D')
        df['TransactionDateTime'] = start_date + pd.to_timedelta(df[col], unit='s')

    # --- 3. ESET (B opció): Szöveges dátum ("2020-01-01...") ---
    else:
        logging.info(" -> Típus: Szöveg/Object (B opció). Dátum parse-olás...")
        # Megpróbáljuk dátummá alakítani. Ami nem sikerül, az NaT lesz.
        df['TransactionDateTime'] = pd.to_datetime(df[col], errors='coerce')

    # --- ELLENŐRZÉS ---
    # Ha a konverzió után sok a hiba (NaT), jelezzük
    if df['TransactionDateTime'].isnull().any():
        null_count = df['TransactionDateTime'].isnull().sum()
        logging.warning(f"VIGYÁZAT: {null_count} sorban nem sikerült az időt értelmezni (NaT)!")

    # --- Feature Generálás ---
    df['Hour_of_Day'] = df['TransactionDateTime'].dt.hour
    df['Day_of_Week'] = df['TransactionDateTime'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

    # Hiányzó értékek kitöltése (hogy a pipeline ne szálljon el később)
    df['Hour_of_Day'] = df['Hour_of_Day'].fillna(-1) 
    df['Day_of_Week'] = df['Day_of_Week'].fillna(-1)
    df['Is_Weekend'] = df['Is_Weekend'].fillna(0)

    # Takarítás: töröljük az eredetit és az ideiglenes dátumot
    cols_to_drop = [col, 'TransactionDateTime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logging.info("Idő feature-ök (Hour, Day, Weekend) sikeresen létrehozva.")
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
    potential_categorical_features = features_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in potential_categorical_features]

    categorical_features = []
    dropped_features = []

    for col in potential_categorical_features:
        unique_count = features_df[col].nunique()
        
        if unique_count <= config.CARDINALITY_THRESHOLD:
            categorical_features.append(col)
        else:
            # Ha túl sok az egyedi érték, eldobjuk
            dropped_features.append(col)
            logging.warning(f"KIZÁRVA: '{col}' túl sok egyedi értéket tartalmaz ({unique_count}). Valószínűleg ID vagy szöveg.")
    
    logging.info(f"Identified {len(numeric_features)} numeric features.")
    logging.info(f"Identified {len(categorical_features)} categorical features.")
    
    return numeric_features, categorical_features 

def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # 1. Numeric pipeline (változatlan)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical pipeline (ITT A VÁLTOZÁS)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        
        # A ritka kategóriák összevonása:
        ('onehot', OneHotEncoder(
            # 'infrequent_if_exist': Ha a teszt halmazban jön egy sosem látott érték,
            # azt is a "ritka" (Other) kategóriába sorolja, nem pedig nullázza.
            handle_unknown='infrequent_if_exist', 
            
            sparse_output=False,
            
            # min_frequency: Ezt állítsd be!
            # Ha float (pl. 0.01): Azok a kategóriák, amik az adatok < 1%-ában vannak, összevonásra kerülnek.
            # Ha int (pl. 20): Amikből kevesebb mint 20 darab van, azok összevonásra kerülnek.
            min_frequency=0.05 
        ))
    ])

    # 3. Combine pipelines (változatlan)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
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
    calibration_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    logging.info("Starting data preprocessing pipeline...")
    
    # 1. Load raw data & 2. Feature Engineering (Ugyanaz, mint nálad)
    # ... (Kódod 1-es és 2-es pontja változatlan) ...
    # Feltételezzük, hogy itt már megvan az X és y
    
    # --- KÓDOD ELEJE ---
    data_file = raw_data_path / RAW_DATA_FILENAME
    df = pd.read_csv(data_file)

    if(config.DATASET_ID == 'bot_accounts'):
        unique_targets = df[TARGET_COLUMN].unique()
        logging.info(f"Eredeti target értékek: {unique_targets}")

        # Kézi leképezés a biztonság kedvéért (hogy biztosan a Bot legyen az 1-es!)
        # Írd át a szótár kulcsait (pl. 'human', 'bot'), ahogy a CSV-ben szerepelnek!
        label_mapping = {
            'human': 0,  # Normál
            'bot': 1     # Anomália/Csalás
        }

        # Ha esetleg más stringek vannak, módosítsd a fenti map-et. 
        # Ha nem találja a kulcsot, NaN lesz, ezért fontos ellenőrizni.
        if df[TARGET_COLUMN].dtype == 'object':
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map(label_mapping)

    if TARGET_COLUMN not in df.columns: raise ValueError("No target")
    df = df.drop(columns=config.COLS_TO_DROP, errors='ignore')
    if TIMESTAMP_COLUMN in df.columns: df = _create_time_features(df)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    # --- KÓDOD VÉGE ---

    # 3. Split data (STRATIFIED - Nagyon fontos!)
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
    logging.info(f"  Calib: {X_calib.shape} (Used for MAPIE Calibration - Original Dist)")
    logging.info(f"  Test:  {X_test.shape} (Used for Final Evaluation)")

# 4. Build & Fit Preprocessor
    numeric_features, categorical_features = _get_feature_types(X_train)

    for col in numeric_features:
        # Ez a sor a "fraud_Bernhard-Lesch" jellegű stringeket átalakítja NaN-ra
        # Így a scaler nem fog hibát dobni
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_calib[col] = pd.to_numeric(X_calib[col], errors='coerce')
        X_test[col]  = pd.to_numeric(X_test[col], errors='coerce')

    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    logging.info("Fitting preprocessor on ORIGINAL training data...")
    preprocessor.fit(X_train)

    # Transzformálás
    X_train_proc = preprocessor.transform(X_train)
    X_calib_proc = preprocessor.transform(X_calib)
    X_test_proc = preprocessor.transform(X_test)

    # --- JAVÍTOTT RÉSZ KEZDETE ---
    
    # 1. Oszlopnevek automatikus lekérdezése a preprocessortól
    # Ez pontosan megmondja, mi lett az eredmény (pl. numeric + onehot oszlopok)
    try:
        new_cols = preprocessor.get_feature_names_out()
        logging.info(f"Feature names extracted successfully: {len(new_cols)} columns.")
    except Exception as e:
        logging.error(f"Error getting feature names: {e}. Using generic indices.")
        new_cols = [f"feat_{i}" for i in range(X_train_proc.shape[1])]

    # 2. Ritka mátrix (Sparse matrix) kezelése
    # Ha a OneHotEncoder sparse mátrixot ad vissza, azt át kell alakítani tömbbé a DataFrame-hez
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_calib_proc = X_calib_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    # 3. DataFrame-ek létrehozása a helyes nevekkel
    X_train_df = pd.DataFrame(X_train_proc, columns=new_cols)
    X_calib_df = pd.DataFrame(X_calib_proc, columns=new_cols)
    X_test_df = pd.DataFrame(X_test_proc, columns=new_cols)
    
    # --- JAVÍTOTT RÉSZ VÉGE ---

    # y indexek igazítása (Ez maradhat, ez fontos!)
    y_train = y_train.reset_index(drop=True)
    y_calib = y_calib.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    logging.info(f"DataFrames reconstructed. Shape: {X_train_df.shape}")

    # -------------------------------------------------------------------------
    # 5. STRATÉGIÁK MEGVALÓSÍTÁSA
    # -------------------------------------------------------------------------

    # A) TRAIN SET (SMOTE-olva a Classifiernek)
    logging.info("Applying SMOTE to Training set...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)
    
    X_train_bal.to_csv(processed_data_path / "train_features_balanced.csv", index=False)
    y_train_bal.to_csv(processed_data_path / "train_target_balanced.csv", index=False)
    
    # B) CALIBRATION SET (EREDETI ELOSZLÁS - MAPIE-nek)
    # Ezt NEM szabad SMOTE-olni!
    logging.info("Saving Calibration set (Original Distribution)...")
    X_calib_df.to_csv(processed_data_path / "calib_features.csv", index=False)
    y_calib.to_csv(processed_data_path / "calib_target.csv", index=False)

    # C) TEST SET (Közös kiértékelés)
    logging.info("Saving Test set...")
    X_test_df.to_csv(processed_data_path / "test_features.csv", index=False)
    y_test.to_csv(processed_data_path / "test_target.csv", index=False)
    
    # D) UNSUPERVISED TRAIN (Csak a "Train" halmaz normális adatai)
    # Fontos: A Calib halmaz normális adatait NE használjuk itt, mert azokat a Mapie-nek tartogatjuk!
    X_train_normal_part = X_train_df[y_train == 0]
    X_calib_normal_part = X_calib_df[y_calib == 0]
    X_unsupervised_full = pd.concat([X_train_normal_part, X_calib_normal_part], axis=0)
    X_unsupervised_full.to_csv(processed_data_path / "train_features_normal.csv", index=False)

    return X_train_bal, X_calib_df, X_test_df, X_unsupervised_full, y_train_bal, y_calib, y_test