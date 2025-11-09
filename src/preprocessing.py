# src/preprocessing.py

import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List

# Sklearn importok a pipeline-hoz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# A segédfüggvényeink (a mentéshez)
import utils


# --- [MUSZÁJ MÓDOSÍTANI] ---
# Itt add meg a célváltozód (target) nevét
TARGET_COLUMN = "isFraud" 

# --- [MUSZÁJ MÓDOSÍTANI] ---
# A nyers adatfájlod neve a 'data/raw/' mappában
RAW_DATA_FILENAME = "ieee_fraud.csv" 
# ------------------------------


def _get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automatikusan azonosítja a numerikus és kategorikus oszlopokat.
    Kihagyja a célváltozót.
    """
    # Távolítsuk el a célváltozót a listából, ha létezik
    if TARGET_COLUMN in df.columns:
        features_df = df.drop(columns=[TARGET_COLUMN])
    else:
        features_df = df

    numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    logging.info(f"Azonosított numerikus oszlopok: {numeric_features}")
    logging.info(f"Azonosított kategorikus oszlopok: {categorical_features}")
    
    return numeric_features, categorical_features

def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Felépíti a ColumnTransformer-t a kétféle oszloptípusra.
    """
    # Numerikus pipeline:
    # 1. Hiányzó értékek pótlása (imputáció) mediánnal
    # 2. Skálázás (StandardScaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Kategorikus pipeline:
    # 1. Hiányzó értékek pótlása (imputáció) a leggyakoribb értékkel
    # 2. One-Hot kódolás (az ismeretlen kategóriákat figyelmen kívül hagyja)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # A kettő kombinálása egy ColumnTransformer-ben
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Meghagyja azokat az oszlopokat, amiket nem soroltunk be
    )
    
    return preprocessor

def _get_processed_column_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """
    Kinyeri az oszlopneveket a transzformáció után (fontos a SHAP-hoz).
    """
    try:
        # Kategorikus nevek kinyerése a OneHotEncoder-ből
        cat_transformer = preprocessor.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features).tolist()
    except Exception as e:
        logging.warning(f"Nem sikerült kinyerni a kategorikus oszlopneveket: {e}. Általános neveket használunk.")
        cat_feature_names = [f"cat_{i}" for i in range(len(categorical_features))] # Ez csak egy fallback

    # A numerikus nevek ugyanazok maradnak
    return numeric_features + cat_feature_names


def load_and_preprocess_data(
    raw_data_path: Path,
    processed_data_path: Path,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    A teljes adatfeldolgozási folyamat fő függvénye.
    """
    logging.info("Adat-előkészítési folyamat indítása...")
    
    # --- 1. Betöltés ---
    try:
        data_file = raw_data_path / RAW_DATA_FILENAME
        df = pd.read_csv(data_file)
        logging.info(f"Nyers adat sikeresen betöltve innen: {data_file}")
    except FileNotFoundError:
        logging.error(f"Kritikus hiba: A '{data_file}' fájl nem található.")
        raise
    except Exception as e:
        logging.error(f"Hiba a nyers adat betöltésekor: {e}")
        raise

    if TARGET_COLUMN not in df.columns:
        logging.error(f"Kritikus hiba: A '{TARGET_COLUMN}' célváltozó nem található az adathalmazban.")
        raise ValueError(f"Célváltozó '{TARGET_COLUMN}' nem létezik.")

    # --- 2. X és y szétválasztása ---
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # --- 3. Train/Test szétválasztás (MÉG a transzformáció ELŐTT) ---
    # Ez a legfontosabb lépés az adatszivárgás elkerülésére.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    logging.info(f"Adatok szétválasztva. Train méret: {X_train.shape}, Test méret: {X_test.shape}")

    # --- 4. Preprocessing Pipeline felépítése ---
    numeric_features, categorical_features = _get_feature_types(X_train)
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    # --- 5. Pipeline illesztése és transzformálása ---
    # A pipeline-t KIZÁRÓLAG a X_train-en illesztjük (fit_transform)
    logging.info("Preprocessing pipeline illesztése (fit) a train adatokra...")
    X_train_processed_np = preprocessor.fit_transform(X_train)
    
    # A X_test-en MÁR CSAK transzformálunk (transform)
    logging.info("Preprocessing pipeline alkalmazása (transform) a test adatokra...")
    X_test_processed_np = preprocessor.transform(X_test)

    # --- 6. Pipeline mentése a jövőbeli használathoz ---
    # Ez kritikus, mert pontosan ezt kell használni az éles predikciókhoz!
    preprocessor_path = processed_data_path / "preprocessor.pkl"
    utils.save_pickle(preprocessor, preprocessor_path)
    logging.info(f"Feldolgozó pipeline (preprocessor) elmentve ide: {preprocessor_path}")

    # --- 7. Visszaalakítás DataFrame-mé (SHAP és olvashatóság miatt) ---
    # A ColumnTransformer numpy tömböt ad vissza, de nekünk kellenek az oszlopnevek.
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
    
    # --- 8. (Opcionális) Feldolgozott adatok mentése ---
    # Jó gyakorlat elmenteni a tiszta, feldolgozott szetteket.
    X_train_processed_df.to_csv(processed_data_path / "train_features.csv", index=False)
    X_test_processed_df.to_csv(processed_data_path / "test_features.csv", index=False)
    y_train.to_csv(processed_data_path / "train_target.csv", index=False)
    y_test.to_csv(processed_data_path / "test_target.csv", index=False)
    logging.info(f"Feldolgozott train/test szettek elmentve a(z) '{processed_data_path}' mappába.")
    
    # --- 9. Visszatérés a feldolgozott adatokkal ---
    return X_train_processed_df, X_test_processed_df, y_train, y_test