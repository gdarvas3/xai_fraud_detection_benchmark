# src/training.py

import logging
from typing import Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np

# Import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
 
# --- Type Hint Definitions ---
Model = BaseEstimator
MetricsDict = Dict[str, float]


# --- Helper Function: Metric Calculation ---
def _calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray, None],
    metrics_to_calc: List[str]
) -> MetricsDict:
    """
    Calculates standard metrics. Handles both 1D and 2D (One-Hot/Set) inputs automatically.
    """
    results = {}
    avg_method = 'weighted' 
    
    # --- 1. Y_TRUE (Igazság) KEZELÉSE ---
    if hasattr(y_true, "values"):
        y_true = y_true.values
        
    # Ha a target One-Hot kódolt (N, 2), visszaalakítjuk 1D-re (N,)
    if y_true.ndim == 2:
        if y_true.shape[1] == 2:
             # [[0, 1], [1, 0]] -> [1, 0]
            y_true = np.argmax(y_true, axis=1)
        else:
            y_true = y_true.ravel()

    # --- 2. Y_PRED (Tipp) KEZELÉSE ---
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
        
    # Ha a predikció 2D (pl. MAPIE set output), visszaalakítjuk
    # FIGYELEM: Standard metrikákhoz a "legvalószínűbb" osztály kell, nem a halmaz!
    if y_pred.ndim == 2:
        # Ha ez egy probability output véletlenül
        if np.issubdtype(y_pred.dtype, np.floating): 
            y_pred = np.argmax(y_pred, axis=1)
        # Ha ez egy MAPIE boolean set [[True, False]]
        elif y_pred.dtype == bool:
            # Ez egy durva egyszerűsítés: ha bármelyik True, azt vesszük (de standard metrikára nem ideális)
            # Jobb megoldás: A hívó félnek a sima .predict() eredményét kell átadnia
            logging.warning("Warning: 2D boolean set passed to standard metrics. Flattening...")
            y_pred = y_pred.ravel() 

    # --- 3. Y_PROBA (Valószínűség) KEZELÉSE ---
    # Ez kell az AUC-hoz
    score_to_use = None
    if y_proba is not None:
        if hasattr(y_proba, "values"):
            y_proba = y_proba.values
            
        # Ha (N, 2) alakú (pl. [[0.1, 0.9], ...]), akkor a 2. oszlop (Class 1) kell nekünk
        if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            score_to_use = y_proba[:, 1] # Csak a pozitív osztály valószínűsége
        else:
            score_to_use = y_proba.ravel()

    # --- SZÁMÍTÁS ---
    for metric_name in metrics_to_calc:
        try:
            if metric_name == 'accuracy':
                results[metric_name] = accuracy_score(y_true, y_pred)
            elif metric_name == 'f1_score':
                results[metric_name] = f1_score(y_true, y_pred, average=avg_method)
            elif metric_name == 'precision':
                results[metric_name] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
            elif metric_name == 'recall':
                results[metric_name] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
            elif metric_name in ['roc_auc', 'pr_auc']:
                if score_to_use is not None:
                    if metric_name == 'roc_auc':
                        results[metric_name] = roc_auc_score(y_true, score_to_use)
                    else:
                        results[metric_name] = average_precision_score(y_true, score_to_use)
                else:
                    results[metric_name] = None
        except Exception as e:
            # logging.error(f"Error calculating {metric_name}: {e}") # Kikommentelheted, ha zavar a sok log
            results[metric_name] = None
            
    return results

# --- Helper Function: Unsupervised Scoring ---
def _get_probability_scores(
    model: Model, 
    X_test: pd.DataFrame
) -> Union[np.ndarray, None]:
    """
    Gets anomaly scores (decision_function or score_samples).
    Always returns scores where a HIGHER value means more anomalous.
    """
    # 1. Use predict_proba
    if hasattr(model, "predict_proba"):
        logging.debug("Using -decision_function() for scores.")
        return model.predict_proba(X_test)
    # 2. Use decision_function
    elif hasattr(model, "decision_function"):
        logging.debug("Using -decision_function() for scores.")
        return -model.decision_function(X_test)
        
    # 3. Use score_samples
    elif hasattr(model, "score_samples"):
        logging.debug("Using -score_samples() for scores.")
        return -model.score_samples(X_test)
        
    # 4. Handle models without scoring
    else:
        logging.warning(f"Model {type(model).__name__} has no decision_function or score_samples. "
                        "PR-AUC and ROC-AUC will be unavailable.")
        return None
        
def _calculate_conformal_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_sets: np.ndarray
) -> MetricsDict:
    """
    Calculates metrics specific to Conformal Prediction (Coverage, Set Size).
    """
    results = {}
    try:
        # 1. Coverage: Milyen arányban tartalmazza a halmaz a valódi címkét?
        # A mapie coverage score-ja ezt számolja ki
        cov = classification_coverage_score(y_true, y_pred_sets)
        results['cp_coverage'] = cov
        
        # 2. Average Set Size: Átlagosan hány osztályt jósoltunk? (Minél kisebb, annál jobb/biztosabb)
        # y_pred_sets alakja: (n_samples, n_classes, 1) vagy (n_samples, n_classes)
        set_sizes = y_pred_sets.sum(axis=1)
        results['cp_avg_set_size'] = float(np.mean(set_sizes))
        
        # 3. Empty Sets: Hány százalékban nem mert semmit mondani?
        empty_sets = (set_sizes == 0).sum()
        results['cp_empty_set_ratio'] = float(empty_sets / len(y_true))
        
    except Exception as e:
        logging.error(f"Error calculating conformal metrics: {e}")
        
    return results

# --- Main Function 1: Supervised Workflow ---
def train_and_evaluate_supervised(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_calc: List[str],
    alpha: float = 0.05
) -> Tuple[MapieClassifier, MetricsDict]:
    """
    Trains and evaluates a single SUPERVISED model.
    
    """
    
    model_name = type(model).__name__
    logging.info(f"Starting model training: {model_name}...")

    if isinstance(model, LinearSVC):
        logging.info("Wrapping LinearSVC in CalibratedClassifierCV to enable predict_proba...")
        model = CalibratedClassifierCV(model, cv=3, n_jobs=-1)

    # 1. Train the model
    logging.info("Step 1: Fitting Base Model on Balanced Data...")
    model.fit(X_train, y_train)

    # 2. Calibrate Mapie on CALIBRATION data (Original Distribution)
    logging.info("Step 2: Calibrating Mapie on Calibration Data...")
    mapie_clf = MapieClassifier(estimator=model, cv="prefit")
    mapie_clf.fit(X_calib, y_calib)

    # 3. Predict on Test Data
    logging.info("Step 3: Generating Predictions on Test Data...")
    y_pred, y_pis = mapie_clf.predict(X_test, alpha=alpha)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        # LinearSVC: Nincs predict_proba, de a decision_function eredménye
        # arányos a valószínűséggel (távolság a hipersíktól), így jó AUC-hoz.
        y_proba = model.decision_function(X_test)
    else:
        # Végső eset (nem ideális AUC-hoz)
        y_proba = None

    # 4. Calculate Standard Metrics (Accuracy, AUC, etc.)
    logging.info("Calculating Standard Metrics...")
    standard_metrics = _calculate_metrics(y_test, y_pred, y_proba, metrics_to_calc)

    # 5. Calculate Conformal Metrics (Coverage, Set Size)
    logging.info("Calculating Conformal Metrics...")
    # y_pis shape is usually (n_samples, n_classes, n_alphas). Since we passed one alpha, take index 0
    y_pis_final = y_pis[:, :, 0] if y_pis.ndim == 3 else y_pis
    conformal_metrics = _calculate_conformal_metrics(y_test, y_pis_final)
    
    # Merge metrics
    all_metrics = {**standard_metrics, **conformal_metrics}
    all_metrics['alpha'] = alpha # Store what confidence level was used
    
    logging.info(f"Training complete. Coverage: {all_metrics.get('cp_coverage', 'N/A'):.4f}")
    
    return mapie_clf, all_metrics

# --- Main Function 2: Unsupervised Workflow ---
def train_and_evaluate_unsupervised(
    model: Model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_calc: List[str]
) -> Tuple[Model, MetricsDict]:
    """
    Trains and evaluates an UNSUPERVISED anomaly detection model.
    """
    
    model_name = type(model).__name__
    logging.info(f"Starting UNSUPERVISED model training: {model_name}...")

    # 1. Train the model (without labels)
    try:
        model.fit(X_train) # Note: y_train is NOT used
        logging.info(f"{model_name} training finished.")
    except Exception as e:
        logging.error(f"Error while training unsupervised model: {model_name}: {e}")
        return model, {"error": str(e)}

    # 2. Generate predictions
    logging.info("Generating predictions on test dataset (unsupervised)...")
    y_pred_raw = model.predict(X_test) # Output is typically [-1, 1]
    
    # 3. Map predictions to standard [0, 1] format
    # Assumes y_test: 1=Fraud, 0=Normal
    # Assumes model: -1=Outlier (Fraud), 1=Inlier (Normal)
    y_pred_mapped = np.where(y_pred_raw == -1, 1, 0)
    logging.debug("Mapped unsupervised predictions: -1 -> 1 (fraud), 1 -> 0 (normal).")

    # 4. Get anomaly scores (for AUC metrics)
    y_proba_scores = None
    proba_needed = any(metric in metrics_to_calc for metric in ['pr_auc', 'roc_auc'])

    if proba_needed:
        y_proba_scores = _get_probability_scores(model, X_test)

    # 5. Calculate metrics using mapped predictions and scores
    logging.info("Calculating metrics (unsupervised mapped)...")
    metrics_results = _calculate_metrics(
        y_test, 
        y_pred_mapped,  # The mapped 0/1 predictions
        y_proba_scores, # The standardized anomaly scores
        metrics_to_calc
    )
    
    # 6. Return trained model and results
    return model, metrics_results