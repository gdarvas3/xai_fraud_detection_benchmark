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
    Calculates the given metrics
    """
    results = {}
    
    # Use 'weighted' average for metrics to account for class imbalance
    avg_method = 'weighted' 
    
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
            
            elif metric_name == 'roc_auc':
                if y_proba is None:
                    logging.warning(f"Cannot calculate ROC-AUC: pred_proba is missing")
                    results[metric_name] = None
                else:
                    # Handle binary (1D) vs. multiclass (2D) proba arrays
                    if len(y_proba.shape) == 1: 
                        results[metric_name] = roc_auc_score(y_true, y_proba)
                    else:
                        results[metric_name] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=avg_method)
            
            elif metric_name == 'pr_auc':
                if y_proba is None:
                    logging.warning(f"Cannot calculate PR-AUC: pred_proba is missing")
                    results[metric_name] = None
                else:
                    # Note: average_precision_score is for binary or multilabel
                    # For PR-AUC, we typically use the positive class probability (1D array)
                    results[metric_name] = average_precision_score(y_true, y_proba)
            else:
                logging.warning(f"Unknown metric: '{metric_name}'")
                
        except Exception as e:
            logging.error(f"Error while calculating: {metric_name} : {e}")
            results[metric_name] = None
            
    return results

# --- Helper Function: Unsupervised Scoring ---
def _get_unsupervised_scores(
    model: Model, 
    X_test: pd.DataFrame
) -> Union[np.ndarray, None]:
    """
    Gets anomaly scores (decision_function or score_samples).
    Always returns scores where a HIGHER value means more anomalous.
    """
    # 1. Use decision_function (e.g., IsolationForest, OneClassSVM)
    if hasattr(model, "decision_function"):
        # Most models output LOWER scores for anomalies, so we negate.
        logging.debug("Using -decision_function() for scores.")
        return -model.decision_function(X_test)
        
    # 2. Use score_samples (e.g., LocalOutlierFactor)
    elif hasattr(model, "score_samples"):
        # LOF's score_samples is a measure of normality (higher=more normal), 
        # so we negate it to get an anomaly score.
        logging.debug("Using -score_samples() for scores.")
        return -model.score_samples(X_test)
        
    # 3. Handle models without scoring
    else:
        logging.warning(f"Model {type(model).__name__} has no decision_function or score_samples. "
                        "PR-AUC and ROC-AUC will be unavailable.")
        return None


# --- Main Function 1: Supervised Workflow ---
def train_and_evaluate_supervised(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_calc: List[str]
) -> Tuple[Model, MetricsDict]:
    """
    Trains and evaluates a single SUPERVISED model.
    
    """
    
    model_name = type(model).__name__
    logging.info(f"Starting model training: {model_name}...")

    # 1. Train the model
    try:
        model.fit(X_train, y_train)
        logging.info(f"{model_name} training finished.")
    except Exception as e:
        logging.error(f"Error while training model: {model_name}: {e}")
        return model, {"error": str(e)}

    # 2. Generate predictions
    logging.info("Generating predictions on test dataset...")
    y_pred = model.predict(X_test)
    
    # 3. Generate prediction probabilities (if needed for metrics)
    y_proba = None
    proba_needed = any(metric in metrics_to_calc for metric in ['pr_auc', 'roc_auc'])
    
    if proba_needed:
        if hasattr(model, "predict_proba"):
            try:
                y_proba_full = model.predict_proba(X_test)
                
                # For binary classification, use the probability of the positive class (class 1)
                if len(y_proba_full.shape) == 2 and y_proba_full.shape[1] == 2:
                    y_proba = y_proba_full[:, 1]
                else:
                    # Fallback for multiclass
                    y_proba = y_proba_full

            except Exception as e:
                logging.warning(f"Error calling 'predict_proba': {e}")
        else:
            logging.warning(f"Model {model_name} lacks 'predict_proba' method. AUC metrics unavailable.")

    # 4. Calculate metrics
    logging.info("Calculating metrics...")
    metrics_results = _calculate_metrics(y_test, y_pred, y_proba, metrics_to_calc)
    
    # 5. Return trained model and results
    return model, metrics_results

# --- Main Function 2: Unsupervised Workflow ---
def train_and_evaluate_unsupervised(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series, # Not used for training, but kept for API consistency
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
        y_proba_scores = _get_unsupervised_scores(model, X_test)

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