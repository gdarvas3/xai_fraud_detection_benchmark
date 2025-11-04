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
    roc_auc_score
)

from sklearn.base import BaseEstimator

Model = BaseEstimator
MetricsDict = Dict[str, float]


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
    
    # Using 'Weighted' average
    avg_method = 'weighted' 
    
    for metric_name in metrics_to_calc:
        try:
            if metric_name == 'accuracy':
                results[metric_name] = accuracy_score(y_true, y_pred)
            
            elif metric_name == 'f1':
                results[metric_name] = f1_score(y_true, y_pred, average=avg_method)
            
            elif metric_name == 'precision':
                results[metric_name] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
            
            elif metric_name == 'recall':
                results[metric_name] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
            
            elif metric_name == 'roc_auc':
                if y_proba is None:
                    logging.warning(f"Can not calculate ROC-AUC pred_proba is missing")
                    results[metric_name] = None
                else:
                    # Kétosztályos esetben a y_proba (n_samples,) alakú
                    if len(y_proba.shape) == 1: 
                        results[metric_name] = roc_auc_score(y_true, y_proba)
                    # Többosztályos esetben (n_samples, n_classes)
                    else:
                        results[metric_name] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=avg_method)
            
            else:
                logging.warning(f"Unkonwn metric: '{metric_name}'")
                
        except Exception as e:
            logging.error(f"Error while calculating: {metric_name} : {e}")
            results[metric_name] = None
            
    return results


def train_and_evaluate(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_calc: List[str]
) -> Tuple[Model, MetricsDict]:
    """
    Train and evaluate a single model

    Args:
        model (Model): Scikit-learn compatible model
        X_train (pd.DataFrame): Train dataset (features)
        y_train (pd.Series): Train dataset (labels)
        X_test (pd.DataFrame): Test dataset (features)
        y_test (pd.Series): Test dataset (labels)
        metrics_to_calc (List[str]): List of metrics to be calculated

    Returns:
        Tuple[Model, MetricsDict]: 
            1. Trained model object
            2. Dict with calculated metrics
    """
    
    model_name = type(model).__name__
    logging.info(f"Starting model training: {model_name}...")

    # 1. Training
    try:
        model.fit(X_train, y_train)
        logging.info(f"{model_name} training finished.")
    except Exception as e:
        logging.error(f"Error while training model: {model_name}: {e}")
        # Visszaadunk egy üres eredményt és a nem tanított modellt
        return model, {"error": str(e)}

    # 2. Prediction
    logging.info("Prediction on test dataset")
    y_pred = model.predict(X_test)
    
    # Generating prediction probabilities (if supported by model)
    y_proba = None
    if 'roc_auc' in metrics_to_calc:
        if hasattr(model, "predict_proba"):
            try:
                # A [:, 1] a "pozitív" osztály valószínűségét adja (bináris esetben)
                # Többosztályos esetben a teljes (n_samples, n_classes) tömböt
                y_proba_full = model.predict_proba(X_test)
                
                if len(y_proba_full.shape) == 2 and y_proba_full.shape[1] == 2:
                    # Klasszikus bináris eset
                    y_proba = y_proba_full[:, 1]
                else:
                    # Többosztályos eset (vagy regresszió, bár itt nem)
                    y_proba = y_proba_full

            except Exception as e:
                logging.warning(f"Hiba a 'predict_proba' hívása során: {e}")
        else:
            logging.warning(f"A(z) {model_name} modell nem rendelkezik 'predict_proba' metódussal. ROC AUC nem számítható.")

    # --- 3. Kiértékelés ---
    logging.info("Metrikák számítása...")
    metrics_results = _calculate_metrics(y_test, y_pred, y_proba, metrics_to_calc)
    
    # --- 4. Visszatérés ---
    # Visszaadjuk a már tanított modellt és a metrikákat
    return model, metrics_results