import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import warnings
from pathlib import Path

import config

# For server environments (headless plotting)
matplotlib.use('Agg')

# Import model types
from sklearn.linear_model import SGDOneClassSVM, LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import LocalOutlierFactor

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# --- HELPER FUNCTIONS ---

def _ensure_dataframe(X, columns=None):
    """Ensures data is a DataFrame (required for SHAP/LIME)."""
    if isinstance(X, np.ndarray):
        if columns is None:
            columns = [f"feat_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)
    return X

def select_interesting_indices(y_test, y_pred, num_examples=1):
    """
    Selects examples from the four quadrants of the confusion matrix.
    """
    # Convert to numpy array
    if hasattr(y_test, 'values'): y_test = y_test.values
    if hasattr(y_pred, 'values'): y_pred = y_pred.values
    
    # Flatten if necessary
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2: 
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = y_pred.ravel().astype(int)

    indices = {
        'TP': np.where((y_test == 1) & (y_pred == 1))[0], # True Positive
        'TN': np.where((y_test == 0) & (y_pred == 0))[0], # True Negative
        'FP': np.where((y_test == 0) & (y_pred == 1))[0], # False Positive (False Alarm)
        'FN': np.where((y_test == 1) & (y_pred == 0))[0]  # False Negative (Missed Fraud)
    }
    
    selected_indices = {}
    for category, idx_list in indices.items():
        if len(idx_list) > 0:
            np.random.seed(42)
            selected = np.random.choice(idx_list, size=min(len(idx_list), num_examples), replace=False)
            selected_indices[category] = selected
        else:
            selected_indices[category] = []
            
    return selected_indices

def _unwrap_model(model):
    """Unwraps Pipeline/Mapie/Calibrated models to retrieve the base estimator."""
    if hasattr(model, "estimator") and model.__class__.__name__ == "MapieClassifier":
        return _unwrap_model(model.estimator)
    if isinstance(model, CalibratedClassifierCV):
        # Use base_estimator if available
        if hasattr(model, "base_estimator"):
            return model.base_estimator
        # Or use the first calibrated classifier if fitted
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
             return model.calibrated_classifiers_[0].base_estimator
    return model

def _get_scoring_function(model):
    """Returns the appropriate scoring function for unsupervised models."""
    if hasattr(model, "decision_function"):
        return model.decision_function
    elif hasattr(model, "score_samples"):
        return model.score_samples
    elif hasattr(model, "predict"):
        return model.predict
    return None

# --- SHAP LOGIC ---

def save_shap_plots(shap_values, X_sample, model_name, save_dir: Path):
    """Saves SHAP plots and feature importance data with cleaned labels."""
    try:
        if shap_values is None:
            return

        # Create directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. FEATURE NAME CLEANING ---
        # Clean up feature names for better visualization (removing technical prefixes)
        if hasattr(shap_values, "feature_names") and shap_values.feature_names is not None:
            clean_names = []
            for name in shap_values.feature_names:
                # Remove specific long dataset prefixes
                new_name = new_name.replace("cat__PERFORM_CNS_SCORE_DESCRIPTION_", "")
                clean_names.append(new_name)
            
            # Overwrite names in the SHAP object for the plots
            shap_values.feature_names = clean_names

        # --- 2. BEESWARM PLOT ---
        logging.info("Generating SHAP beeswarm plot...")
        plt.figure(figsize=(12, 8))
        
        # Create plot
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        
        # Adjust left margin to fit long labels (Safety measure)
        plt.subplots_adjust(left=0.35) 
        
        plt.savefig(save_dir / f"{model_name}_shap_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()

        # --- 3. BAR PLOT ---
        logging.info("Generating SHAP bar plot...")
        plt.figure(figsize=(12, 8))
        
        shap.plots.bar(shap_values, max_display=15, show=False)
        
        # Adjust left margin here as well
        plt.subplots_adjust(left=0.35)
        
        plt.savefig(save_dir / f"{model_name}_shap_bar.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- 4. CSV EXPORT (Feature Importance) ---
        logging.info("Saving SHAP feature importance to CSV...")
        
        # Extract values
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            # Use the cleaned names we created above
            feature_names = shap_values.feature_names 
        else:
            vals = shap_values
            feature_names = X_sample.columns if hasattr(X_sample, "columns") else [f"feat_{i}" for i in range(vals.shape[1])]

        # Reduce dimensionality (multiclass or interaction values)
        if vals.ndim == 3: 
             vals = np.abs(vals[:,:,1]).mean(0) # Use positive class
        elif vals.ndim == 2: 
             vals = np.abs(vals).mean(0)
        else:
             vals = np.abs(vals).mean(0)

        # Save DataFrame
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        feature_importance.to_csv(save_dir / f"{model_name}_shap_top_features.csv", index=False)

    except Exception as e:
        logging.error(f"Error saving SHAP plots for {model_name}: {e}", exc_info=True)

def generate_shap_values(model, X_train_full, X_to_explain, model_name):
    """Generates SHAP values."""
    try:
        if model is None: return None, None

        # Downsample data for performance
        if len(X_to_explain) > 1000:
            X_to_explain = X_to_explain.sample(1000, random_state=42)
        
        background_data = shap.sample(X_train_full, 50) 
        
        # Unwrap model
        core_model = _unwrap_model(model)
        
        explainer = None
        shap_values = None

        # --- A) TREE MODELS (Fast) ---
        if hasattr(core_model, "tree_") or hasattr(core_model, "estimators_") or isinstance(core_model, IsolationForest) or isinstance(core_model, XGBClassifier):
            try:
                # Special case for Isolation Forest
                if isinstance(core_model, IsolationForest):
                    explainer = shap.TreeExplainer(core_model)
                    shap_values = explainer(X_to_explain)
                else:
                    # model_output='probability' ensures consistent interpretation
                    explainer = shap.TreeExplainer(core_model, data=background_data, model_output="probability")
                    shap_values = explainer(X_to_explain)
            except Exception as e:
                logging.warning(f"TreeExplainer failed for {model_name} ({e}), falling back to KernelExplainer...")

        # --- B) OTHER MODELS (Slower, KernelExplainer) ---
        if explainer is None:
            
            feature_names = X_train_full.columns.tolist()

            # B.1. Unsupervised (OCSGD, LOF)
            if isinstance(core_model, (SGDOneClassSVM, OneClassSVM, LocalOutlierFactor)):
                
                score_fn = _get_scoring_function(core_model)
                if score_fn is None: return None, None
                
                def predict_fn_ocsgd(X):
                    X_df = _ensure_dataframe(X, feature_names)
                    return score_fn(X_df)
                
                explainer = shap.KernelExplainer(predict_fn_ocsgd, background_data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_values = explainer(X_to_explain)

            # B.2. Supervised (Linear, SVM, etc.)
            else:
                # Prediction wrapper for KernelExplainer
                def predict_fn_proba(X):
                    X_df = _ensure_dataframe(X, feature_names)
                    if hasattr(model, "predict_proba"): # Call original model (might be calibrated)
                        return model.predict_proba(X_df)[:, 1]
                    elif hasattr(model, "decision_function"):
                        return model.decision_function(X_df)
                    else:
                        return model.predict(X_df)

                # Optimize for Linear models if possible
                if isinstance(core_model, (LogisticRegression, LinearSVC)):
                    try:
                        explainer = shap.LinearExplainer(core_model, background_data)
                        shap_values = explainer(X_to_explain)
                    except:
                        explainer = shap.KernelExplainer(predict_fn_proba, background_data)
                        shap_values = explainer(X_to_explain)
                else:
                    explainer = shap.KernelExplainer(predict_fn_proba, background_data)
                    shap_values = explainer(X_to_explain)

        # --- FORMATTING (Handle Multiclass or List output) ---
        if isinstance(shap_values, list):
            # We are generally interested in the positive class (index 1)
            if len(shap_values) > 1:
                return explainer, shap_values[1]
            return explainer, shap_values[0]
        
        return explainer, shap_values

    except Exception as e:
        logging.error(f"Critical error in generate_shap_values for {model_name}: {e}", exc_info=True)
        return None, None

# --- LIME LOGIC ---

def generate_lime_explanation(model, X_train, X_test, instance_index, model_name,unique_name, save_path):
    """Generates a single LIME explanation."""
    try:
        feature_names = X_train.columns.tolist()
        
        # Detect unsupervised models
        core_model = _unwrap_model(model)
        unsupervised_types = (IsolationForest, SGDOneClassSVM, OneClassSVM, LocalOutlierFactor)
        is_unsupervised = isinstance(core_model, unsupervised_types) or hasattr(model, "score_samples")

        # Initialize Explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=['Normal', 'Fraud'] if not is_unsupervised else None,
            mode='regression' if is_unsupervised else 'classification',
            random_state=42
        )
        
        # Prediction wrapper (DataFrame <-> Numpy)
        def predict_fn(X):
            if isinstance(X, np.ndarray): 
                X_df = pd.DataFrame(X, columns=feature_names)
            else: 
                X_df = X

            if is_unsupervised:
                score_func = _get_scoring_function(model)
                return score_func(X_df) if score_func else model.predict(X_df)
            else:
                # Supervised: Expecting probabilities (N, 2)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X_df)
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_df)
                    probs = 1 / (1 + np.exp(-scores)) # Sigmoid
                    return np.vstack([1-probs, probs]).T
                else:
                    preds = model.predict(X_df)
                    return np.vstack([1-preds, preds]).T

        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=X_test.iloc[instance_index].values,
            predict_fn=predict_fn,
            num_features=10
        )
        
        # Save explanation
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

            html_filename = save_path / f"{model_name}_{unique_name}_lime_idx{instance_index}.html"
            explanation.save_to_file(str(html_filename))
            
    except Exception as e:
        logging.error(f"LIME error for {model_name}, index {instance_index}: {e}")

def run_local_explanations(model, X_train, X_test, y_test, model_name, save_path, num_examples=1):
    """LIME workflow: Selects TP/FP/TN/FN examples and generates explanations."""
    logging.info(f"Starting LOCAL LIME workflow for {model_name}...")
    
    try:
        # 1. Get predictions (0/1 format)
        if hasattr(model, "predict"):
            raw_pred = model.predict(X_test)
            if isinstance(raw_pred, tuple): raw_pred = raw_pred[0] # Mapie fix
        else:
            logging.warning(f"Model {model_name} has no predict method. Skipping LIME.")
            return

        # Binarization
        if model_name in ['isolation_forest', 'ocsgd', 'lof', 'one_class_svm']:
            # -1 (Anomaly) -> 1, 1 (Normal) -> 0
            y_pred_binary = np.where(raw_pred == -1, 1, 0)
        else:
            if raw_pred.ndim == 2:
                 # Probabilities or MAPIE set
                 if raw_pred.shape[1] > 1: 
                     y_pred_binary = np.argmax(raw_pred, axis=1)
                 else: 
                     y_pred_binary = raw_pred.astype(int).ravel()
            else:
                y_pred_binary = raw_pred.astype(int)

        # 2. Select indices
        indices_dict = select_interesting_indices(y_test, y_pred_binary, num_examples)

        # 3. Generate LIME for selected indices
        for category, indices in indices_dict.items():
            for idx in indices:
                unique_name = f"{category}" # e.g., TP, FP
                generate_lime_explanation(
                    model, X_train, X_test, 
                    instance_index=idx, 
                    model_name=model_name,
                    unique_name = unique_name, 
                    save_path=save_path
                )
                
    except Exception as e:
        logging.error(f"Error in run_local_explanations for {model_name}: {e}", exc_info=True)

# --- MASTER WRAPPER ---

def run_xai_pipeline(model, X_train, X_test, y_test, model_name, results_path_shap: Path, results_path_lime : Path):
    """
    Runs SHAP and LIME analysis for a given model.
    """
    
    print(f"\n>>> Starting XAI Pipeline: {model_name} <<<")
    
    # 1. SHAP
    if config.RUN_SHAP:
        logging.info(f"Calculating SHAP values ({model_name})...")
        explainer, shap_vals = generate_shap_values(model, X_train, X_test, model_name)
        if shap_vals is not None:
            save_shap_plots(shap_vals, X_test, model_name, results_path_shap)
            logging.info(f"SHAP completed. Saved to: {results_path_shap}")
        else:
            logging.warning(f"SHAP failed for: {model_name}")

    # 2. LIME
    if config.RUN_LIME:
        logging.info(f"Calculating LIME ({model_name})...")
        run_local_explanations(
            model=model, 
            X_train=X_train, 
            X_test=X_test, 
            y_test=y_test, 
            model_name=model_name, 
            save_path=results_path_lime,
            num_examples=1
        )
        logging.info(f"LIME completed.")