# src/explain.py

import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

import utils

# --- Model Type Definitions ---

LINEAR_MODELS = [
    'LogisticRegression',
    'LinearSVC',
    'SGDOneClassSVM'
]

TREE_BASED_MODELS = [
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier'
]

UNSUPERVISED_MODELS = [
    'IsolationForest',
    'LocalOutlierFactor',
    'OneClassSVM',
    'LinearSVC',
    'SGDOneClassSVM'
]

# --- Helper Function for Unsupervised Models ---

def _get_anomaly_score_function(model):
    # 1. Check for 'decision_function' (e.g., IsolationForest, OCSVM)
    if hasattr(model, "decision_function"):
        logging.debug("Explainer: Using -model.decision_function()")
        def predict_fn(X):
            return -model.decision_function(X)
        return predict_fn
    
    # 2. Check for 'score_samples' (e.g., LOF)
    elif hasattr(model, "score_samples"):
        logging.debug("Explainer: Using -model.score_samples()")
        def predict_fn(X):
            return -model.score_samples(X)
        return predict_fn
    
    # 3. Handle models without a scoring function
    else:
        logging.error(f"Model {type(model).__name__} is not explainable (no score/decision function).")
        return None

# --- SHAP Explanation ---

def generate_shap_values(
    model, 
    X_train_full: pd.DataFrame,  
    X_to_explain: pd.DataFrame,  
    model_name: str, 
    save_path: Path
):
    logging.info(f"Starting SHAP explanation on model: {model_name}...")
    
    try:
        # 1. Create background data sample for explainers
        X_background = shap.sample(X_train_full, 100) 
        
        # 2. Select the correct SHAP explainer based on model type
        model_type = type(model).__name__
        
        # 2.1. Handle fast TreeExplainers
        if model_type in TREE_BASED_MODELS:
            logging.debug("SHAP: Using TreeExplainer.")
            explainer = shap.TreeExplainer(model, X_train_full, model_output="probability")
        
        # 2.2. Handle fast LinearExplainers
        elif model_type in LINEAR_MODELS:
            logging.debug("SHAP: Using LinearExplainer.")
            explainer = shap.LinearExplainer(model, X_background)

        # 2.3. Handle slow KernelExplainers (e.g., RBF SVC, LOF)
        else:
            # 2.3.1. Downsample data for slow explainers to avoid long runtimes
            X_to_explain = X_to_explain.sample(100, random_state=42)

            logging.warning(f"SHAP: Using KernelExplainer (model: {model_type}). This may be slow.")
            
            # 2.3.2. Get the correct prediction function for KernelExplainer
            predict_fn_for_kernel = None
            
            if model_type in UNSUPERVISED_MODELS:
                # Case A: Unsupervised model (use anomaly score)
                predict_fn_for_kernel = _get_anomaly_score_function(model)
                if predict_fn_for_kernel is None:
                    logging.error(f"{model_type} cannot be explained by SHAP.")
                    return 
            else:
                # Case B: Supervised model (e.g., RBF SVC, use probability)
                if not hasattr(model, 'predict_proba'):
                     logging.error(f"{model_type} has no 'predict_proba' method. (Set probability=True?)")
                     return 
                
                def predict_fn_for_kernel(X):
                    # Return probability for the positive class (class 1)
                    return model.predict_proba(X)[:, 1]

            # 2.3.3. Initialize the slow explainer
            explainer = shap.KernelExplainer(predict_fn_for_kernel, X_background)
            
        # 3. Calculate SHAP values
        logging.info("Calculating SHAP values...")
        shap_values_obj = explainer(X_to_explain)
        logging.info("SHAP calculation finished.")
        
        # --- 4. Generate and save SHAP summary plots ---
        
        # 4.1. Save beeswarm plot (global summary)
        try:
            logging.info("Generating SHAP beeswarm plot...")
            plt.figure() 
            shap.summary_plot(
                shap_values_obj, 
                X_to_explain, 
                plot_type="beeswarm", 
                show=False
            )
            plot_filename = save_path / f"{model_name}_shap_beeswarm.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close() 
            logging.info(f"SHAP beeswarm plot saved to {plot_filename}")

        except Exception as e:
            logging.warning(f"Could not generate SHAP beeswarm plot: {e}")

        # 4.2. Save bar plot (global feature importance)
        try:
            logging.info("Generating SHAP bar plot...")
            plt.figure()
            
            # Select correct SHAP values for bar plot
            if len(shap_values_obj.shape) == 3:
                # Use values for the positive class (class 1)
                shap_values_for_bar = shap_values_obj.values[:,:,1]
            else:
                # Use the 2D array as-is (unsupervised/regression)
                shap_values_for_bar = shap_values_obj
                
            shap.summary_plot(
                shap_values_for_bar, 
                X_to_explain, 
                plot_type="bar", 
                show=False
            )
            plot_filename = save_path / f"{model_name}_shap_bar.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            logging.info(f"SHAP bar plot saved to {plot_filename}")
        
        except Exception as e:
            logging.warning(f"Could not generate SHAP bar plot: {e}")

        # --- 5. Calculate and save top features to CSV ---
        try:
            logging.info("Calculating and saving SHAP feature importance to table...")

            # 5.1. Get the correct SHAP values to process
            if len(shap_values_obj.shape) == 3:
                # Binary classification: use positive class (1)
                shap_values_to_process = shap_values_obj.values[:,:,1]
            else:
                # Regression / Unsupervised: use the 2D array
                shap_values_to_process = shap_values_obj.values
                
            # 5.2. Calculate mean absolute SHAP value (global importance)
            mean_abs_shap = np.mean(np.abs(shap_values_to_process), axis=0)
            
            # 5.3. Create DataFrame
            feature_names = X_to_explain.columns.tolist()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap
            })
            
            # 5.4. Sort features by importance
            importance_df = importance_df.sort_values(by='mean_abs_shap', ascending=False)
            
            # 5.5. Save table to CSV
            table_filename = save_path / f"{model_name}_shap_top_features.csv"
            importance_df.to_csv(table_filename, index=False)
            logging.info(f"SHAP feature importance table saved to: {table_filename}")

        except Exception as e:
            logging.warning(f"Could not save SHAP feature importance table: {e}")

    except Exception as e:
        logging.error(f"Error during SHAP explanation for {model_name}: {e}", exc_info=True)

# --- LIME Explanation ---

def generate_lime_explanation(model, X_train, X_test, instance_index, model_name, save_path):
    try:
        logging.info(f"Starting LIME explanation for model: ({model_name}), index: {instance_index}...")
        
        model_type = type(model).__name__
        predict_fn_to_use = None
        
        # 1. Determine LIME mode and prediction function
        if model_type in UNSUPERVISED_MODELS:
            # 1.1. Set up for Unsupervised (Regression mode on anomaly score)
            logging.info(f"LIME: Using Unsupervised (regression) mode for {model_type}.")
            predict_fn_to_use = _get_anomaly_score_function(model)
            lime_mode = 'regression'
            class_names_to_use = None 
        else:
            # 1.2. Set up for Supervised (Classification mode)
            logging.info(f"LIME: Using Classification mode for {model_type}.")
            if not hasattr(model, 'predict_proba'):
                 logging.error(f"{model_type} has no 'predict_proba' method. LIME canceled.")
                 return
            predict_fn_to_use = model.predict_proba
            lime_mode = 'classification'
            class_names_to_use = ['class_0', 'class_1']
        
        if predict_fn_to_use is None:
            logging.error(f"{model_type} cannot be explained by LIME.")
            return

        # 2. Initialize LIME Tabular Explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=class_names_to_use,
            mode=lime_mode                  
        )
        
        # 3. Get the specific instance to explain
        instance_to_explain = X_test.iloc[instance_index]
        
        # 4. Generate explanation for the instance
        explanation = explainer.explain_instance(
            data_row=instance_to_explain.values,
            predict_fn=predict_fn_to_use,
            num_features=10
        )
        
        # 5. Save explanation report as HTML
        html_filename = save_path / f"{model_name}_lime_instance_{instance_index}.html"
        explanation.save_to_file(str(html_filename))
        logging.info(f"LIME explanation saved to: {html_filename}")

    except Exception as e:
        logging.error(f"Error during LIME explanation: {e}", exc_info=True)