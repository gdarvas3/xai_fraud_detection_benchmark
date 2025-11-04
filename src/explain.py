# src/explain.py

import shap
import lime
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

import utils

# List of tree-based models
TREE_BASED_MODELS = [
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier'
]

# SHAP explanation
def generate_shap_values(model, X_data: pd.DataFrame, model_name: str, save_path: Path):
    """
    Generates SHAP explanation for a given model and saves the figure and the SHAP values
    
    Args:
        model: Traind model object
        X_data (pd.DataFrame): Pandas dataframe on which the explanation is generated
        model_name (str): Name of the model
        save_path (Path): Path to save
    """
    logging.info(f"Starting SHAP explanation on model: {model_name}...")
    
    try:
        # 1. Select explainer
        model_type = type(model).__name__
        
        if model_type in TREE_BASED_MODELS:
            logging.debug("SHAP: Using TreeExplainer.")
            # For tree-based models TreeExplainer is more efficient
            explainer = shap.TreeExplainer(model)
            # SHAP values 
            shap_values_obj = explainer(X_data)

        else:
            logging.debug("SHAP: KernelExplainer vagy PermutationExplainer használata.")
            # General but slower SHAP explainer for the other models
            explainer = shap.Explainer(model.predict_proba, X_data)
            shap_values_obj = explainer(X_data)

        # 2. SHAP values for the positive class

        #The Explainer returns and 'Explanation' object, we must check if the shape is 3D (binary classification)
        
        shap_values_for_plot = shap_values_obj
        shap_values_for_csv = shap_values_obj.values

        if len(shap_values_obj.values.shape) == 3:
            shap_values_for_plot = shap_values_obj[:, :, 1]
            shap_values_for_csv = shap_values_obj.values[:, :, 1]
        elif len(shap_values_obj.values.shape) != 2:
            logging.warning("Unexpected SHAP dimension.")

        # 3. Save summary plot
        plot_filename = save_path / f"{model_name}_shap_summary_plot.png"
        logging.info(f"SHAP summary plot generated and saved to: {plot_filename}")

        plt.figure()

        shap.summary_plot(
            shap_values_for_plot,
            X_data,
            plot_type="beeswarm",
            show=False
        )
        
        # Save matplotlib figure
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

        # 4. Save raw SHAP values
        shap_filename = save_path / f"{model_name}_shap_values.csv"
        logging.info(f"Raw SHAP values saved to: {shap_filename}")
        
        # Create a dataframe from raw values
        shap_df = pd.DataFrame(shap_values_for_csv, columns=X_data.columns)
        shap_df.to_csv(shap_filename, index=False)

    except Exception as e:
        # Log error
        logging.error(f"Hiba a SHAP magyarázat generálása során ({model_name}): {e}", exc_info=True)


# LIME explanation


def generate_lime_explanation(model, X_train, X_test, instance_index, model_name, save_path):
    """
    Generates LIME explanation for a given instance of a given model and saves it as HTML
    """
    try:
        logging.info(f"Starting LIME explanation for model: ({model_name}), index: {instance_index}...")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['class_0', 'class_1'],
            mode='classification'
        )
        
        instance_to_explain = X_test.iloc[instance_index]
        
        # Explanation for top 10 features
        explanation = explainer.explain_instance(
            data_row=instance_to_explain.values,
            predict_fn=model.predict_proba,
            num_features=10
        )
        
        # Save as HTML
        html_filename = save_path / f"{model_name}_lime_instance_{instance_index}.html"
        explanation.save_to_file(str(html_filename))
        logging.info(f"LIME explanation saved to: {html_filename}")

    except Exception as e:
        logging.error(f"Error during lime explanation: {e}")