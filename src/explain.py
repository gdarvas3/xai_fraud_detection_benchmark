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
def generate_shap_values(
    model, 
    X_train_full: pd.DataFrame,  # <-- Új paraméter: a teljes train adat
    X_to_explain: pd.DataFrame,  # <-- Új paraméter: a test adat
    model_name: str, 
    save_path: Path
):
    """
    Generates SHAP explanation for a given model...
    
    Args:
        model: Trained model object
        X_train_full (pd.DataFrame): A TELJES train adathalmaz, amiből mintát veszünk.
        X_to_explain (pd.DataFrame): Az adathalmaz, amit magyarázni akarunk (pl. X_test).
        ...
    """
    logging.info(f"Starting SHAP explanation on model: {model_name}...")
    
    try:
        # 1. Háttéradat létrehozása (a te javaslatod alapján)
        # Ezt a mintát használjuk referenciaként.
        # 100-200 minta elég, hogy ne legyen lassú a KernelExplainer.
        X_background = shap.sample(X_train_full, 100) 
        
        # 2. A helyes explainer kiválasztása
        model_type = type(model).__name__
        
        if model_type in TREE_BASED_MODELS:
            logging.debug("SHAP: Using TreeExplainer.")
            explainer = shap.TreeExplainer(model, X_train_full, model_output="probability")
        
        elif model_type == 'LogisticRegression':
            logging.debug("SHAP: Using LinearExplainer.")
            # A LinearExplainer is igényli a háttéradatot a korrelációkhoz
            explainer = shap.LinearExplainer(model, X_background)

        else:
            logging.warning(f"SHAP: No specific explainer for {model_type}. Using slow KernelExplainer.")
            # A KernelExplainer a háttéradatot (X_background) használja inicializáláshoz
            explainer = shap.KernelExplainer(model.predict_proba, X_background)

        # 3. SHAP értékek számítása a magyarázandó adaton (X_to_explain)
        logging.info("Calculating SHAP values...")
        shap_values_obj = explainer(X_to_explain)
        logging.info("SHAP calculation finished.")

        # 1. Summary Plot (Beeswarm) mentése
        # Ez a leggyakoribb globális magyarázat
        try:
            logging.info("Generating SHAP beeswarm plot...")
            plt.figure() # Fontos, hogy tiszta ábrát kezdjünk
            shap.summary_plot(
                shap_values_obj, 
                X_to_explain, 
                plot_type="beeswarm", 
                show=False
            )
            plot_filename = save_path / f"{model_name}_shap_beeswarm.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close() # Zárd be az ábrát, hogy ne szemetelje tele a memóriát
            logging.info(f"SHAP beeswarm plot saved to {plot_filename}")

        except Exception as e:
            logging.warning(f"Could not generate SHAP beeswarm plot: {e}")

        # 2. Bar Plot (feature importance) mentése
        try:
            logging.info("Generating SHAP bar plot...")
            plt.figure()
            # Figyelem: A 'shap_values_obj' struktúrája függhet a modelltől
            # (pl. binárisnál lehet, hogy shap_values_obj[:,:,1]-et kell használnod)
            # Tegyük fel, hogy a shap_values_obj a "pozitív" osztályra vonatkozik
            shap.summary_plot(
                shap_values_obj, 
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

    except Exception as e:
        logging.error(f"Hiba a SHAP magyarázat generálása során ({model_name}): {e}", exc_info=True)
    except Exception as e:
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