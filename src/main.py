# src/main.py

import logging
import os
from pathlib import Path

# Import modules from source directory
import config
import models
import preprocessing
import training
import explain
import utils

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)

def run_benchmark():
    """
    Main function to run the benchmarking
    """
    logging.info("Strating benchmarking...")

    # 1. Set basic settings
    utils.set_seed(config.RANDOM_STATE)
    
    # Creating directories
    Path(config.METRICS_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.PLOTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.EXPLANATIONS_SHAP_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.EXPLANATIONS_LIME_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_PATH).mkdir(parents=True, exist_ok=True)


    # 2. Data preprocessing
    logging.info("Load and preprocess data...")
    try:
        X_train, X_test, y_train, y_test = preprocessing.load_and_preprocess_data(
            raw_data_path=config.RAW_DATA_PATH,
            processed_data_path=config.PROCESSED_DATA_PATH,
            test_size=config.TEST_SPLIT_SIZE,
            random_state=config.RANDOM_STATE
        )
        logging.info(f"Successfuly loaded data. Train size: {X_train.shape}, Test size: {X_test.shape}")
    except Exception as e:
        logging.error(f"Error during data loading: {e}")
        return

    # 3. Iteration: Train, Evaluate, Explain ---
    all_metrics = {}

    logging.info(f"Starting iteration with following models: {config.MODELS_TO_RUN}")

    for model_name in config.MODELS_TO_RUN:
        logging.info(f"--- Model: {model_name} ---")
        
        try:
            # 3.1 Initialize model
            logging.info("Initialize model parameters from config...")
            params = config.MODEL_CONFIGS.get(model_name, {})
            model = models.get_model(model_name, params)

            # Unsupervised model training
            if(model_name in config.SUPERVISED_MODELS):
                # 3.2 Train and Evaluate
                logging.info("Training and evaluating (supervised)...")
                trained_model, metrics = training.train_and_evaluate_supervised(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics_to_calc=config.METRICS
                )
            elif (model_name in config.UNSUPERVISED_MODELS):
                logging.info("Training and evaluating (supervised)...")
                trained_model, metrics = training.train_and_evaluate_unsupervised(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    metrics_to_calc=config.METRICS
                )
            
            all_metrics[model_name] = metrics
            logging.info(f"Results ({model_name}): {metrics}")

            # 3.3 Save trained model
            model_save_path = Path(config.MODELS_PATH) / f"{model_name}.pkl"
            utils.save_pickle(trained_model, model_save_path)
            logging.info(f"Model saved to: {model_save_path}")

            # 3.4 Generate explanation
            if config.RUN_SHAP:
                logging.info(f"SHAP magyarázat generálása ({model_name})...")
                explain.generate_shap_values(
                    model=trained_model,
                    X_train_full=X_train,
                    X_to_explain = X_test,
                    model_name=model_name,
                    save_path=config.EXPLANATIONS_SHAP_PATH
                )
                logging.info(f"SHAP magyarázat elmentve.")
            if config.RUN_LIME:
                logging.info(f"LIME magyarázat generálása ({model_name})...")
                explain.generate_lime_explanation(
                    model=trained_model,
                    X_train=X_train,
                    X_test = X_test,
                    instance_index=123,
                    model_name=model_name,
                    save_path=config.EXPLANATIONS_LIME_PATH
                )
                logging.info(f"LIME magyarázat elmentve.")

        except Exception as e:
            logging.error(f"Error with model: {model_name}: {e}")
            all_metrics[model_name] = {"error": str(e)}

    # 4. Summarize and save results
    logging.info("Benchmarking finished. Saving results...")

    try:
        metrics_save_path = Path(config.RESULTS_PATH) / "benchmark_summary.json"
        utils.save_json(all_metrics, metrics_save_path)
        logging.info(f"Metrics summary saved to: {metrics_save_path}")

    except Exception as e:
        logging.error(f"Failed to save results: {e}")

    logging.info("Benchmarking finished.")


if __name__ == "__main__":
    run_benchmark()