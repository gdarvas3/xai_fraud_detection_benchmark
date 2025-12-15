# src/utils.py

import random
import os
import numpy as np
import pandas as pd
import json
import pickle
import logging
import seaborn as sns
from pathlib import Path
from typing import Any, Dict

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Reproducibility ---

def set_seed(seed: int = 42):
    """
    Sets the global seed for reproducibility.
    Affects 'random', 'numpy', and 'os' (PYTHONHASHSEED).
    """
    random.seed(seed)
    np.random.seed(seed)
    # Fix Python hashing for certain data structures
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to: {seed}")


# --- File Handling (I/O) ---

def save_pickle(data: Any, filepath: Path):
    """
    Saves data (e.g., model, scaler) to a pickle file.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.debug(f"Pickle file saved successfully to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving pickle file ({filepath}): {e}")

def load_pickle(filepath: Path) -> Any:
    """
    Loads data from a pickle file.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logging.debug(f"Pickle file loaded successfully from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file ({filepath}): {e}")
        return None

def save_json(data: Dict, filepath: Path):
    """
    Saves a dictionary (e.g., metrics) to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.debug(f"JSON file saved successfully to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON file ({filepath}): {e}")

def load_json(filepath: Path) -> Dict:
    """
    Loads a JSON file as a dictionary.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"JSON file loaded successfully from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON file ({filepath}): {e}")
        return {}


# --- Visualization ---

def plot_metric_comparison(
    metrics_data: Dict[str, Dict[str, float]],
    metric_to_plot: str,
    save_path: Path
):
    """
    Creates a bar plot from the benchmark results (all_metrics)
    for a specific metric and saves it.

    Args:
        metrics_data: The 'all_metrics' dict collected by main.py.
                      Format: {'model_name': {'metric1': val1, 'metric2': val2}}
        metric_to_plot: The metric to plot (e.g., 'f1' or 'roc_auc').
        save_path: The .png file path to save the plot.
    """
    logging.info(f"Generating comparison plot for metric: '{metric_to_plot}'...")

    # Prepare data for plotting
    try:
        # Extract models and the requested metric value
        plot_data = []
        for model_name, metrics in metrics_data.items():
            if 'error' in metrics:
                logging.warning(f"Model {model_name} failed, skipping from plot.")
                continue
            
            value = metrics.get(metric_to_plot)
            if value is not None:
                plot_data.append({'Modell': model_name, metric_to_plot: value})
            else:
                logging.warning(f"Metric {metric_to_plot} is missing for model {model_name}.")

        if not plot_data:
            logging.error("No data available to plot.")
            return

        df = pd.DataFrame(plot_data)
        df = df.sort_values(by=metric_to_plot, ascending=False)

        # Create the plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x='Modell',
            y=metric_to_plot,
            data=df,
            palette='viridis' 
        )
        
        # Set titles and labels
        ax.set_title(f"Model Comparison: {metric_to_plot.upper()}", fontsize=16)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(f"Score ({metric_to_plot})", fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotate model names if they are long

        # Annotate bars with their values
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}", # Format to 3 decimal places
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )
        
        # Save plot
        plt.tight_layout() # Ensure everything fits
        plt.savefig(save_path)
        plt.close() # Close plot to free memory
        
        logging.info(f"Comparison plot saved to: {save_path}")

    except Exception as e:
        logging.error(f"Error during plot generation: {e}")


# --- Logging Setup ---

def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """
    Sets up a centralized logging configuration.
    (Can be used instead of the simpler setup in main.py)
    """
    level = logging.getLevelName(log_level.upper())
    
    # Basic log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # Console handler (StreamHandler)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # File handler (FileHandler), if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    logging.info("Logging successfully configured.")