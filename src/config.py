# src/config.py

from pathlib import Path

# Dataset data
DATASET_ID = 'ieee_fraud_demo'
TARGET_COLUMN = "isFraud" 
ID_COLUMN = 'TransactionID'
TIMESTAMP_COLUMN = 'TransactionDT'

# --- Project Path Definitions ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"

# Data I/O Paths
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Results & Output Paths
EXPLANATIONS_SHAP_PATH = RESULTS_PATH / "explanations" / "shap" / DATASET_ID
EXPLANATIONS_LIME_PATH = RESULTS_PATH / "explanations" / "lime" / DATASET_ID
METRICS_PATH = RESULTS_PATH / "metrics"
PLOTS_PATH = RESULTS_PATH / "plots"
MODELS_PATH = RESULTS_PATH / "models"


# --- Experiment Hyperparameters ---
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42


# --- Execution Toggles ---
RUN_SHAP = True
RUN_LIME = True


# --- Model & Metric Selection ---

# Define which models to run
SUPERVISED_MODELS = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svc']
UNSUPERVISED_MODELS = ['isolation_forest', 'ocsvm']
MODELS_TO_RUN = SUPERVISED_MODELS + UNSUPERVISED_MODELS

# Define which metrics to calculate
METRICS = ['accuracy', 'f1_score','precision', 'recall', 'roc_auc', 'pr_auc']


# --- Model Hyperparameters ---
MODEL_CONFIGS = {
    # Supervised Models
    'logistic_regression': {
        'C': 1.0,
        'solver': 'liblinear',
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'svc': {
        'C': 1.0,
        # 'kernel': 'linear',
        # 'gamma': 'scale',
        # 'probability': True,
        'random_state': 42
    },

    # Unsupervised Models
    'lof': {
        'novelty': True,
        'contamination': 0.05
    },
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 0.05,
        'random_state': 42
    },
    'ocsvm': {
        'nu': 0.05, 
        'learning_rate': 'adaptive',
        'eta0': 0.01,
        'tol': 1e-3,
        'max_iter': 1000,
        'random_state': 42
    }
}