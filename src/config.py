# src/config.py

from pathlib import Path

# Dataset data - vehicle_loan
DATASET_ID = 'vehicle_loan'
TARGET_COLUMN = "LOAN_DEFAULT" 
COLS_TO_DROP =['UNIQUEID', 'DATE_OF_BIRTH']
TIMESTAMP_COLUMN = ''

# # Dataset data - ulb_creditcard
# DATASET_ID = 'ulb_creditcard'
# TARGET_COLUMN = "Class" 
# COLS_TO_DROP =[]
# TIMESTAMP_COLUMN = 'Time'

# # Dataset data - bot_accounts
# DATASET_ID = 'bot_accounts'
# TARGET_COLUMN = "account_type" 
# COLS_TO_DROP =['description','profile_background_image_url','profile_image_url','screen_name','profile_background_image_path','profile_image_path', 'location', 'id']
# TIMESTAMP_COLUMN = 'created_at'

# # # Dataset data - ieee_fraud
# DATASET_ID = 'ieee_fraud'
# TARGET_COLUMN = "isFraud" 
# COLS_TO_DROP =['TransactionID']
# TIMESTAMP_COLUMN = 'TransactionDT'

# # # Dataset data - sparkov_fraud
# DATASET_ID = 'sparkov_fraud'
# TARGET_COLUMN = "is_fraud" 
# COLS_TO_DROP =['cc_num', 'first', 'last', 'street', 'trans_num']
# TIMESTAMP_COLUMN = 'trans_date_trans_time'

# --- Project Path Definitions ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"

# Data I/O Paths
RAW_DATA_PATH = DATA_PATH / "raw" / DATASET_ID
PROCESSED_DATA_PATH = DATA_PATH / "processed" / DATASET_ID

# Results & Output Paths
EXPLANATIONS_SHAP_PATH = RESULTS_PATH / "explanations" / "shap" / DATASET_ID
EXPLANATIONS_LIME_PATH = RESULTS_PATH / "explanations" / "lime" / DATASET_ID
METRICS_PATH = RESULTS_PATH / "metrics"
PLOTS_PATH = RESULTS_PATH / "plots"
MODELS_PATH = RESULTS_PATH / "models" / DATASET_ID


# --- Experiment Hyperparameters ---
TEST_SPLIT_SIZE = 0.2
CALIBRATION_SPLIT_SIZE = 0.1
RANDOM_STATE = 42
CARDINALITY_THRESHOLD = 1000


# --- Execution Toggles ---
RUN_SHAP = True
RUN_LIME = True


# --- Model & Metric Selection ---

# Define which models to run
SUPERVISED_MODELS = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svc']
UNSUPERVISED_MODELS = ['isolation_forest', 'ocsgd']
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
    'ocsgd': {
        'nu': 0.05, 
        'learning_rate': 'adaptive',
        'eta0': 0.01,
        'tol': 1e-3,
        'max_iter': 1000,
        'random_state': 42
    }
}