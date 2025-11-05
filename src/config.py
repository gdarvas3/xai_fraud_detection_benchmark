from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"

RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
EXPLANATIONS_SHAP_PATH = RESULTS_PATH / "explanations" / "shap"
EXPLANATIONS_LIME_PATH = RESULTS_PATH / "explanations" / "lime"
METRICS_PATH = RESULTS_PATH / "metrics"
PLOTS_PATH = RESULTS_PATH / "plots"
MODELS_PATH = RESULTS_PATH / "models"


TEST_SPLIT_SIZE = 5
RANDOM_STATE = 42

MODELS_TO_RUN = ['logistic_regression', 'random_forest', 'gradient_boosting']
METRICS = ['accuracy', 'f1_score', 'roc_auc']

MODEL_CONFIGS = {
    'logistic_regression': {
        'C': 1.0,
        'solver': 'liblinear'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.1
    }
}