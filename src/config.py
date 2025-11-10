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


TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

RUN_SHAP = True
RUN_LIME = True

SUPERVISED_MODELS = ['logistic_regression', 'random_forest']
UNSUPERVISED_MODELS = ['lof']
MODELS_TO_RUN = SUPERVISED_MODELS + UNSUPERVISED_MODELS

METRICS = ['accuracy', 'f1_score', 'pr_auc']

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
    },
    'lof':{
        'novelty': True
    }
}