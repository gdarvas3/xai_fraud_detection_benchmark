from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(model_name, params):
    """
    Visszaad egy inicializált modellt a neve és paraméterei alapján.
    """
    if model_name == 'logistic_regression':
        return LogisticRegression(**params)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_name == 'gradient_boosting':
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Model not found in model list: {model_name}")