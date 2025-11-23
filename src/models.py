# src/models.py

# --- Model Class Imports ---

# Scikit-learn Supervised
from sklearn.linear_model import LogisticRegression, SGDOneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

# Scikit-learn Unsupervised
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# External Libraries
from xgboost import XGBClassifier

# --- Model Factory Function ---

def get_model(model_name, params):
    """
    Returns an intialized model.
    """
    
    # Supervised: Logistic Regression
    if model_name == 'logistic_regression':
        return LogisticRegression(**params)
    
    # Supervised: Random Forest
    elif model_name == 'random_forest':
        return RandomForestClassifier(**params)
    
    # Supervised: Gradient Boosting (XGBoost)
    elif model_name == 'gradient_boosting':
        return XGBClassifier(**params)
    
    # Supervised: Support Vector Classifier
    elif model_name == 'svc':
        return LinearSVC(**params)
    
    # Unsupervised: Local Outlier Factor
    elif model_name == 'lof':
        return LocalOutlierFactor(**params)
    
    # Unsupervised: Isolation Forest
    elif model_name == 'isolation_forest':
        return IsolationForest(**params)
    
    # Unsupervised: One-Class SVM
    elif model_name == 'ocsgd':
        return SGDOneClassSVM(**params)
    
    # Error handling for unknown model
    else:
        raise ValueError(f"Model not found in model list: {model_name}")