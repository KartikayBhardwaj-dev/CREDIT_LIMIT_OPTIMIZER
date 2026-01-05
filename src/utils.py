import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import roc_auc_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})
            if params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    scoring="roc_auc",
                    cv=3,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob)
            report[model_name] = roc_auc
            best_estimators[model_name] = best_model
        return report, best_estimators
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def probability_to_risk_score(prob: float) -> int:
    """
    Convert Probability of Default (PD) to Risk Score (0–100)
    """
    return int(round(prob * 100))
def get_risk_category_and_limit(score):
    if score <= 40:
        return "Low Risk", 200000
    elif score <= 70:
        return "Medium Risk", 100000
    else:
        return "High Risk", 40000