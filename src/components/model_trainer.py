import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts", "model.pkl"
    )
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into Training and test set")
            X_train, y_train, = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=42
                )
            }
            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 10, 20],
                    "min_samples_leaf": [1, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 8, 12],
                    "min_samples_leaf": [1, 5, 10],
                    "max_features": ["sqrt", "log2"]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 150, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 4],
                    "subsample": [0.7, 0.8, 0.9]
                },
                "KNN": {
                    "n_neighbors": [5, 10, 15, 20],
                    "weights": ["uniform", "distance"]
                },
                "Naive Bayes": {
        # GaussianNB has no major tunable hyperparameters
    },

    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"]
    }
        }
            model_report, best_estimators = evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                param = params
            )
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_estimators[best_model_name]

            logging.info(f"Best model: {best_model_name} | ROC_AUC: {best_model_score}")
            
            if best_model_score <0.70 :
                raise CustomException("No acceptable Model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_prob)

            return best_model_name, roc_auc
        except Exception as e:
            raise CustomException(e, sys)
        