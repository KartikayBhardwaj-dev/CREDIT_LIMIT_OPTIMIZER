import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    # data preprocessing
    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                "SEX",
                "EDUCATION",
                "MARRIAGE"
            ]
            numerical_columns = [
                "LIMIT_BAL",
                "AGE",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
            ]
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler())
                ]
            )
            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipe, numerical_columns),
                    ("cat_pipeline", cat_pipe, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

# applying data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test dataset loaded")
            target_column = "default.payment.next.month"
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Applying preprocessing")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)