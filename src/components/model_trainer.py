import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier":AdaBoostRegressor()
            }
            
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],"
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best', 'random'],
                    # 'max_features':['auto', 'sqrt', 'log2']
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth':[3,5,7,9,11,13,15]
                },
                "Linear Regression":{},
                "K-Neighbors":{
                    'n_neighbors':[5,7,9,11,13,15],
                    # 'weights':['uniform','distance'],
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth':[3,5,7,9,11,13,15]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                    # 'loss_function': ['RMSE', 'MAE']
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                    # 'loss_function': ['RMSE', 'MAE']
                },  
            }
            
            model_report:dict=evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            # To get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # To get best model from models
            best_model=models[best_model_name]
            
            if(best_model_score<0.6):
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            
            # Get Predicted Score from best model
            predicted=best_model.predict(x_test)
            score=r2_score(y_test,predicted)
            
            return score
            
        except Exception as e:
            raise CustomException(e,sys)
