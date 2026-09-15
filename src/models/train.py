"""Model training utilities for Alzheimer's classification."""

import pickle
import json
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """Train multiple classifiers for Alzheimer's prediction."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}

    def train_svm_rbf(self, X_train, y_train, hyperparameter_tuning=False):
        """
        Train RBF SVM classifier.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            hyperparameter_tuning (bool): Whether to perform grid search

        Returns:
            SVC: Trained model
            dict: Best parameters and CV results
        """
        if hyperparameter_tuning:
            print("Training RBF SVM with hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf']
            }

            svm = SVC(random_state=self.random_state)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            self.best_params['svm_rbf'] = grid_search.best_params_

            print(f"Best SVM parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            best_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                           random_state=self.random_state)
            best_model.fit(X_train, y_train)
            self.best_params['svm_rbf'] = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

        self.models['svm_rbf'] = best_model
        return best_model, self.best_params['svm_rbf']

    def train_xgboost(self, X_train, y_train, hyperparameter_tuning=False):
        """
        Train XGBoost classifier.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            hyperparameter_tuning (bool): Whether to perform grid search

        Returns:
            XGBClassifier: Trained model
            dict: Best parameters and CV results
        """
        if hyperparameter_tuning:
            print("Training XGBoost with hyperparameter tuning...")
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'subsample': [0.8, 0.9, 1.0]
            }

            xgb = XGBClassifier(random_state=self.random_state, use_label_encoder=False,
                              eval_metric='logloss')
            grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            self.best_params['xgboost'] = grid_search.best_params_

            print(f"Best XGBoost parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            best_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200,
                                      random_state=self.random_state, use_label_encoder=False,
                                      eval_metric='logloss')
            best_model.fit(X_train, y_train)
            self.best_params['xgboost'] = {'max_depth': 5, 'learning_rate': 0.1,
                                          'n_estimators': 200, 'subsample': 1.0}

        self.models['xgboost'] = best_model
        return best_model, self.best_params['xgboost']

    def train_logistic_regression(self, X_train, y_train, hyperparameter_tuning=False):
        """
        Train Logistic Regression classifier.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            hyperparameter_tuning (bool): Whether to perform grid search

        Returns:
            LogisticRegression: Trained model
            dict: Best parameters and CV results
        """
        if hyperparameter_tuning:
            print("Training Logistic Regression with hyperparameter tuning...")
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [100, 200, 500]
            }

            lr = LogisticRegression(random_state=self.random_state)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_

            print(f"Best LR parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            best_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500,
                                           random_state=self.random_state)
            best_model.fit(X_train, y_train)
            self.best_params['logistic_regression'] = {'C': 1.0, 'solver': 'lbfgs',
                                                      'max_iter': 500}

        self.models['logistic_regression'] = best_model
        return best_model, self.best_params['logistic_regression']

    def save_model(self, model_name, filepath):
        """
        Save trained model to disk.

        Args:
            model_name (str): Name of the model
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")

        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)

        print(f"Saved {model_name} to {filepath}")

    def save_model_metadata(self, model_name, metadata_filepath):
        """
        Save model metadata (training date, parameters, etc.).

        Args:
            model_name (str): Name of the model
            metadata_filepath (str): Path to save metadata
        """
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'parameters': self.best_params.get(model_name, {}),
            'model_type': type(self.models[model_name]).__name__
        }

        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {model_name} metadata to {metadata_filepath}")
