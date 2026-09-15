"""
End-to-end training pipeline for Alzheimer's classification.
This script orchestrates data loading, preprocessing, feature engineering, model training, and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loaders import load_raw_data, save_processed_data
from src.data.processors import DataProcessor
from src.data.features import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluation import ModelEvaluator

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Execute the complete training pipeline."""

    print("\n" + "=" * 80)
    print("ALZHEIMER'S DISEASE CLASSIFICATION MODEL TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # Load configuration
    config = load_config()
    print("✓ Configuration loaded")

    # Create output directories
    os.makedirs(config['models']['save_directory'], exist_ok=True)
    os.makedirs(config['results']['metrics_directory'], exist_ok=True)
    os.makedirs(config['results']['predictions_directory'], exist_ok=True)
    os.makedirs(config['results']['plots_directory'], exist_ok=True)

    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print("\n[STEP 1] Loading Raw Data")
    print("-" * 80)
    df_raw = load_raw_data(config['data']['raw_data_path'])
    print(f"Data shape: {df_raw.shape}")
    print(f"Target distribution:\n{df_raw['amyloid_positive'].value_counts()}\n")

    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n[STEP 2] Data Preprocessing")
    print("-" * 80)
    processor = DataProcessor()
    df_processed = processor.process(df_raw, fit_scaler=True)
    print(f"Processed data shape: {df_processed.shape}")
    save_processed_data(df_processed, config['data']['processed_data_path'])

    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n[STEP 3] Feature Engineering")
    print("-" * 80)
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df_processed)
    save_processed_data(df_engineered, config['data']['features_engineered_path'])
    print(f"Engineered data shape: {df_engineered.shape}")

    # ========================================================================
    # STEP 4: TRAIN-TEST SPLIT
    # ========================================================================
    print("\n[STEP 4] Train-Test Split")
    print("-" * 80)

    # Use engineered features for training
    feature_columns = [col for col in df_engineered.columns if col != 'amyloid_positive']
    X = df_engineered[feature_columns].values
    y = df_engineered['amyloid_positive'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['train_test_split']['test_size'],
        random_state=config['train_test_split']['random_state'],
        stratify=y if config['train_test_split']['stratify'] else None
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test set target distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # ========================================================================
    # STEP 5: MODEL TRAINING
    # ========================================================================
    print("\n[STEP 5] Model Training")
    print("-" * 80)

    trainer = ModelTrainer(random_state=config['training']['random_state'])
    evaluator = ModelEvaluator()
    results = {}

    # Train RBF SVM
    print("\nTraining RBF SVM...")
    svm_model, svm_params = trainer.train_svm_rbf(
        X_train, y_train,
        hyperparameter_tuning=config['training']['hyperparameter_tuning']
    )

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model, xgb_params = trainer.train_xgboost(
        X_train, y_train,
        hyperparameter_tuning=config['training']['hyperparameter_tuning']
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model, lr_params = trainer.train_logistic_regression(
        X_train, y_train,
        hyperparameter_tuning=config['training']['hyperparameter_tuning']
    )

    print("✓ All models trained successfully")

    # ========================================================================
    # STEP 6: MODEL EVALUATION
    # ========================================================================
    print("\n[STEP 6] Model Evaluation")
    print("-" * 80)

    # Evaluate each model
    for model_name in ['svm_rbf', 'xgboost', 'logistic_regression']:
        model = trainer.models[model_name]

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Probabilities (for ROC-AUC)
        try:
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba_train = model.decision_function(X_train)
            y_pred_proba_test = model.decision_function(X_test)

        # Calculate metrics
        train_metrics = evaluator.calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred_test, y_pred_proba_test)

        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'parameters': trainer.best_params[model_name]
        }

        # Print results
        print(f"\n{model_name.upper().replace('_', ' ')}")
        evaluator.print_metrics(train_metrics, f"{model_name} (Training)")
        evaluator.print_metrics(test_metrics, f"{model_name} (Test)")
        evaluator.print_classification_report(y_test, y_pred_test, model_name)

        # Save metrics
        metrics_path = os.path.join(
            config['results']['metrics_directory'],
            f"{model_name}_metrics.json"
        )
        evaluator.save_metrics(test_metrics, metrics_path)

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred_test,
            'predicted_probability': y_pred_proba_test
        })
        predictions_path = os.path.join(
            config['results']['predictions_directory'],
            f"{model_name}_predictions.csv"
        )
        predictions_df.to_csv(predictions_path, index=False)

        # Save plots
        if config['evaluation']['save_plots']:
            cm_path = os.path.join(
                config['results']['plots_directory'],
                f"{model_name}_confusion_matrix.png"
            )
            evaluator.plot_confusion_matrix(y_test, y_pred_test, model_name, cm_path)

            roc_path = os.path.join(
                config['results']['plots_directory'],
                f"{model_name}_roc_curve.png"
            )
            evaluator.plot_roc_curve(y_test, y_pred_proba_test, model_name, roc_path)

        # Save model
        model_path = os.path.join(
            config['models']['save_directory'],
            f"{model_name}.pkl"
        )
        trainer.save_model(model_name, model_path)

        # Save metadata
        metadata_path = os.path.join(
            config['models']['save_directory'],
            f"{model_name}_metadata.json"
        )
        trainer.save_model_metadata(model_name, metadata_path)

    # ========================================================================
    # STEP 7: MODEL COMPARISON
    # ========================================================================
    print("\n[STEP 7] Model Comparison")
    print("-" * 80)

    comparison_data = {
        model_name: data['test']
        for model_name, data in results.items()
    }
    evaluator.compare_models(comparison_data)

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nArtifacts saved to:")
    print(f"  Models: {config['models']['save_directory']}")
    print(f"  Metrics: {config['results']['metrics_directory']}")
    print(f"  Predictions: {config['results']['predictions_directory']}")
    print(f"  Plots: {config['results']['plots_directory']}")
    print()

if __name__ == "__main__":
    main()
