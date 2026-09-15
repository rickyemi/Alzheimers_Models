"""Model evaluation and metrics utilities."""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)

class ModelEvaluator:
    """Evaluate model performance on test data."""

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities

        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """
        Print evaluation metrics in a formatted way.

        Args:
            metrics (dict): Metrics dictionary
            model_name (str): Name of the model
        """
        print(f"\n{model_name} Performance Metrics:")
        print("=" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():.<30} {value:.4f}")
        print("=" * 50)

    @staticmethod
    def print_classification_report(y_true, y_pred, model_name="Model"):
        """
        Print detailed classification report.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_true, y_pred,
                                   target_names=['Amyloid Negative', 'Amyloid Positive']))

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
        """
        Create and optionally save confusion matrix plot.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            save_path (str): Path to save the plot

        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues', aspect='auto')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.colorbar()

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        return cm

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
        """
        Create and optionally save ROC curve plot.

        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve to {save_path}")

    @staticmethod
    def save_metrics(metrics, filepath):
        """
        Save metrics to JSON file.

        Args:
            metrics (dict): Metrics dictionary
            filepath (str): Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {filepath}")

    @staticmethod
    def compare_models(results_dict):
        """
        Compare performance across multiple models.

        Args:
            results_dict (dict): Dictionary with model names as keys and metrics as values
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        # Create comparison table
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # Print header
        print(f"{'Model':<20}", end='')
        for metric in metrics_to_compare:
            print(f"{metric.upper():<12}", end='')
        print()
        print("-" * 80)

        # Print rows
        for model_name, metrics in results_dict.items():
            print(f"{model_name:<20}", end='')
            for metric in metrics_to_compare:
                value = metrics.get(metric, np.nan)
                if isinstance(value, float):
                    print(f"{value:<12.4f}", end='')
                else:
                    print(f"{'N/A':<12}", end='')
            print()

        print("=" * 80)
