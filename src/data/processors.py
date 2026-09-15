"""Data processing and cleaning utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Clean and preprocess Alzheimer's biomarker data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['p_tau', 'np_tau', 'ab_42', 'age', 'years_education', 'mmse']
        self.target_column = 'amyloid_positive'

    def handle_missing_values(self, df):
        """
        Handle missing values by removing rows with any NaN.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data without missing values
        """
        initial_count = len(df)
        df = df.dropna()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with missing values")
        return df

    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR method.

        Args:
            df (pd.DataFrame): Input data
            method (str): 'iqr' for interquartile range
            threshold (float): IQR multiplier for outlier detection

        Returns:
            pd.DataFrame: Data with outliers flagged
            dict: Outlier statistics
        """
        outlier_indices = set()
        stats = {}

        for col in self.feature_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.update(outliers)

            stats[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': len(outliers)
            }

        print(f"Detected {len(outlier_indices)} outlier rows")
        return df, stats

    def normalize_features(self, X_train, X_test=None, fit=True):
        """
        Standardize features to zero mean and unit variance.

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            fit (bool): Whether to fit the scaler

        Returns:
            np.ndarray: Normalized training data
            np.ndarray or None: Normalized test data
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def process(self, df, fit_scaler=True):
        """
        Apply full preprocessing pipeline.

        Args:
            df (pd.DataFrame): Raw data
            fit_scaler (bool): Whether to fit the scaler

        Returns:
            pd.DataFrame: Processed data
        """
        # Handle missing values
        df = self.handle_missing_values(df)

        # Detect and report outliers (but don't remove)
        df, outlier_stats = self.detect_outliers(df)

        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()

        # Normalize features
        X_scaled, _ = self.normalize_features(X, fit=fit_scaler)

        # Create processed dataframe
        df_processed = pd.DataFrame(X_scaled, columns=self.feature_columns)
        df_processed[self.target_column] = y.values

        return df_processed
