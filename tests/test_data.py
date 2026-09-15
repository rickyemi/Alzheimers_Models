"""Tests for data loading and processing modules."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loaders import load_raw_data, save_processed_data
from src.data.processors import DataProcessor

class TestDataLoading:
    """Test data loading functionality."""

    def test_load_raw_data_exists(self):
        """Test loading existing data file."""
        filepath = "data/raw/alzheimers_biomarkers.csv"
        if os.path.exists(filepath):
            df = load_raw_data(filepath)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'amyloid_positive' in df.columns

    def test_load_raw_data_missing(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")

class TestDataProcessing:
    """Test data preprocessing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'p_tau': [50, 60, np.nan, 70],
            'np_tau': [45, 55, 65, 75],
            'ab_42': [900, 800, 700, 600],
            'age': [70, 72, 68, 75],
            'years_education': [14, 16, 12, 15],
            'mmse': [25, 22, 28, 19],
            'amyloid_positive': [0, 1, 0, 1]
        })

    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        processor = DataProcessor()
        df_clean = processor.handle_missing_values(sample_data)

        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) == 3  # One row removed

    def test_normalize_features(self, sample_data):
        """Test feature normalization."""
        processor = DataProcessor()

        # Remove missing values first
        df_clean = processor.handle_missing_values(sample_data)

        X = df_clean[['p_tau', 'np_tau', 'ab_42', 'age', 'years_education', 'mmse']]
        X_scaled, _ = processor.normalize_features(X, fit=True)

        # Check shape
        assert X_scaled.shape == X.shape

        # Check standardization (mean ≈ 0, std ≈ 1)
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10
        assert np.abs(X_scaled.std(axis=0) - 1).max() < 1e-10

    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        processor = DataProcessor()
        df, stats = processor.detect_outliers(sample_data)

        assert isinstance(stats, dict)
        assert 'p_tau' in stats
        assert 'lower_bound' in stats['p_tau']
        assert 'upper_bound' in stats['p_tau']

class TestFeatureColumns:
    """Test that required columns exist."""

    def test_required_columns(self):
        """Test that raw data has required columns."""
        filepath = "data/raw/alzheimers_biomarkers.csv"
        if os.path.exists(filepath):
            df = load_raw_data(filepath)
            required_cols = ['p_tau', 'np_tau', 'ab_42', 'age', 'years_education', 'mmse', 'amyloid_positive']
            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
