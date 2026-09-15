"""Feature engineering for Alzheimer's biomarker data."""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Create domain-specific features for Alzheimer's prediction."""

    @staticmethod
    def create_tau_ratio(df):
        """
        Create phosphorylated to total tau ratio.
        Higher ratio may indicate neurodegeneration.
        """
        df['tau_ratio'] = df['p_tau'] / (df['p_tau'] + df['np_tau'])
        return df

    @staticmethod
    def create_biomarker_composite(df):
        """
        Create composite biomarker score combining multiple markers.
        Normalized weighted sum of key biomarkers.
        """
        # Normalize each biomarker to 0-1 range
        p_tau_norm = (df['p_tau'] - df['p_tau'].min()) / (df['p_tau'].max() - df['p_tau'].min())
        np_tau_norm = (df['np_tau'] - df['np_tau'].min()) / (df['np_tau'].max() - df['np_tau'].min())
        ab_norm = 1 - ((df['ab_42'] - df['ab_42'].min()) / (df['ab_42'].max() - df['ab_42'].min()))

        # Weighted combination
        df['biomarker_score'] = (0.4 * p_tau_norm + 0.3 * np_tau_norm + 0.3 * ab_norm)
        return df

    @staticmethod
    def create_cognitive_age_interaction(df):
        """
        Create interaction between age and MMSE (cognitive performance).
        May capture cognitive decline adjusted for age.
        """
        df['cognitive_age_interaction'] = df['age'] * (30 - df['mmse'])
        return df

    @staticmethod
    def create_education_adjusted_cognition(df):
        """
        Create education-adjusted cognitive score.
        MMSE scores should be interpreted with education in mind.
        """
        df['education_adjusted_cognition'] = df['mmse'] / (df['years_education'] / 12.0)
        return df

    @staticmethod
    def create_tau_amyloid_index(df):
        """
        Create tau/amyloid index (phospho-tau/AB42).
        Key biomarker for Alzheimer's pathology.
        """
        df['tau_amyloid_index'] = df['p_tau'] / df['ab_42']
        return df

    @classmethod
    def engineer_features(cls, df):
        """
        Apply all feature engineering transformations.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with engineered features
        """
        df = df.copy()

        df = cls.create_tau_ratio(df)
        df = cls.create_biomarker_composite(df)
        df = cls.create_cognitive_age_interaction(df)
        df = cls.create_education_adjusted_cognition(df)
        df = cls.create_tau_amyloid_index(df)

        print(f"Engineered {5} new features")
        print(f"Final feature set shape: {df.shape}")

        return df
