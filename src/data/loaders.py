"""Data loading utilities for Alzheimer's classification."""

import pandas as pd
import os

def load_raw_data(filepath):
    """
    Load raw biomarker data from CSV.

    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    return df

def load_processed_data(filepath):
    """
    Load processed and feature-engineered data.

    Args:
        filepath (str): Path to processed CSV file

    Returns:
        pd.DataFrame: Processed data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} processed samples from {filepath}")
    return df

def save_processed_data(df, filepath):
    """
    Save processed data to CSV.

    Args:
        df (pd.DataFrame): Data to save
        filepath (str): Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} samples to {filepath}")
