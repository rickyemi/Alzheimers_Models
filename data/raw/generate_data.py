"""
Generate synthetic Alzheimer's disease biomarker dataset.
This script creates realistic synthetic data for demonstration purposes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_alzheimers_data(n_samples=500, random_state=42):
    """
    Generate synthetic Alzheimer's biomarker data.

    Predictors:
    - p_tau: phosphorylated tau levels (pg/mL)
    - np_tau: non-phosphorylated tau levels (pg/mL)
    - ab_42: amyloid beta 1-42 levels (pg/mL)
    - age: age in years
    - years_education: years of formal education
    - mmse: Mini-Mental State Examination score (0-30, higher is better)

    Target:
    - amyloid_positive: 1 if Amyloid positive, 0 if negative
    """
    np.random.seed(random_state)

    n_positive = int(n_samples * 0.6)  # 60% positive cases
    n_negative = n_samples - n_positive

    data = []

    # Amyloid Positive cases (higher p-tau, np-tau, lower AB42)
    for _ in range(n_positive):
        p_tau = np.random.normal(70, 25)  # Higher mean for positive
        np_tau = np.random.normal(65, 20)
        ab_42 = np.random.normal(750, 150)  # Lower AB42 for positive
        age = np.random.normal(72, 7)
        years_edu = np.random.normal(14, 3)
        mmse = np.random.normal(20, 6)  # Lower scores for positive
        mmse = np.clip(mmse, 0, 30)

        data.append([p_tau, np_tau, ab_42, age, years_edu, mmse, 1])

    # Amyloid Negative cases (lower p-tau, np-tau, higher AB42)
    for _ in range(n_negative):
        p_tau = np.random.normal(45, 20)  # Lower mean for negative
        np_tau = np.random.normal(40, 18)
        ab_42 = np.random.normal(900, 200)  # Higher AB42 for negative
        age = np.random.normal(68, 8)
        years_edu = np.random.normal(15, 3)
        mmse = np.random.normal(26, 4)  # Higher scores for negative
        mmse = np.clip(mmse, 0, 30)

        data.append([p_tau, np_tau, ab_42, age, years_edu, mmse, 0])

    df = pd.DataFrame(
        data,
        columns=['p_tau', 'np_tau', 'ab_42', 'age', 'years_education', 'mmse', 'amyloid_positive']
    )

    # Ensure values are in realistic ranges
    df['p_tau'] = np.clip(df['p_tau'], 10, 150)
    df['np_tau'] = np.clip(df['np_tau'], 10, 150)
    df['ab_42'] = np.clip(df['ab_42'], 500, 1500)
    df['age'] = np.clip(df['age'], 50, 90)
    df['years_education'] = np.clip(df['years_education'], 8, 20)

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_alzheimers_data(n_samples=500)
    df.to_csv('alzheimers_biomarkers.csv', index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"\nTarget distribution:\n{df['amyloid_positive'].value_counts()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData statistics:\n{df.describe()}")
