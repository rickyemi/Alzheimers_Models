# Data Dictionary: Alzheimer's Disease Biomarker Dataset

## Overview
Dataset contains 500 patient samples with biomarker measurements and cognitive assessments for Alzheimer's disease classification.

## Features

### Biomarkers

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| p_tau | float | pg/mL | 10-150 | Phosphorylated tau protein levels. Elevated levels indicate tau pathology associated with neurodegeneration. Key biomarker for Alzheimer's disease. |
| np_tau | float | pg/mL | 10-150 | Non-phosphorylated tau protein levels. Represents total tau levels. Used to assess overall tau metabolism. |
| ab_42 | float | pg/mL | 500-1500 | Amyloid beta 1-42 levels. Lower levels indicate amyloid aggregation in the brain. Inverse relationship with amyloid pathology. |

### Demographics

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| age | float | years | 50-90 | Age of the patient at assessment. Important risk factor for Alzheimer's disease. |
| years_education | float | years | 8-20 | Years of formal education completed. Associated with cognitive reserve. |

### Cognitive Assessment

| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| mmse | float | score | 0-30 | Mini-Mental State Examination score. Higher scores indicate better cognitive function. Scores below 24 suggest cognitive impairment. |

## Target Variable

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| amyloid_positive | int | 0, 1 | Binary classification target. 1 = Amyloid Positive (pathological), 0 = Amyloid Negative (normal). |

## Data Quality

### Missing Values
- None: All fields are complete in the dataset
- If missing values occur, they are handled by removing affected rows

### Outliers
- Detected using Interquartile Range (IQR) method with threshold of 1.5
- Reported but not removed during preprocessing
- Outliers may represent genuine pathological cases

### Data Distribution
- **Target Distribution**: 60% positive (n=300), 40% negative (n=200)
- **Imbalance**: Slight class imbalance toward positive cases
- **Stratified Split**: Train-test split maintains target distribution

## Feature Relationships

### Amyloid Positive Pattern
- Higher p_tau levels (mean ≈ 70 pg/mL)
- Higher np_tau levels (mean ≈ 65 pg/mL)
- Lower ab_42 levels (mean ≈ 750 pg/mL)
- Older age (mean ≈ 72 years)
- Lower MMSE scores (mean ≈ 20)

### Amyloid Negative Pattern
- Lower p_tau levels (mean ≈ 45 pg/mL)
- Lower np_tau levels (mean ≈ 40 pg/mL)
- Higher ab_42 levels (mean ≈ 900 pg/mL)
- Younger age (mean ≈ 68 years)
- Higher MMSE scores (mean ≈ 26)

## Engineered Features

| Column | Type | Description |
|--------|------|-------------|
| tau_ratio | float | p_tau / (p_tau + np_tau). Ratio of phosphorylated to total tau. Higher values indicate greater phosphorylation. |
| biomarker_score | float | Normalized composite score combining all three biomarkers. Ranges 0-1. |
| cognitive_age_interaction | float | age × (30 - mmse). Captures cognitive decline adjusted for age. |
| education_adjusted_cognition | float | mmse / (years_education / 12). Cognition adjusted for education level. |
| tau_amyloid_index | float | p_tau / ab_42. Key ratio for Alzheimer's pathology assessment. |

## Unit Conversions

For reference with other studies:
- pg/mL (picograms/milliliter): Standard unit for biomarker measurement
- Years: Standard unit for age and education

## Data Collection Notes

- Dataset is synthetic for demonstration purposes
- Created to mimic realistic Alzheimer's biomarker patterns
- Represents cross-sectional data (single timepoint per patient)
- No longitudinal information available

## References

- Alzheimer's Disease Neuroimaging Initiative (ADNI)
- Biomarkers for diagnosing Alzheimer's disease (Blennow et al., 2018)
- Mini-Cog: A simple screening tool for cognitive impairment
