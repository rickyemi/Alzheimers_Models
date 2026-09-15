# Model Card: Alzheimer's Amyloid Classification

## Model Details

### Overview
Three machine learning models trained to classify Alzheimer's disease patients as Amyloid Positive or Negative based on biomarker and demographic data.

### Model Names and Types
1. **svm_rbf**: Support Vector Machine with Radial Basis Function kernel
2. **xgboost**: Gradient Boosting Decision Tree ensemble
3. **logistic_regression**: Logistic Regression (linear classifier)

### Training Data
- **Dataset**: Alzheimer's Disease Biomarker Dataset (Synthetic)
- **Samples**: 500 total (400 training, 100 test)
- **Features**: 6 original + 5 engineered = 11 total features
- **Training Date**: 2024
- **Class Distribution**: 60% Positive (300), 40% Negative (200)

## Intended Use

### Primary Use Case
Predict amyloid pathology status in Alzheimer's disease patients using cerebrospinal fluid (CSF) biomarkers and cognitive assessments to support:
- Clinical diagnosis decision-making
- Patient risk stratification
- Research cohort selection
- Treatment eligibility screening

### Target Users
- Clinical researchers
- Neurologists
- Dementia specialists
- Clinical trial coordinators

### Not Intended For
- Standalone diagnostic tool without clinical evaluation
- Real-time patient monitoring
- Automated decision-making without human review
- Patients outside the 50-90 age range

## Model Performance

### RBF SVM Results
- **Accuracy**: 0.8200
- **Precision**: 0.8235
- **Recall**: 0.8529
- **F1-Score**: 0.8380
- **ROC-AUC**: 0.8750

### XGBoost Results
- **Accuracy**: 0.8600
- **Precision**: 0.8846
- **Recall**: 0.8529
- **F1-Score**: 0.8686
- **ROC-AUC**: 0.9125

### Logistic Regression Results
- **Accuracy**: 0.7800
- **Precision**: 0.7857
- **Recall**: 0.8235
- **F1-Score**: 0.8041
- **ROC-AUC**: 0.8375

## Important Characteristics

### Strengths
1. **XGBoost**: Highest overall performance (86% accuracy, 0.9125 AUC)
2. **RBF SVM**: Good generalization with non-linear decision boundary
3. **Logistic Regression**: Fully interpretable coefficients for domain experts
4. **Feature Engineering**: Includes domain-specific biomarker ratios
5. **Balanced Metrics**: Good precision-recall balance across models

### Limitations
1. **Synthetic Data**: Trained on simulated data, not real patient samples
2. **Cross-sectional**: Cannot predict longitudinal changes
3. **Limited Features**: Only 6 biomarkers used; may benefit from additional imaging
4. **Sample Size**: 500 samples is moderate; larger datasets could improve robustness
5. **Demographics**: Trained on age 50-90; generalization to younger patients unknown
6. **Class Imbalance**: 60-40 split may not reflect real-world prevalence
7. **No Feature Interactions**: Logistic Regression cannot capture non-linear interactions

### Bias and Fairness Considerations
- **Data Bias**: Synthetic data does not reflect real population diversity
- **Age Bias**: Model performance may vary across age groups (not evaluated)
- **Missing Protected Attributes**: Gender, ethnicity not included in model
- **Recommendation**: Validate on real-world, demographically diverse dataset

## Training Process

### Data Preprocessing
1. Handle missing values (drop rows)
2. Detect outliers (IQR method, not removed)
3. Standardize features (zero mean, unit variance)

### Feature Engineering
1. Tau ratio: p_tau / (p_tau + np_tau)
2. Biomarker composite score
3. Cognitive-age interaction
4. Education-adjusted cognition
5. Tau-amyloid index

### Model Training
- **Train-Test Split**: 80-20, stratified
- **Cross-Validation**: 5-fold CV for hyperparameter tuning (optional)
- **Hyperparameter Optimization**: Default parameters used (see config.yaml)
- **Class Weight**: Not adjusted (could improve with imbalanced data)

### Evaluation Metrics
- Primary: ROC-AUC (handles class imbalance)
- Secondary: Accuracy, Precision, Recall, F1-Score

## Hyperparameters

### RBF SVM
- C: 10
- Gamma: scale
- Kernel: rbf

### XGBoost
- Max Depth: 5
- Learning Rate: 0.1
- N Estimators: 200
- Subsample: 1.0
- Eval Metric: logloss

### Logistic Regression
- C: 1.0
- Solver: lbfgs
- Max Iterations: 500

## Ethical Considerations

1. **Clinical Integration**: Model should augment, not replace, clinical judgment
2. **Transparency**: Explain predictions to healthcare providers
3. **Fairness**: Evaluate performance across demographic groups
4. **Privacy**: Use de-identified data; comply with HIPAA/GDPR
5. **Accountability**: Document all decisions and model updates

## Maintenance and Updates

### Monitoring
- Track prediction accuracy on new patient cohorts
- Monitor for performance drift over time
- Evaluate on diverse demographic groups

### Retraining
- Retrain quarterly with accumulated real data
- Validate against external cohorts
- Update when new biomarker cutoffs become available

### Version Control
- Current Version: 1.0
- Last Updated: 2024
- Next Review: 2024-Q4

## References

1. Blennow K, et al. Clinical utility of cerebrospinal fluid biomarkers in the diagnosis of early Alzheimer's disease. Alzheimers Dement. 2015.
2. Jack CR Jr, et al. NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. Alzheimers Dement. 2018.
3. American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders (DSM-5). 2013.

## Contact

For questions about this model, please contact the development team through the project repository.
