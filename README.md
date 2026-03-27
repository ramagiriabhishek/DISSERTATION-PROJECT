# DISSERTATION-PROJECT

This repository contains a full MSc-level churn prediction and explainable AI workflow for the dissertation topic:

**“Explainable Artificial Intelligence for Customer Churn Prediction: A Data-Driven and Ethical Machine Learning Study.”**

## Repository Structure

- `notebooks/churn_xai_dissertation.ipynb`  
  Complete notebook with:
  - Data loading
  - EDA
  - Preprocessing
  - Train-test split
  - Baseline modelling (Logistic Regression, Random Forest, SVM)
  - Evaluation metrics and plots
  - SHAP explainability
  - Advanced models (Gradient Boosting, HistGradientBoosting, XGBoost if available)

- `docs/dissertation_writing_pack.md`  
  Dissertation-ready writing support:
  - Methodology draft
  - Results draft with figure/table placeholders
  - Evaluation metric explanations
  - SHAP write-up guidance
  - Advanced model academic justification
  - Figure/table organisation best practices

- `outputs/figures/`  
  Saved dissertation figures (generated when notebook runs).

- `outputs/tables/`  
  Saved dissertation tables (generated when notebook runs).

## How to Run

1. Place your churn CSV file in the project root as:
   - `churn.csv`
2. Open and run:
   - `notebooks/churn_xai_dissertation.ipynb`
3. Update `DATA_PATH` in the notebook if your CSV uses a different name/path.

## Core Python Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap
- xgboost (optional, for advanced extension)
