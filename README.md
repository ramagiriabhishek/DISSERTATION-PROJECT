# Explainable Artificial Intelligence for Customer Churn Prediction

**A Data-Driven and Ethical Machine Learning Study**

MSc Data Science Dissertation — COM 752

---

## Project Overview

This project implements a complete machine learning pipeline for predicting customer churn in banking, with a focus on Explainable AI (XAI) using SHAP. The analysis combines predictive modelling with model interpretation to support transparent and ethical decision-making.

## Repository Structure

```
├── Bank Customer Churn Prediction.csv    # Synthetic bank customer dataset (10,000 records)
├── Customer_Churn_XAI_Analysis.ipynb     # Main analysis notebook (fully executed)
├── Dissertation_Writing_Guide.md         # Dissertation-ready academic writing
├── figures/                              # All generated figures (300 DPI)
│   ├── fig1_churn_distribution.png
│   ├── fig2_feature_distributions.png
│   ├── fig3_categorical_churn_rates.png
│   ├── fig4_correlation_heatmap.png
│   ├── fig5_boxplots.png
│   ├── fig6_products_churn.png
│   ├── fig7_model_comparison.png
│   ├── fig8_confusion_matrices.png
│   ├── fig9_roc_curves.png
│   ├── fig10_feature_importance.png
│   ├── fig11_shap_rf_summary.png
│   ├── fig12_shap_rf_bar.png
│   ├── fig13_shap_xgb_summary.png
│   ├── fig14_shap_xgb_bar.png
│   ├── fig15_shap_lgbm_summary.png
│   ├── fig16_shap_dependence.png
│   └── fig17_shap_waterfall.png
├── outputs/
│   └── model_comparison.csv              # Performance metrics table
└── README.md
```

## Models Implemented

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8375 | 0.5915 | 0.6511 | 0.6199 | 0.8639 |
| XGBoost | 0.8225 | 0.5524 | 0.6732 | 0.6069 | 0.8469 |
| LightGBM | 0.8150 | 0.5361 | 0.6757 | 0.5978 | 0.8501 |
| SVM | 0.7865 | 0.4838 | 0.7346 | 0.5834 | 0.8496 |
| Logistic Regression | 0.7135 | 0.3872 | 0.7002 | 0.4987 | 0.7771 |

## Key Technologies

- Python 3, pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (Logistic Regression, Random Forest, SVM)
- XGBoost, LightGBM
- SHAP (Explainable AI)

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap xgboost lightgbm jupyter
jupyter notebook Customer_Churn_XAI_Analysis.ipynb
```
