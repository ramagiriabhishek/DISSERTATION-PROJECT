# Dissertation Writing Pack (MSc, UK)  
**Title:** *Explainable Artificial Intelligence for Customer Churn Prediction: A Data-Driven and Ethical Machine Learning Study*

This document provides dissertation-ready text and structure aligned with the notebook:
`notebooks/churn_xai_dissertation.ipynb`.

---

## 1) Methodology Chapter Draft (Dissertation-Ready)

### 1.1 Research Design
This study adopts a quantitative, data-driven machine learning design to investigate customer churn prediction in a banking context. The research uses a secondary synthetic dataset and evaluates multiple supervised classification algorithms. In addition to predictive performance, the study explicitly examines model explainability through feature importance analysis and SHAP (SHapley Additive exPlanations), supporting transparency and responsible AI deployment.

### 1.2 Data Source and Characteristics
The dataset used in this study is a synthetic and anonymised bank customer churn dataset in CSV format. It includes variables commonly associated with customer behaviour and financial profile, including credit score, geography, gender, age, tenure, account balance, number of products, credit card status, active membership status, estimated salary, and churn outcome.  

Because the dataset is synthetic and contains no personally identifiable information, direct privacy and confidentiality risks are substantially reduced. However, ethical reflection remains necessary, as predictive models can still encode indirect bias through feature relationships.

### 1.3 Data Preparation and Preprocessing
Data preparation was implemented in Python using pandas and scikit-learn. Initial checks included dataset dimensions, data types, and missing value profiling. The target variable (churn/exited) was isolated from predictor variables, and identifier-like attributes (e.g., customer ID fields) were excluded from modelling to prevent non-informative leakage.

Preprocessing was handled through a unified scikit-learn `Pipeline` and `ColumnTransformer` framework:
- **Numeric features:** median imputation followed by standardisation (`StandardScaler`);
- **Categorical features:** most-frequent imputation followed by one-hot encoding (`OneHotEncoder` with `handle_unknown='ignore'`).

This pipeline-based design ensured consistent preprocessing across all models and prevented data leakage by fitting transformations only on training data.

### 1.4 Train-Test Strategy
The dataset was partitioned into training and test subsets using an 80:20 split. Stratified sampling was applied to preserve the churn class distribution in both subsets. A fixed random seed (`random_state = 42`) was used to improve reproducibility.

### 1.5 Model Development
Three baseline supervised classification models were selected:
1. Logistic Regression  
2. Random Forest  
3. Support Vector Machine (RBF kernel)

These models were chosen to provide methodological breadth: a linear probabilistic model (Logistic Regression), a non-linear ensemble method (Random Forest), and a margin-based kernel classifier (SVM). The comparison enables evaluation of whether non-linear learners provide superior churn discrimination.

To strengthen the analysis, advanced models were also included:
- Gradient Boosting Classifier  
- HistGradientBoosting Classifier  
- XGBoost (where available in environment)

### 1.6 Evaluation Framework
Model performance was assessed on the held-out test set using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC curve and ROC-AUC

Using multiple metrics provides a balanced assessment, particularly in churn problems where class distributions can be uneven and false negatives may carry substantial business cost.

### 1.7 Explainability Framework
Explainability was implemented in two complementary forms:
1. **Global feature importance** from Random Forest;
2. **SHAP analysis** for global feature contribution ranking and directional influence patterns.

SHAP was selected because it offers theoretically grounded additive attributions based on Shapley values from cooperative game theory, improving interpretability and auditability.

### 1.8 Ethical Considerations
Although the dataset is synthetic and anonymised, ethical AI principles remain central. The study addresses:
- Transparency (clear feature influence reporting),
- Accountability (documented modelling and evaluation process),
- Fairness risk awareness (potential indirect bias via proxy features),
- Governance recommendations (monitoring, periodic review, and communication of model limitations).

---

## 2) Results Section Draft with Placeholders

### 2.1 Descriptive Findings
Initial exploratory analysis showed the distribution of churn outcomes and key predictor variables. Class distribution is presented in **Figure R1**. Numeric feature distributions and central tendencies are shown in **Figure R2**, while pairwise linear associations among numeric variables are visualised in **Figure R3**.

**[Insert Figure R1: Target Class Distribution]**  
**[Insert Figure R2: Numeric Feature Distributions]**  
**[Insert Figure R3: Correlation Heatmap]**

### 2.2 Baseline Model Performance
The three baseline classifiers were evaluated on the held-out test set. Comparative results are summarised in **Table R1**, with ROC performance shown in **Figure R4** and error structure visualised via confusion matrices in **Figure R5**.

**[Insert Table R1: Baseline Model Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)]**  
**[Insert Figure R4: ROC Curves for Baseline Models]**  
**[Insert Figure R5: Confusion Matrices for Baseline Models]**

The best-performing baseline model was **[MODEL_NAME]**, which achieved the highest **[PRIMARY_METRIC]** of **[VALUE]**. This suggests that **[BRIEF INTERPRETATION, e.g., non-linear interactions were important]**.

### 2.3 Explainability Results
Random Forest feature importance identified the most influential predictors of churn (see **Figure R6**). SHAP global summary outputs further showed both magnitude and direction of feature effects (see **Figure R7** and **Figure R8**).

**[Insert Figure R6: Random Forest Feature Importance (Top 20)]**  
**[Insert Figure R7: SHAP Summary Bar Plot]**  
**[Insert Figure R8: SHAP Beeswarm Plot]**

The most influential features were **[FEATURE_1]**, **[FEATURE_2]**, and **[FEATURE_3]**, indicating that churn risk is primarily associated with **[INTERPRETIVE STATEMENT]**.

### 2.4 Advanced Model Comparison
Advanced boosting models were evaluated and compared with baseline methods in **Table R2**.

**[Insert Table R2: All Model Metrics (Baseline + Advanced)]**

Where performance gains were observed, these indicate that boosted ensembles captured additional non-linear patterns not fully represented by simpler classifiers.

---

## 3) Evaluation Metrics (Dissertation-Ready Explanation)

- **Accuracy:** proportion of correctly classified observations across all classes. Useful as an overall indicator but can be misleading when classes are imbalanced.  
- **Precision:** among predicted churn cases, the proportion that are truly churners. Important when minimising false alarms is a priority.  
- **Recall (Sensitivity):** among actual churners, the proportion correctly identified. Critical when missed churners are costly.  
- **F1-score:** harmonic mean of precision and recall, providing a balanced measure where both error types matter.  
- **Confusion Matrix:** tabulates true positives, true negatives, false positives, and false negatives, enabling practical error analysis.  
- **ROC-AUC:** evaluates discrimination across all classification thresholds; higher AUC indicates better ranking ability between churn and non-churn customers.

---

## 4) SHAP Results (How to Explain in Dissertation)

### 4.1 What SHAP Shows
SHAP quantifies the contribution of each feature to individual predictions and to model behaviour overall. Positive SHAP values indicate movement towards the churn class, while negative values indicate movement away from churn (depending on class encoding).

### 4.2 How to Report SHAP Findings
In the results narrative:
1. Identify top global drivers from SHAP bar summary;
2. Explain directionality from beeswarm patterns (high vs low feature values);
3. Link findings to domain logic (e.g., low activity or product engagement increasing churn risk);
4. Discuss limitations (association does not imply causation).

### 4.3 Example Academic Wording
“SHAP analysis indicated that feature contributions were not uniform across customers. While age and account activity were globally influential, their local effects varied by profile, supporting the use of post-hoc explainability to complement aggregate performance metrics. These findings enhance model transparency and improve the interpretability of retention-related decisions.”

---

## 5) Advanced Models: Justification + Code Context

### 5.1 Suggested Models
1. **Gradient Boosting Classifier**  
2. **HistGradientBoosting Classifier**  
3. **XGBoost** (if available)

### 5.2 Why They Can Improve the Study
- Strong performance on structured/tabular data;
- Better capture of non-linear interactions and threshold effects;
- Often improved recall/F1/AUC in churn tasks;
- Provide a stronger comparative benchmark for academic rigour.

### 5.3 Academic Justification (Dissertation Language)
“Advanced boosting methods were introduced to test whether sequential ensemble learning improves classification quality beyond baseline models. Their inclusion strengthens methodological robustness and supports a more comprehensive empirical comparison.”

### 5.4 Code Location
Implementation is included in notebook Section G (`11. ADVANCED MODELS IMPLEMENTATION`).

---

## 6) Best Practices: Figures, Naming, and Output Organisation

### 6.1 Figure Saving for Dissertation
- Use high resolution (`dpi=300`) for print clarity.
- Use consistent dimensions and readable axis labels.
- Save in PNG for compatibility and quality.
- Keep visual style consistent (font size, palette, line widths).

### 6.2 Naming Conventions
Use deterministic, descriptive names:
- Figures: `fig_<topic>_<model/variant>.png`
- Tables: `table_<topic>_<scope>.csv`

Examples:
- `fig_roc_curves_baseline.png`
- `fig_shap_summary_beeswarm_top20.png`
- `table_model_metrics_all.csv`

### 6.3 Folder Structure
- `notebooks/` → analysis notebooks  
- `outputs/figures/` → dissertation figures  
- `outputs/tables/` → metric and summary tables  
- `docs/` → narrative text for dissertation chapters

### 6.4 Dissertation Insertion Workflow
1. Generate outputs by running notebook end-to-end;
2. Select final figures/tables from `outputs/`;
3. Insert in dissertation with chapter-specific numbering (e.g., Figure 4.1, Table 5.2);
4. Add concise captions and explicit in-text references.

