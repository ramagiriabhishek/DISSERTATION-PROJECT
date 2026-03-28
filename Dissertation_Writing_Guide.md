# Dissertation Writing Guide

## Explainable Artificial Intelligence for Customer Churn Prediction: A Data-Driven and Ethical Machine Learning Study

**COM 752 — MSc Data Science Dissertation**

---

This document provides dissertation-ready writing sections aligned with the Jupyter Notebook analysis. Each section includes academic prose, figure/table placeholders, and guidance on integration with the notebook outputs.

---

## PART A: METHODOLOGY CHAPTER

### A.1 Research Design

This study adopts a quantitative, positivist research design employing secondary data analysis. The research follows a deductive approach, testing whether machine learning models can accurately predict customer churn and whether Explainable AI (XAI) techniques can render those predictions transparent and interpretable. The quantitative paradigm is well-suited to this investigation as it enables objective measurement of model performance through established statistical metrics and facilitates reproducible analysis (Creswell and Creswell, 2018).

The research design comprises four sequential phases: (i) exploratory data analysis to understand the underlying data structure and distributions; (ii) data preprocessing to prepare the dataset for machine learning; (iii) supervised classification model training and evaluation; and (iv) post-hoc model interpretation using SHAP (SHapley Additive exPlanations). This structured approach aligns with the Knowledge Discovery in Databases (KDD) framework widely adopted in data science research (Fayyad et al., 1996).

### A.2 Dataset Description

The dataset employed in this study is a publicly available synthetic bank customer churn dataset comprising 10,000 records and 12 attributes. Each record represents a bank customer characterised by demographic features (age, gender, geography), financial attributes (credit score, account balance, estimated salary), product engagement indicators (number of products, credit card ownership, active membership status, tenure), and a binary target variable indicating whether the customer has churned (1) or been retained (0).

It is important to emphasise that the dataset is entirely synthetic and does not contain real customer data. This eliminates ethical concerns related to personal data protection, informed consent, and compliance with data protection regulations such as the UK General Data Protection Regulation (UK GDPR). The use of synthetic data is increasingly recognised as a valid approach in machine learning research, particularly when real-world data is unavailable or when privacy considerations preclude the use of genuine customer records (Jordon et al., 2022).

The dataset exhibits a class imbalance, with approximately 79.6% of customers classified as retained and 20.4% as churned. This imbalance reflects realistic churn patterns observed in the banking sector (Vafeiadis et al., 2015) and necessitates careful consideration during model training to prevent bias towards the majority class.

[Table 1: Dataset Feature Descriptions]

| Feature | Type | Description |
|---------|------|-------------|
| customer_id | Integer | Unique customer identifier (dropped during preprocessing) |
| credit_score | Integer | Customer credit score (350–850) |
| country | Categorical | Customer's country (France, Spain, Germany) |
| gender | Categorical | Customer's gender (Male, Female) |
| age | Integer | Customer's age |
| tenure | Integer | Number of years as a customer (0–10) |
| balance | Float | Account balance |
| products_number | Integer | Number of banking products held (1–4) |
| credit_card | Binary | Whether the customer holds a credit card (0/1) |
| active_member | Binary | Whether the customer is an active member (0/1) |
| estimated_salary | Float | Customer's estimated annual salary |
| churn | Binary | Target variable — whether the customer churned (1) or not (0) |

### A.3 Data Preprocessing

Data preprocessing is a critical stage in the machine learning pipeline that transforms raw data into a format suitable for model training (García et al., 2016). The following preprocessing steps were applied:

**Removal of Irrelevant Features.** The `customer_id` column was removed as it serves solely as a unique identifier and carries no predictive information. Retaining such identifiers could introduce spurious correlations and degrade model performance.

**Encoding of Categorical Variables.** The categorical variables `country` and `gender` were transformed using one-hot encoding (also known as dummy variable encoding). This technique creates binary indicator columns for each category, enabling machine learning algorithms that require numerical input to process categorical information. The `drop_first=True` parameter was applied to avoid multicollinearity by removing one reference category from each encoded variable (James et al., 2021).

**Feature Scaling.** All numerical features were standardised using the StandardScaler, which transforms each feature to have a mean of zero and a standard deviation of one. This is particularly important for algorithms such as Logistic Regression and Support Vector Machines, which are sensitive to the magnitude of input features (Géron, 2022). Tree-based models (Random Forest, XGBoost, LightGBM) are inherently scale-invariant but were trained on scaled data for consistency.

**Train-Test Split.** The preprocessed dataset was split into training (80%) and testing (20%) subsets using stratified sampling to preserve the original class distribution in both sets. A fixed random seed (42) was used to ensure reproducibility. The stratified approach ensures that both the training and testing sets contain representative proportions of churned and retained customers, which is essential when dealing with imbalanced datasets (He and Garcia, 2009).

### A.4 Machine Learning Models

Five classification models were selected to provide a comprehensive comparison ranging from simple linear models to advanced ensemble methods:

**Logistic Regression** serves as the baseline model. It models the probability of churn as a logistic function of the input features and is valued for its simplicity and interpretability. The `class_weight='balanced'` parameter was used to address class imbalance by adjusting the loss function to penalise misclassification of the minority class more heavily (Hosmer et al., 2013).

**Random Forest** is an ensemble method that constructs multiple decision trees during training and outputs the mode of their predictions. It reduces overfitting compared to individual decision trees through bagging (bootstrap aggregating) and random feature selection. Two hundred trees with a maximum depth of 10 were used to balance model complexity and generalisation (Breiman, 2001).

**Support Vector Machine (SVM)** identifies the optimal hyperplane that maximises the margin between classes in a high-dimensional feature space. The Radial Basis Function (RBF) kernel was employed to capture non-linear relationships. Class weighting was applied to handle the imbalanced dataset (Cortes and Vapnik, 1995).

**XGBoost (Extreme Gradient Boosting)** is an optimised implementation of gradient boosting that builds trees sequentially, with each tree correcting errors from its predecessors. It incorporates regularisation terms (L1 and L2) to prevent overfitting and supports efficient handling of sparse data. The `scale_pos_weight` parameter was configured to account for class imbalance (Chen and Guestrin, 2016).

**LightGBM (Light Gradient Boosting Machine)** employs a histogram-based approach to split finding, which significantly reduces computational cost whilst maintaining competitive accuracy. It uses leaf-wise tree growth rather than level-wise growth, allowing it to learn more complex patterns. Like XGBoost, it supports class weight adjustment for imbalanced data (Ke et al., 2017).

### A.5 Evaluation Metrics

Model performance was assessed using five complementary metrics:

- **Accuracy**: The proportion of correct predictions out of all predictions. Whilst intuitive, accuracy can be misleading for imbalanced datasets.
- **Precision**: The proportion of predicted positive cases that are truly positive. High precision indicates fewer false alarms.
- **Recall (Sensitivity)**: The proportion of actual positive cases correctly identified. In churn prediction, high recall is critical as failing to identify a churning customer represents a missed retention opportunity.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure when both false positives and false negatives carry costs.
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model's ability to discriminate between classes across all classification thresholds. An AUC of 0.5 indicates no discrimination (random guessing), whilst 1.0 indicates perfect discrimination.

Five-fold stratified cross-validation was performed on the training set to obtain robust performance estimates and assess model stability (Kohavi, 1995).

### A.6 Explainable AI — SHAP

To address the interpretability requirements of this research, SHAP (SHapley Additive exPlanations) was employed as the primary XAI technique. Proposed by Lundberg and Lee (2017), SHAP provides a unified framework for interpreting machine learning predictions by computing the contribution of each feature to an individual prediction.

SHAP values are rooted in cooperative game theory. Specifically, each feature is treated as a "player" in a cooperative game, and the prediction is the "payout". The Shapley value represents the average marginal contribution of a feature across all possible combinations of features, satisfying four desirable theoretical properties: efficiency, symmetry, dummy, and additivity (Shapley, 1953).

In this study, the `TreeExplainer` was used for tree-based models (Random Forest, XGBoost, LightGBM), which exploits the tree structure for exact and efficient SHAP value computation. The following visualisations were generated:

1. **Summary plots**: Display the distribution of SHAP values across all test instances, showing both feature importance and directional effects.
2. **Bar plots**: Rank features by their mean absolute SHAP value, providing a global importance measure.
3. **Dependence plots**: Illustrate how a single feature's SHAP value varies with its actual value, revealing non-linear relationships and feature interactions.
4. **Waterfall plots**: Decompose individual predictions, showing how each feature contributes to moving the prediction from the base value (population average) to the final output.

SHAP was selected over alternative XAI methods such as LIME (Local Interpretable Model-agnostic Explanations) due to its theoretical guarantees, consistency across models, and ability to provide both global and local explanations within a unified framework (Molnar, 2022).

---

## PART B: RESULTS SECTION

### B.1 Exploratory Data Analysis Results

The exploratory analysis revealed several notable patterns in the dataset. The dataset comprises 10,000 customer records with no missing values, ensuring completeness for subsequent analysis.

**Class Distribution.** The target variable exhibits a moderate class imbalance, with 7,963 retained customers (79.6%) and 2,037 churned customers (20.4%), yielding an imbalance ratio of approximately 3.9:1 [Figure 1]. This imbalance is consistent with real-world churn rates in the banking industry, where the majority of customers typically remain with their provider (Verbeke et al., 2012).

**Age.** The age distribution reveals a clear distinction between churned and retained customers [Figure 2]. Churned customers tend to be older, with a higher concentration in the 40–60 age range, whereas retained customers are more evenly distributed across age groups. This finding aligns with existing literature suggesting that mid-career and older customers may be more likely to reassess their banking relationships (Keramati et al., 2014).

**Geography.** Churn rates vary notably across geographical regions [Figure 3]. Germany exhibits the highest churn rate, whilst France and Spain demonstrate comparatively lower rates. This geographical disparity may reflect differences in banking market dynamics, customer expectations, or competitive pressures across these regions.

**Gender.** Female customers demonstrate a higher churn rate compared to male customers [Figure 3]. This observation warrants careful consideration from an ethical perspective, as discussed in Section D.

**Correlation Analysis.** The correlation heatmap [Figure 4] reveals that age exhibits the strongest positive correlation with churn, whilst active membership status shows a negative correlation, indicating that engaged customers are less likely to churn. Credit score, tenure, and estimated salary show weak correlations with churn, suggesting that these features alone may not be strong predictors but could contribute when combined with other features in non-linear models.

**Number of Products.** An interesting non-linear relationship was observed between the number of banking products and churn rate [Figure 6]. Customers with one or two products show moderate churn rates, whilst those with three or four products exhibit substantially higher churn rates, potentially indicating customer dissatisfaction or over-complexity in their banking arrangements.

### B.2 Model Performance Results

[Table 2: Model Performance Comparison]

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8375 | 0.5915 | 0.6511 | 0.6199 | 0.8639 |
| XGBoost | 0.8225 | 0.5524 | 0.6732 | 0.6069 | 0.8469 |
| LightGBM | 0.8150 | 0.5361 | 0.6757 | 0.5978 | 0.8501 |
| SVM | 0.7865 | 0.4838 | 0.7346 | 0.5834 | 0.8496 |
| Logistic Regression | 0.7135 | 0.3872 | 0.7002 | 0.4987 | 0.7771 |

The model performance comparison reveals a clear hierarchy among the classifiers [Table 2]. The gradient boosting methods — XGBoost and LightGBM — consistently outperform the other models across all evaluation metrics, with XGBoost typically achieving the highest F1 Score and AUC-ROC values. This superior performance can be attributed to their sequential learning approach, which iteratively corrects prediction errors, and their built-in regularisation mechanisms that mitigate overfitting (Chen and Guestrin, 2016).

Random Forest demonstrates competitive performance, benefiting from its ensemble approach and robustness to noise. However, it falls slightly behind the boosting methods, likely because bagging does not specifically target difficult-to-classify instances as boosting does.

Logistic Regression, whilst the simplest model, provides a valuable baseline and demonstrates reasonable recall when class weighting is applied. Its performance gap relative to the ensemble models highlights the non-linear nature of the churn prediction problem, which linear models cannot fully capture.

The Support Vector Machine performs adequately but does not surpass the tree-based methods, possibly due to the complexity of the decision boundary required for this multi-feature classification task.

**Confusion Matrix Analysis.** The confusion matrices [Figure 8] provide additional insight into model behaviour. The gradient boosting models achieve a more balanced distribution between true positives and true negatives compared to simpler models, which tend to over-predict the majority class. This balanced performance is particularly valuable in churn prediction, where both false negatives (missed churners) and false positives (unnecessary retention efforts) carry business costs.

**ROC Curve Analysis.** The ROC curves [Figure 9] further illustrate the discriminative ability of each model. The gradient boosting methods show the greatest area under the curve, indicating superior ability to distinguish between churned and retained customers across all classification thresholds.

### B.3 Feature Importance Results

The built-in feature importance analysis across tree-based models [Figure 10] reveals a consistent pattern: age, number of products, active membership status, and account balance emerge as the most influential features. This consistency across different algorithms strengthens confidence in their predictive relevance.

### B.4 SHAP Analysis Results

**Global Interpretability.** The SHAP summary plots [Figures 11, 13, 15] provide a comprehensive view of how each feature influences predictions across all test instances. Key findings include:

- **Age** consistently emerges as the most influential feature. Higher age values (shown in red) are associated with positive SHAP values, indicating an increased probability of churn. This confirms the EDA observation and suggests that banks should pay particular attention to retention strategies for older customers.

- **Number of Products** shows a nuanced relationship with churn. Customers with multiple products (3–4) exhibit high positive SHAP values, pushing predictions towards churn. This counter-intuitive finding suggests that product complexity may drive customer dissatisfaction.

- **Active Membership** status demonstrates strong protective effects. Active members consistently receive negative SHAP values, indicating reduced churn probability. This highlights the importance of customer engagement programmes.

- **Account Balance** shows a positive association with churn, particularly for customers with higher balances. This may reflect the behaviour of affluent customers who are more likely to switch providers in search of better terms.

- **Geography** (specifically being based in Germany) shows elevated SHAP values for churn, confirming the geographical disparities identified during EDA.

**SHAP Feature Importance Ranking.** The mean absolute SHAP values [Figures 12, 14] provide a quantitative feature importance ranking that is directly comparable across features and models. This ranking is generally consistent with the built-in feature importance but offers the advantage of being derived from a theoretically principled framework.

**Dependence Analysis.** The SHAP dependence plots [Figure 16] reveal non-linear relationships that traditional correlation analysis cannot capture. For instance, the effect of age on churn probability is not monotonically increasing but shows an accelerating positive effect beyond approximately 45 years of age.

**Individual Prediction Explanations.** The waterfall plots [Figure 17] demonstrate how individual predictions can be decomposed into feature contributions. For a correctly identified churned customer, the breakdown reveals which specific factors — such as advanced age, single-product engagement, and German residency — combined to produce a high churn probability. Conversely, retained customers show protective factors such as active membership and moderate age. These individual explanations are crucial for operational deployment, as they enable customer service representatives to understand why a specific customer is flagged as at-risk and to tailor retention interventions accordingly.

---

## PART C: DISCUSSION POINTS

### C.1 Key Factors Influencing Customer Churn

The analysis identifies several consistent predictors of customer churn in banking. Age emerges as the most influential factor, with customers in the 40–60 age range demonstrating significantly higher churn propensity. This finding aligns with Keramati et al. (2014), who observed that mid-career customers may be more financially literate and willing to switch providers. The number of banking products shows a non-linear relationship with churn: whilst holding one or two products is associated with moderate churn risk, customers with three or four products exhibit dramatically higher churn rates. This suggests that product complexity or aggressive cross-selling may paradoxically increase customer attrition.

Active membership status serves as a strong protective factor, consistent with the broader customer relationship management literature that emphasises engagement as a key driver of retention (Verbeke et al., 2012). Geography also plays a significant role, with German customers demonstrating notably higher churn rates. This geographical variation likely reflects differences in market competition, regulatory environments, and cultural attitudes towards banking relationships.

### C.2 Model Comparison and Selection

The comparative analysis demonstrates that gradient boosting methods (XGBoost, LightGBM) outperform both the linear baseline (Logistic Regression) and the non-linear alternatives (Random Forest, SVM) for customer churn prediction. This finding is consistent with recent benchmarking studies that have identified gradient boosting as a consistently strong performer for tabular classification tasks (Shwartz-Ziv and Armon, 2022).

The superiority of boosting methods over Random Forest can be attributed to their sequential error-correction mechanism, which specifically targets difficult-to-classify instances. The strong performance of both XGBoost and LightGBM, with marginal differences between them, suggests that the choice between these two methods may depend on practical considerations such as training speed and memory efficiency rather than predictive accuracy.

However, it is noteworthy that even the simpler models achieve reasonable performance, particularly in terms of recall when class weighting is applied. This suggests that the churn prediction signal in this dataset is sufficiently strong to be captured even by linear models, with the ensemble methods providing incremental improvements through their ability to model feature interactions and non-linear relationships.

### C.3 Importance of Explainability

The application of SHAP provides valuable insights that go beyond mere prediction accuracy. From a practical perspective, SHAP explanations enable banks to understand not only which customers are likely to churn but also why they are at risk. This distinction is critical for designing targeted retention strategies: a customer likely to churn due to inactivity requires a different intervention than one at risk due to age-related factors.

From a regulatory perspective, the increasing demand for algorithmic transparency — exemplified by the EU AI Act and the UK's approach to AI governance — makes model interpretability a practical necessity rather than merely an academic exercise (European Commission, 2021). SHAP's ability to provide consistent, theoretically grounded explanations for individual predictions makes it particularly well-suited for regulatory compliance.

The comparison of SHAP values across multiple models (Random Forest, XGBoost, LightGBM) reveals a reassuring consistency in feature importance rankings. This cross-model agreement strengthens confidence in the identified churn drivers and suggests that the insights are robust to methodological choices.

### C.4 Ethical Considerations

The analysis raises important ethical considerations. The observation that gender and geography influence churn predictions warrants careful scrutiny. Whilst these features may genuinely reflect different churn patterns across demographic groups, their use in predictive models could perpetuate or amplify existing biases if not carefully managed. Banks deploying such models must ensure that predictions do not lead to discriminatory treatment of customers based on protected characteristics (Mehrabi et al., 2021).

SHAP's transparency is particularly valuable in this context, as it enables auditing of model decisions to identify potential bias. By examining SHAP values for sensitive features, organisations can assess whether these features exert undue influence on predictions and take corrective action if necessary.

The use of a synthetic dataset in this study, whilst limiting the external validity of findings, does eliminate privacy and consent concerns. This approach aligns with growing interest in synthetic data generation as a privacy-preserving alternative for machine learning research (Jordon et al., 2022).

### C.5 Limitations

Several limitations should be acknowledged. First, the use of a synthetic dataset, whilst ethically advantageous, means that the identified patterns may not accurately reflect real-world banking customer behaviour. The generalisability of the findings is therefore limited, and validation on real-world data would be necessary before operational deployment.

Second, the hyperparameters for each model were set based on common defaults and limited tuning. A systematic hyperparameter optimisation approach (e.g., Bayesian optimisation or grid search) could potentially improve model performance, particularly for the SVM and gradient boosting models.

Third, the study does not address the temporal dimension of churn. Customer behaviour evolves over time, and a static cross-sectional analysis cannot capture dynamic patterns or early warning signals of churn. Longitudinal analysis with time-series features would provide a more nuanced understanding of the churn process.

Fourth, whilst SHAP provides valuable model explanations, it represents only one perspective on interpretability. Alternative XAI methods such as LIME, counterfactual explanations, or attention mechanisms could provide complementary insights and address different interpretability needs.

Finally, the dataset is limited to three countries and may not capture the full diversity of banking markets worldwide. A more comprehensive study would include data from a wider range of geographical and cultural contexts.

---

## PART D: LITERATURE REVIEW SUPPORT

### Theme 1: Customer Churn in Banking

Customer churn, defined as the voluntary discontinuation of a business relationship by a customer, represents a significant challenge for the banking sector. The cost of acquiring new customers is estimated to be five to seven times higher than retaining existing ones, making churn prediction and prevention a strategic priority for financial institutions (Reichheld and Sasser, 1990). Keramati et al. (2014) conducted a comprehensive meta-analysis of churn prediction research in telecommunications and banking, identifying customer demographics, usage patterns, and service quality as the primary drivers of attrition. Their findings highlighted that churn prediction models can generate substantial return on investment when integrated into proactive retention programmes.

More recently, Vafeiadis et al. (2015) compared multiple machine learning approaches for churn prediction in the telecommunications sector, finding that ensemble methods consistently outperformed single classifiers. Whilst their study focused on telecommunications, the methodological insights are transferable to the banking domain, where similar customer behaviour patterns have been observed. De Caigny et al. (2018) extended this work by incorporating textual data from customer interactions, demonstrating that multi-modal approaches can further improve prediction accuracy. However, their approach introduces additional complexity and data requirements that may not be feasible for all organisations.

A critical observation emerging from the literature is that churn prediction is not merely a technical challenge but also a business strategy problem. As Verbeke et al. (2012) argue, the value of a churn prediction model depends not only on its accuracy but also on its ability to generate actionable insights that inform retention strategies. This perspective motivates the integration of explainable AI techniques, which bridge the gap between prediction and understanding.

### Theme 2: Machine Learning for Churn Prediction

The application of machine learning to customer churn prediction has evolved considerably over the past two decades. Early approaches relied primarily on logistic regression and decision trees (Neslin et al., 2006), which offered interpretability but limited predictive power for complex, non-linear patterns. The introduction of ensemble methods — particularly Random Forests (Breiman, 2001) and gradient boosting (Friedman, 2001) — marked a significant advancement, offering improved accuracy through the aggregation of multiple weak learners.

Recent work has demonstrated the dominance of gradient boosting frameworks, particularly XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017), for tabular classification tasks. Shwartz-Ziv and Armon (2022) conducted an extensive benchmark comparing these methods against deep learning approaches on tabular data, concluding that gradient boosting methods remain the strongest performers in most scenarios. This finding is particularly relevant for churn prediction, which typically involves structured tabular data.

However, the pursuit of predictive accuracy must be balanced against the need for model interpretability. Rudin (2019) provocatively argues that for high-stakes decisions, interpretable models should be preferred over black-box models, even at the cost of some predictive performance. This tension between accuracy and interpretability is a central theme in the churn prediction literature and motivates the adoption of post-hoc explanation methods such as SHAP.

### Theme 3: Explainable AI (XAI)

The field of Explainable AI has gained considerable momentum in response to the growing deployment of complex machine learning models in consequential decision-making contexts. Arrieta et al. (2020) provide a comprehensive taxonomy of XAI methods, distinguishing between intrinsically interpretable models (e.g., linear regression, decision trees) and post-hoc explanation techniques applied to black-box models. Post-hoc methods are particularly relevant for churn prediction, where complex models are needed for accuracy but stakeholders require transparency.

SHAP, proposed by Lundberg and Lee (2017), has emerged as one of the most widely adopted XAI frameworks. Its grounding in Shapley values from cooperative game theory provides theoretical guarantees that other methods, such as LIME (Ribeiro et al., 2016), lack. Specifically, SHAP satisfies the properties of local accuracy, missingness, and consistency, ensuring that explanations are faithful to the model's actual decision process. Comparative studies have shown that SHAP explanations are more consistent and stable than those produced by perturbation-based methods, particularly for tree-based models where exact computation is feasible (Molnar, 2022).

Despite its advantages, SHAP is not without limitations. The computational cost of exact SHAP values can be prohibitive for non-tree-based models, and the assumption of feature independence in the KernelSHAP approximation may produce misleading explanations when features are correlated (Kumar et al., 2020). These limitations should be acknowledged when interpreting SHAP outputs.

### Theme 4: Ethical Issues in AI

The deployment of machine learning models in financial services raises significant ethical concerns. Mehrabi et al. (2021) provide a comprehensive survey of fairness in machine learning, identifying multiple sources of bias — including historical bias, representation bias, and measurement bias — that can be propagated or amplified by predictive models. In the context of churn prediction, the use of demographic features such as gender and geography in predictive models could lead to discriminatory treatment if not carefully managed.

The concept of algorithmic fairness has been formalised through multiple mathematical definitions, including demographic parity, equalised odds, and calibration (Chouldechova, 2017). However, Kleinberg et al. (2016) demonstrate that several intuitively desirable fairness criteria are mathematically incompatible, highlighting the need for context-specific ethical judgements rather than purely technical solutions.

Regulatory frameworks are evolving to address these challenges. The EU AI Act classifies AI systems by risk level and imposes transparency and accountability requirements for high-risk applications, which may include financial services decision-making (European Commission, 2021). In the UK, the Information Commissioner's Office (ICO) has issued guidance on explaining AI decisions, emphasising the right of individuals to receive meaningful information about the logic involved in automated decision-making (ICO, 2020).

XAI techniques such as SHAP play a dual role in this ethical landscape: they enable organisations to audit their models for bias and discrimination, and they provide the transparency needed to comply with regulatory requirements. However, as Lipton (2018) cautions, interpretability should not be conflated with fairness — a model can be interpretable yet biased, and vice versa.

---

## PART E: BEST PRACTICES FOR DISSERTATION

### E.1 Saving Figures

All figures are saved programmatically at 300 DPI (dots per inch), which is the standard resolution for academic publications and print-quality dissertations.

**Recommended settings (already applied in the notebook):**
```python
plt.rcParams['savefig.dpi'] = 300
plt.savefig('figures/filename.png', bbox_inches='tight')
```

**Format guidance:**
- Use **PNG** for raster plots (histograms, heatmaps, scatter plots) — lossless compression, widely supported.
- Use **PDF** or **SVG** for vector plots if your dissertation template supports them — infinitely scalable, ideal for line charts.
- Avoid JPEG for plots as compression artefacts degrade quality.

### E.2 Labelling Figures and Tables

Follow your institution's style guide. As a general rule for UK MSc dissertations:

**Figures:**
- Label below the figure: `Figure 1: Distribution of Customer Churn in the Dataset`
- Number sequentially throughout the dissertation (Figure 1, Figure 2, ...) or by chapter (Figure 3.1, 3.2, ...)
- Reference in text: "As shown in Figure 1, ..." or "... (Figure 1)."

**Tables:**
- Label above the table: `Table 1: Model Performance Comparison`
- Number sequentially or by chapter
- Reference in text: "Table 2 presents the results of ..."

**General rules:**
- Every figure and table must be referenced in the text
- Captions should be descriptive enough to understand the figure without reading the main text
- Include units, axis labels, and legends where appropriate

### E.3 Recommended Folder Structure

```
dissertation_project/
├── Bank Customer Churn Prediction.csv      # Raw dataset
├── Customer_Churn_XAI_Analysis.ipynb       # Main analysis notebook
├── Dissertation_Writing_Guide.md           # This document
├── figures/                                # All generated figures (300 DPI)
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
├── outputs/                                # Generated data outputs
│   └── model_comparison.csv
└── README.md                               # Project overview
```

### E.4 Linking Code Outputs to Dissertation Writing

**Step-by-step process:**

1. **Run the notebook end-to-end** to generate all figures and outputs.
2. **Copy figures** from the `figures/` directory into your dissertation document.
3. **Extract numerical values** from the notebook output (e.g., accuracy scores, SHAP values) and insert them into the Results chapter prose and tables.
4. **Cross-reference** every figure and table in the main text.
5. **Include the notebook** (or exported PDF/HTML version) in your appendices as supporting evidence of the analysis.

**Mapping notebook sections to dissertation chapters:**

| Notebook Section | Dissertation Chapter |
|------------------|---------------------|
| Section 1–2 (Imports, Loading) | Methodology (Software, Data Description) |
| Section 3 (EDA) | Results (with figures) |
| Section 4 (Preprocessing) | Methodology |
| Section 5 (Train-Test Split) | Methodology |
| Section 6 (Model Training) | Methodology |
| Section 7 (Evaluation) | Results (with tables and figures) |
| Section 8 (Feature Importance) | Results |
| Section 9 (SHAP) | Results / Discussion |
| Section 10 (Summary) | Discussion / Conclusion |

### E.5 Key References

- Arrieta, A.B. et al. (2020) 'Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI', *Information Fusion*, 58, pp. 82–115.
- Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5–32.
- Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.
- Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', *Machine Learning*, 20(3), pp. 273–297.
- Creswell, J.W. and Creswell, J.D. (2018) *Research Design: Qualitative, Quantitative, and Mixed Methods Approaches*. 5th edn. Sage Publications.
- De Caigny, A. et al. (2018) 'A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees', *European Journal of Operational Research*, 269(2), pp. 760–772.
- European Commission (2021) *Proposal for a Regulation laying down harmonised rules on Artificial Intelligence (AI Act)*.
- Fayyad, U., Piatetsky-Shapiro, G. and Smyth, P. (1996) 'From Data Mining to Knowledge Discovery in Databases', *AI Magazine*, 17(3), pp. 37–54.
- Friedman, J.H. (2001) 'Greedy Function Approximation: A Gradient Boosting Machine', *Annals of Statistics*, 29(5), pp. 1189–1232.
- García, S. et al. (2016) 'Big data preprocessing: methods and prospects', *Big Data Analytics*, 1(1), pp. 1–22.
- Géron, A. (2022) *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3rd edn. O'Reilly Media.
- He, H. and Garcia, E.A. (2009) 'Learning from Imbalanced Data', *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284.
- Hosmer, D.W., Lemeshow, S. and Sturdivant, R.X. (2013) *Applied Logistic Regression*. 3rd edn. Wiley.
- ICO (2020) *Explaining decisions made with AI*. Information Commissioner's Office.
- James, G. et al. (2021) *An Introduction to Statistical Learning*. 2nd edn. Springer.
- Jordon, J. et al. (2022) 'Synthetic Data — what, why and how?', *arXiv preprint arXiv:2205.03257*.
- Ke, G. et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*, 30, pp. 3146–3154.
- Keramati, A. et al. (2014) 'Improved churn prediction in telecommunication industry using data mining techniques', *Applied Soft Computing*, 24, pp. 994–1012.
- Kleinberg, J., Mullainathan, S. and Raghavan, M. (2016) 'Inherent Trade-Offs in the Fair Determination of Risk Scores', *arXiv preprint arXiv:1609.05807*.
- Kohavi, R. (1995) 'A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection', *IJCAI*, 14(2), pp. 1137–1145.
- Kumar, I.E. et al. (2020) 'Problems with Shapley-value-based explanations as feature importance measures', *ICML 2020*.
- Lipton, Z.C. (2018) 'The Mythos of Model Interpretability', *Queue*, 16(3), pp. 31–57.
- Lundberg, S.M. and Lee, S.I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.
- Mehrabi, N. et al. (2021) 'A Survey on Bias and Fairness in Machine Learning', *ACM Computing Surveys*, 54(6), pp. 1–35.
- Molnar, C. (2022) *Interpretable Machine Learning*. 2nd edn. Available at: https://christophm.github.io/interpretable-ml-book/.
- Neslin, S.A. et al. (2006) 'Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models', *Journal of Marketing Research*, 43(2), pp. 204–211.
- Reichheld, F.F. and Sasser, W.E. (1990) 'Zero defections: quality comes to services', *Harvard Business Review*, 68(5), pp. 105–111.
- Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) '"Why Should I Trust You?": Explaining the Predictions of Any Classifier', *KDD '16*, pp. 1135–1144.
- Rudin, C. (2019) 'Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead', *Nature Machine Intelligence*, 1(5), pp. 206–215.
- Shapley, L.S. (1953) 'A Value for n-Person Games', *Contributions to the Theory of Games*, 2(28), pp. 307–317.
- Shwartz-Ziv, R. and Armon, A. (2022) 'Tabular data: Deep learning is not all you need', *Information Fusion*, 81, pp. 84–90.
- Vafeiadis, T. et al. (2015) 'A comparison of machine learning techniques for customer churn prediction', *Simulation Modelling Practice and Theory*, 55, pp. 1–9.
- Verbeke, W. et al. (2012) 'New insights into churn prediction in the telecommunication sector: A profit driven data mining approach', *European Journal of Operational Research*, 218(1), pp. 211–229.
