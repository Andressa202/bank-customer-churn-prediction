# 🏦 Bank Customer Churn Prediction | Machine Learning & Retention Analytics

### Predicting Customer Attrition Using Random Forest and Imbalanced Classification Techniques

**Python • Pandas • NumPy • Scikit-learn • Random Forest • Classification • Model Evaluation**

## 📌 Project Overview

Customer retention is a critical challenge in the banking industry. Understanding which customers are likely to leave can help financial institutions develop targeted retention strategies and reduce customer acquisition costs.

This project develops a machine learning classification model for **Beta Bank**, a fictional financial institution, to predict customer churn using historical customer data.

A key challenge is the imbalance between customers who remain with the bank and those who leave. The project evaluates different techniques to address this imbalance and improve predictive performance.

**Business Goal:** Develop a model capable of identifying customers at risk of leaving the bank, achieving a minimum F1-score of 0.59 on the test dataset.

## 🎯 Project Objectives

- Explore and prepare customer data for predictive modeling.
- Identify and address missing values.
- Encode categorical variables and scale numerical features.
- Investigate the distribution of churn classes.
- Train and optimize a Random Forest classifier.
- Compare different class imbalance handling strategies.
- Evaluate model performance using F1-score and ROC-AUC.
- Identify a suitable approach for customer churn prediction.

## 🛠️ Technologies Used

| Technology | Application |
|---|---|
| Python | Programming and data processing |
| Pandas | Data cleaning and feature preparation |
| NumPy | Numerical operations |
| Scikit-learn | Classification, preprocessing, and model evaluation |
| Matplotlib | Data visualization |
| Jupyter Notebook | Experimentation and documentation |

## 📂 Dataset Description

The project uses `Churn.csv`, containing customer information from Beta Bank.

The dataset includes **10,000 customer records**.

| Feature | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Country of residence |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Number of years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of banking products |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Customer activity status |
| `EstimatedSalary` | Estimated salary |
| `Exited` | Target variable indicating customer churn |

### Target Variable

- `0` — Customer remained with the bank
- `1` — Customer left the bank

The identifier fields `RowNumber`, `CustomerId`, and `Surname` were removed before model training.

## 🧹 Data Preparation

The preprocessing workflow included:

1. Removing irrelevant identifier columns.
2. Checking missing values.
3. Filling missing `Tenure` values using the median.
4. Encoding categorical variables using one-hot encoding.
5. Separating features and target.
6. Splitting the dataset into training, validation, and test sets.
7. Standardizing numerical variables using `StandardScaler`.

The data was divided as follows:

| Dataset | Proportion |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

Stratified splitting was applied to preserve the target class distribution.

## ⚖️ Class Imbalance Analysis

The target distribution revealed a substantial class imbalance.

| Customer Status | Percentage |
|---|---:|
| Remained | 79.63% |
| Churned | 20.37% |

Because approximately one-fifth of customers belonged to the churn class, relying exclusively on accuracy could produce misleading results.

The project therefore prioritized **F1-score** and also evaluated **ROC-AUC**.

## 🤖 Machine Learning Methodology

The project trained a Random Forest classifier and compared three strategies.

### 1. Baseline Random Forest

A Random Forest model was trained without adjusting class imbalance.

Different values of `max_depth` were tested, and the best configuration was selected using validation F1-score.

### 2. Random Forest with Class Weights

The second approach used:

`class_weight='balanced'`

This configuration assigns greater importance to the minority class during training.

### 3. Random Forest with Oversampling

The third approach used random oversampling with replacement to increase the number of minority-class training examples.

Oversampling was applied to the training data, keeping the validation and test sets separate.

## 📊 Model Comparison

### Validation Performance

| Approach | F1-score |
|---|---:|
| Random Forest without balancing | 0.6152 |
| Random Forest with class weights | 0.6490 |
| **Random Forest with oversampling** | **0.6519** |

The oversampling approach achieved the highest validation F1-score and was selected for final testing.

## 🧪 Final Model Evaluation

The selected Random Forest model was evaluated on the independent test dataset.

| Metric | Result |
|---|---:|
| **F1-score** | **0.6026** |
| **ROC-AUC** | **0.8564** |
| Accuracy | 0.82 |
| Precision — churn class | 0.55 |
| Recall — churn class | 0.67 |
| Minimum required F1-score | 0.59 |

### Interpretation

The model achieved an F1-score of **0.6026**, exceeding the project's minimum requirement of 0.59.

The ROC-AUC score of **0.8564** indicates good discrimination between customers who leave and those who remain.

The model identified approximately 67% of customers who actually churned, although its precision of 55% indicates that some customers predicted to leave did not do so.

These results illustrate the trade-off between identifying more at-risk customers and avoiding unnecessary retention interventions.

## 💡 Key Findings

- Customer churn represents approximately 20% of the dataset.
- Class imbalance handling improved validation F1-score.
- Random oversampling achieved the highest validation F1-score among the tested approaches.
- The final model met the required F1-score threshold.
- ROC-AUC provided an additional perspective on model discrimination.
- Precision and recall remain important considerations for practical customer retention strategies.

## ⚠️ Limitations and Future Improvements

Potential improvements include:

- Testing Logistic Regression and alternative classification algorithms.
- Using cross-validation for more robust model selection.
- Applying `GridSearchCV` or `RandomizedSearchCV`.
- Exploring SMOTE and other resampling strategies.
- Tuning classification thresholds based on business priorities.
- Investigating feature importance and model explainability.
- Estimating the financial impact of false positives and false negatives.

An additional preprocessing improvement would be to fit missing-value imputation parameters exclusively on the training set to reduce the risk of data leakage.

## 🚀 How to Run the Project

1. Clone the repository.
2. Install the required Python libraries.
3. Place `Churn.csv` in the project data directory.
4. Update the dataset path in the notebook.
5. Open the notebook in Jupyter Notebook or JupyterLab.
6. Execute all cells sequentially.

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib notebook
```

## 📈 Conclusion

This project demonstrates the application of supervised machine learning to a customer retention problem in the banking sector.

By comparing class imbalance handling techniques, the project selected a Random Forest model with oversampling that achieved an F1-score of **0.6026** and ROC-AUC of **0.8564** on the test dataset.

The project highlights practical skills in data preprocessing, feature encoding, model optimization, imbalanced classification, and predictive performance evaluation.

## 👩‍💻 Author

**Andressa Cruz Miranda**

Computer Science | Data Analytics | Python | Machine Learning

[GitHub Profile](https://github.com/Andressa202)

*Developed as part of the TripleTen Data Science program.*
