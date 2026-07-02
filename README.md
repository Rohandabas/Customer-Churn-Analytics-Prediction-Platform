# Customer Churn Analytics Platform

An end-to-end, production-grade Machine Learning system designed to identify and analyze customer churn. The platform includes a modular data preprocessing pipeline, multi-model evaluation and tuning via `RandomizedSearchCV`, SHAP-based local explainability, and an interactive Streamlit dashboard.

## Final Project Structure

```
customer-churn-platform/
│
├── data/
│   └── customer_churn.csv          # Raw IBM Telco Churn Dataset
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis Notebook
│
├── models/
│   ├── customer_churn_pipeline.pkl  # Serialized Scikit-learn Pipeline (Preprocessor + Classifier)
│   ├── preprocessed_background.pkl # Preprocessed background samples for SHAP
│   ├── model_comparison.csv        # Metrics summary of candidate models
│   └── feature_importances.csv     # Calculated feature importances
│
├── src/
│   ├── preprocess.py               # Data loading, cleaning & feature engineering
│   ├── train.py                    # Model training, comparison & grid search tuning
│   ├── predict.py                  # Predict utilities for single customer records
│   ├── explain.py                  # SHAP value generation and visualization
│   └── utils.py                    # Metrics, comparison tables, and evaluation plotting
│
├── images/
│   ├── confusion_matrix.png        # Confusion Matrix plot on holdout test set
│   ├── feature_importance.png      # Global Feature Importance plot
│   └── roc_curve.png               # ROC Curve plot
│
├── app.py                          # Streamlit Interactive Dashboard
└── requirements.txt                # Python environment requirements
```

## Features & Analytics

1. **Feature Engineering**:
   - `MonthlyChargePerTenure`: Captures the rate of charges normalized by length of relationship.
   - `AverageSpend`: Calculates total customer spending scaled over tenure.
   - `LongTermCustomer`: Boolean flag for customer tenure > 24 months.
   - `HighValueCustomer`: Boolean flag for customers with monthly charges > $80.
2. **Model Selection & Hyperparameter Tuning**:
   - Train 4 candidate models: `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`.
   - Select top 2 models based on holdout **ROC-AUC** metrics.
   - Perform hyperparameter tuning using `RandomizedSearchCV` on the top 2 models.
   - Package the best overall model inside a Scikit-learn preprocessing `Pipeline` to prevent train/test feature mismatches.
3. **Explainability**:
   - Utilizes **SHAP (SHapley Additive exPlanations)** to dissect single-customer predictions and show exactly which service/demographic items influenced the churn probability.

## Setup and Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
To run the end-to-end preprocessing, model ranking, tuning, evaluation, and generate final plots:
```bash
python src/train.py
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

## Model Results

The following metrics were calculated on the 20% holdout validation dataset:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 80.17% | 64.98% | 55.08% | 59.62% | 0.8381 |
| **Decision Tree** | 72.42% | 48.18% | 49.47% | 48.81% | 0.6508 |
| **Random Forest** | 79.10% | 63.51% | 50.27% | 56.12% | 0.8206 |
| **XGBoost** | 76.47% | 56.34% | 51.07% | 53.58% | 0.8108 |

*Tuned Model Results:*
- **Logistic Regression (Tuned)**: **ROC AUC = 0.8369**, Accuracy = 79.74%, F1 Score = 0.5864.
- **Random Forest (Tuned)**: **ROC AUC = 0.8337**, Accuracy = 80.60%, F1 Score = 0.5898.

The final model selected for deployment is the **Tuned Logistic Regression** pipeline.
