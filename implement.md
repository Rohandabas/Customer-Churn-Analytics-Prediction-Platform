Customer Churn Analytics Platform



Tech Stack
Python

Pandas

NumPy

Scikit-learn

XGBoost

SHAP

Matplotlib

Streamlit

Joblib



Keep it simple.

Final Architecture
Customer Dataset (CSV)
        │
        ▼
Data Cleaning
        │
        ▼
EDA
        │
        ▼
Feature Engineering
        │
        ▼
Train Multiple Models
        │
        ▼
Model Comparison
        │
        ▼
Best Model
        │
        ▼
SHAP Explainability
        │
        ▼
Save Model
        │
        ▼
Streamlit App
Folder Structure
customer-churn-platform/

│
├── data/
│      customer_churn.csv
│
├── notebooks/
│      01_eda.ipynb
│
├── models/
│      best_model.pkl
│      scaler.pkl
│
├── src/
│      preprocess.py
│      train.py
│      predict.py
│      explain.py
│
├── app.py
│
├── requirements.txt
│
└── README.md

Very clean.

Phase 1
Dataset

Use the IBM Telco Customer Churn dataset.

Columns like

Gender

SeniorCitizen

Partner

InternetService

Contract

MonthlyCharges

TotalCharges

Tenure

Churn

Target

Churn

Yes

No
Phase 2
EDA

Perform

Dataset Shape
Missing Values
Duplicate Values
Class Distribution
Correlation
Histograms
Boxplots
Churn Distribution

Questions to answer

Which customers churn most?

Which contract churns more?

Does tenure affect churn?

Does monthly charge matter?

Phase 3
Data Cleaning

Handle

Missing Values

Categorical Encoding

Label Encoding

One Hot Encoding

Feature Scaling
Phase 4
Feature Engineering

Create

MonthlyChargePerTenure

AverageSpend

LongTermCustomer

HighValueCustomer

Simple but shows initiative.

Phase 5
Train Multiple Models

Train

Logistic Regression

Decision Tree

Random Forest

XGBoost

Compare

Accuracy

Precision

Recall

F1 Score

ROC AUC

Pick the best.

Phase 6
Hyperparameter Tuning

Use

GridSearchCV

Only for

Random Forest

or

XGBoost

No need to tune every model.

Phase 7
Explainability

Use SHAP

Example

Customer

Tenure = 2

Monthly Charges = 110

Contract = Month-to-Month

Prediction

Churn Probability

92%

SHAP explains

High Monthly Charges

+

Low Tenure

+

Month-to-Month Contract

Recruiters love this.

Phase 8
Save Model
joblib.dump(model)

Save

model.pkl

encoder.pkl

scaler.pkl
Phase 9
Build Streamlit App

Home

Customer Churn Analytics Platform

Sidebar

Gender

Age

Tenure

Monthly Charges

Contract

Internet Service

Payment Method

etc.

Click

Predict

Output

Prediction

Likely to Churn

Probability

87%

Also show

Top Factors

Tenure

Monthly Charges

Contract
Nice Dashboard
+------------------------------------+

Customer Churn Analytics

--------------------------------------

Gender

Age

Monthly Charges

Tenure

Contract

Internet Service

[ Predict ]

--------------------------------------

Prediction

Likely To Churn

Probability

87%

Reason

✓ High Monthly Charges

✓ Month-to-Month Contract

✓ Low Tenure

--------------------------------------

Feature Importance Graph