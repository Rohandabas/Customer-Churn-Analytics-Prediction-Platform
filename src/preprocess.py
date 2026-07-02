import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """
    Loads customer churn data from a CSV file.
    """
    return pd.read_csv(file_path)

def clean_data(df, is_training=True):
    """
    Cleans the raw DataFrame:
    - Removes customerID
    - Handles spaces in TotalCharges and converts to float
    - Removes rows with missing TotalCharges
    - Maps target Churn column to 0/1 if is_training is True
    """
    df = df.copy()
    
    # Drop customerID if it exists
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Convert TotalCharges to numeric, replacing spaces with NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
        # Drop rows where TotalCharges is missing (only 11 rows)
        df = df.dropna(subset=['TotalCharges'])
        
    if is_training and 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def engineer_features(df):
    """
    Performs feature engineering on the cleaned dataframe:
    - MonthlyChargePerTenure = MonthlyCharges / (tenure + 1)
    - AverageSpend = TotalCharges / (tenure + 1)
    - LongTermCustomer = 1 if tenure > 24 else 0
    - HighValueCustomer = 1 if MonthlyCharges > 80 else 0
    """
    df = df.copy()
    
    df['MonthlyChargePerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['AverageSpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['LongTermCustomer'] = (df['tenure'] > 24).astype(int)
    df['HighValueCustomer'] = (df['MonthlyCharges'] > 80).astype(int)
    
    return df

def get_preprocessor():
    """
    Returns a ColumnTransformer preprocessor:
    - Scales numerical features
    - One-hot encodes categorical features
    - Passes through binary features
    """
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyChargePerTenure', 'AverageSpend']
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    # SeniorCitizen, LongTermCustomer, HighValueCustomer will be matched as passthrough in remainder
    
    # We list all columns explicitly to ensure the ColumnTransformer maintains a strict ordering
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, num_cols + cat_cols + ['SeniorCitizen', 'LongTermCustomer', 'HighValueCustomer']
