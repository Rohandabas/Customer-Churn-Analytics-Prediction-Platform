import os
import sys
import pandas as pd
import numpy as np

# Add the directory of this file to sys.path to resolve local imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from preprocess import clean_data, engineer_features
from utils import load_pipeline


def predict_single_customer(customer_dict, pipeline_path=None):
    """
    Predicts churn probability and class for a single customer.
    
    Parameters:
    - customer_dict (dict): Keys should match original CSV column names:
      'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
      'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
      'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    
    Returns:
    - dict: {'prediction': int (0/1), 'probability': float (0.0 to 1.0)}
    """
    if pipeline_path is None:
        pipeline_path = os.path.join('models', 'customer_churn_pipeline.pkl')
        
    pipeline = load_pipeline(pipeline_path)
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Preprocess
    df_clean = clean_data(df, is_training=False)
    df_feat = engineer_features(df_clean)
    
    # Predict
    pred = pipeline.predict(df_feat)[0]
    prob = pipeline.predict_proba(df_feat)[0][1]
    
    return {
        'prediction': int(pred),
        'probability': float(prob)
    }

if __name__ == '__main__':
    # Test sample prediction
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.7,
        'TotalCharges': '151.65' # Note it can be string or numeric
    }
    
    try:
        res = predict_single_customer(sample_customer)
        print("Test prediction completed successfully:")
        print(res)
    except Exception as e:
        print(f"Failed to run test prediction (model might not be trained yet): {e}")
