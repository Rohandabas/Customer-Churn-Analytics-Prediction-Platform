import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Add the directory of this file to sys.path to resolve local imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from preprocess import clean_data, engineer_features
from utils import load_pipeline

def get_shap_explanation(customer_dict, pipeline_path=None, background_path=None):
    """
    Computes SHAP explanation for a single customer.
    
    Parameters:
    - customer_dict (dict): Single customer features dictionary
    
    Returns:
    - explanation: SHAP Explanation object
    - feature_names: List of preprocessed feature names
    - preprocessed_df: DataFrame containing the preprocessed sample
    """
    if pipeline_path is None:
        pipeline_path = os.path.join('models', 'customer_churn_pipeline.pkl')
    if background_path is None:
        background_path = os.path.join('models', 'preprocessed_background.pkl')
        
    pipeline = load_pipeline(pipeline_path)
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # 1. Clean and engineer features for the single customer
    df = pd.DataFrame([customer_dict])
    df_clean = clean_data(df, is_training=False)
    df_feat = engineer_features(df_clean)
    
    # 2. Transform the customer using the preprocessor
    # Note: ColumnTransformer output is a numpy array. We'll reconstruct a DataFrame with proper column names.
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean prefix names
        cleaned_feature_names = []
        for name in feature_names:
            if name.startswith('num__'):
                cleaned_feature_names.append(name.replace('num__', ''))
            elif name.startswith('cat__'):
                cleaned_feature_names.append(name.replace('cat__', ''))
            elif name.startswith('remainder__'):
                cleaned_feature_names.append(name.replace('remainder__', ''))
            else:
                cleaned_feature_names.append(name)
        feature_names = cleaned_feature_names
    except Exception:
        # Fallback names
        feature_names = [f"Feature_{i}" for i in range(preprocessor.transform(df_feat).shape[1])]
        
    preprocessed_arr = preprocessor.transform(df_feat)
    preprocessed_df = pd.DataFrame(preprocessed_arr, columns=feature_names)
    
    # 3. Load background data for SHAP explainer
    background_data = None
    if os.path.exists(background_path):
        background_data = joblib.load(background_path)
        # Ensure it has column names matching feature_names
        background_data = pd.DataFrame(background_data, columns=feature_names)
        
    # 4. Initialize explainer
    # Tree models (XGBoost, Random Forest, Decision Tree) can use TreeExplainer which is fast.
    # Logistic Regression can use LinearExplainer.
    # Otherwise fallback to general Explainer.
    try:
        model_type = type(classifier).__name__
        if model_type in ['XGBClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            explainer = shap.TreeExplainer(classifier)
        elif model_type in ['LogisticRegression']:
            if background_data is not None:
                explainer = shap.LinearExplainer(classifier, background_data)
            else:
                explainer = shap.LinearExplainer(classifier, preprocessed_df)
        else:
            if background_data is not None:
                explainer = shap.Explainer(classifier.predict_proba, background_data)
            else:
                explainer = shap.Explainer(classifier.predict_proba, preprocessed_df)
                
        # 5. Compute SHAP values
        if model_type in ['XGBClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            # TreeExplainer outputs list of arrays for multi-class classification or 3D arrays.
            # Binary classification RF/DT may output list of len 2 (No Churn, Churn).
            # XGBoost binary classification outputs values for Churn (positive class) directly.
            shap_values = explainer(preprocessed_df)
            
            # Handle multi-class outputs or list outputs (e.g. for RF/DT)
            if isinstance(shap_values, list):
                # Class 1 (Churn) is index 1
                explanation = shap_values[1]
            elif len(shap_values.shape) == 3: # shape: (samples, features, classes)
                # Select positive class
                explanation = shap_values[:, :, 1]
            else:
                explanation = shap_values
        else:
            # LinearExplainer or General Explainer
            # For predict_proba models: shape could be (samples, features, classes)
            shap_values = explainer(preprocessed_df)
            if len(shap_values.shape) == 3:
                explanation = shap_values[:, :, 1]
            else:
                explanation = shap_values
                
    except Exception as e:
        print(f"Error initializing or running SHAP explainer: {e}")
        # Fallback to general explainer
        if background_data is not None:
            explainer = shap.Explainer(classifier.predict_proba, background_data)
        else:
            explainer = shap.Explainer(classifier.predict_proba, preprocessed_df)
        shap_values = explainer(preprocessed_df)
        if len(shap_values.shape) == 3:
            explanation = shap_values[:, :, 1]
        else:
            explanation = shap_values
            
    # Set the feature names on the explanation object so plots show them
    explanation.feature_names = feature_names
    
    return explanation, feature_names, preprocessed_df

def plot_shap_waterfall(explanation, output_path=None):
    """
    Generates a SHAP waterfall plot and saves it.
    """
    plt.figure(figsize=(10, 6))
    # explanation[0] gets the explanation for the single sample
    shap.plots.waterfall(explanation[0], show=False)
    plt.title("SHAP Feature Attribution (Why did the model predict this?)", fontsize=12, pad=15)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        return plt.gcf()
