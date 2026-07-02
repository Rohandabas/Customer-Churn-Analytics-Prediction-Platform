import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Local imports
from preprocess import load_data, clean_data, engineer_features, get_preprocessor
from utils import get_metrics, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, print_comparison_table

def train_models():
    # 1. Load and Preprocess Data
    data_path = os.path.join('data', 'customer_churn.csv')
    print(f"Loading data from {data_path}...")
    df_raw = load_data(data_path)
    
    print("Cleaning and engineering features...")
    df_clean = clean_data(df_raw, is_training=True)
    df_feat = engineer_features(df_clean)
    
    X = df_feat.drop(columns=['Churn'])
    y = df_feat['Churn']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 2. Get preprocessor and column names
    preprocessor, raw_feature_names = get_preprocessor()
    
    # Define the 4 base models
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Define search spaces for tuning
    param_distributions = {
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 3, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 150, 200],
            'classifier__max_depth': [3, 4, 5, 6, 8],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
    }
    
    # 3. Train and compare base models
    print("\n--- Training Base Models ---")
    base_metrics = {}
    fitted_base_pipelines = {}
    
    for name, model in base_models.items():
        print(f"Training base {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        fitted_base_pipelines[name] = pipeline
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = get_metrics(y_test, y_pred, y_probs)
        base_metrics[name] = metrics
        
    # Display comparison table
    print_comparison_table(base_metrics)
    
    # 4. Select top 2 models based on ROC AUC
    sorted_models = sorted(base_metrics.items(), key=lambda item: item[1]['ROC AUC'], reverse=True)
    top_2_names = [sorted_models[0][0], sorted_models[1][0]]
    print(f"\nTop 2 models selected for hyperparameter tuning: {top_2_names}")
    
    # 5. Tune top 2 models
    tuned_models = {}
    tuned_metrics = {}
    
    for name in top_2_names:
        print(f"\n--- Tuning {name} with RandomizedSearchCV ---")
        model = base_models[name]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Grid/Randomized search
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions[name],
            n_iter=15,
            scoring='roc_auc',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        
        best_pipeline = search.best_estimator_
        tuned_models[name] = best_pipeline
        
        # Evaluate
        y_pred = best_pipeline.predict(X_test)
        y_probs = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_probs)
        tuned_metrics[name] = metrics
        print(f"Best parameters: {search.best_params_}")
        print(f"Tuned ROC AUC: {metrics['ROC AUC']:.4f}")
        
    # 6. Select overall best model among tuned models
    best_tuned_name = max(tuned_metrics.items(), key=lambda item: item[1]['ROC AUC'])[0]
    final_pipeline = tuned_models[best_tuned_name]
    print(f"\nOverall Best Model is: {best_tuned_name} (Tuned)")
    
    # Evaluate final pipeline
    y_pred_final = final_pipeline.predict(X_test)
    y_probs_final = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = get_metrics(y_test, y_pred_final, y_probs_final)
    
    print("\nFinal Model Evaluation Metrics:")
    for metric_name, val in final_metrics.items():
        print(f"- {metric_name}: {val:.4f}")
        
    # 7. Save Final Pipeline & Background Data for SHAP
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    pipeline_save_path = os.path.join(models_dir, 'customer_churn_pipeline.pkl')
    print(f"\nSaving final pipeline to {pipeline_save_path}...")
    joblib.dump(final_pipeline, pipeline_save_path)
    
    # Save a small subset of preprocessed training data as background data for SHAP
    print("Saving background data for SHAP explainer...")
    X_train_preprocessed = final_pipeline.named_steps['preprocessor'].transform(X_train)
    sample_indices = np.random.choice(X_train_preprocessed.shape[0], min(100, X_train_preprocessed.shape[0]), replace=False)
    background_data = X_train_preprocessed[sample_indices]
    joblib.dump(background_data, os.path.join(models_dir, 'preprocessed_background.pkl'))
    
    # Save a JSON configuration of training metrics for App comparison
    # We can compile all metrics together to show in the Streamlit app
    comparison_df = pd.DataFrame(base_metrics).T
    comparison_df.loc[f"{best_tuned_name} (Tuned)"] = final_metrics
    comparison_save_path = os.path.join(models_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_save_path)
    print(f"Saved model comparison table to {comparison_save_path}")
    
    # 8. Generate and save Plots
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)
    
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred_final, output_path=os.path.join(images_dir, 'confusion_matrix.png'))
    
    print("Generating ROC Curve...")
    plot_roc_curve(y_test, y_probs_final, model_name=best_tuned_name, output_path=os.path.join(images_dir, 'roc_curve.png'))
    
    # Extract Feature Importances
    print("Extracting Feature Importances...")
    classifier = final_pipeline.named_steps['classifier']
    
    # Get feature names out of the ColumnTransformer
    preprocessor_obj = final_pipeline.named_steps['preprocessor']
    try:
        feature_names = preprocessor_obj.get_feature_names_out()
        # Clean prefix names (e.g. num__tenure -> tenure, cat__gender_Male -> gender_Male)
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
    except Exception as e:
        print(f"Warning: Could not get feature names using get_feature_names_out: {e}")
        # fallback to raw feature names (order might differ, but this is a fallback)
        feature_names = raw_feature_names
        
    importances = None
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
        
    if importances is not None:
        # Match length of importances and feature_names in case of mismatch
        if len(importances) == len(feature_names):
            plot_feature_importance(
                importances, 
                feature_names, 
                output_path=os.path.join(images_dir, 'feature_importance.png'), 
                top_n=20
            )
            # Save feature importances to CSV for easy display in streamlit
            pd.DataFrame({'Feature': feature_names, 'Importance': importances})\
              .sort_values(by='Importance', ascending=False)\
              .to_csv(os.path.join(models_dir, 'feature_importances.csv'), index=False)
            print("Feature importances plotted and saved.")
        else:
            print(f"Warning: Mismatch between importance count ({len(importances)}) and feature names ({len(feature_names)}). Skipping plot.")
    else:
        print("Warning: Classifier does not support feature importances / coefficients.")

if __name__ == '__main__':
    train_models()
