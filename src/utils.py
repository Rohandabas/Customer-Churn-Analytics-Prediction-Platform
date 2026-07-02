import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def load_pipeline(pipeline_path):
    """
    Loads the saved scikit-learn pipeline.
    """
    return joblib.load(pipeline_path)

def get_metrics(y_true, y_pred, y_probs):
    """
    Computes accuracy, precision, recall, f1, and roc-auc scores.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_probs)
    }

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plots a confusion matrix heatmap and optionally saves it.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        return plt.gcf()

def plot_roc_curve(y_true, y_probs, model_name='Best Model', output_path=None):
    """
    Plots the ROC curve and optionally saves it.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        return plt.gcf()

def plot_feature_importance(importances, feature_names, output_path=None, top_n=20):
    """
    Plots the top N features based on model feature importances.
    """
    # Create a DataFrame for plotting
    df_feat = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_feat, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        return plt.gcf()

def print_comparison_table(metrics_dict):
    """
    Prints a formatted markdown table comparing model performance.
    """
    df = pd.DataFrame(metrics_dict).T
    print("\n=== Model Comparison Table ===")
    print(df.to_markdown())
    print("==============================\n")
    return df
