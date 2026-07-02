import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add src to python path to resolve sub-module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Local imports
from src.predict import predict_single_customer
from src.explain import get_shap_explanation, plot_shap_waterfall
from src.utils import load_pipeline


# Page Configuration
st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Customer Churn Analytics & Prediction Platform")
st.markdown("An interactive machine learning platform to predict customer churn, explain predictions using SHAP, and explore dataset insights.")

# Cache loading dataset
@st.cache_data
def get_dataset():
    path = os.path.join('data', 'customer_churn.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_raw = get_dataset()

# Setup Sidebar for Single Customer input
st.sidebar.header("Customer Information Input")

# Create inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=24)

st.sidebar.markdown("---")
st.sidebar.subheader("Services Subscribed")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.sidebar.markdown("---")
st.sidebar.subheader("Contract & Billing")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)

# Calculate recommended total charges
recommended_total = round(monthly_charges * tenure, 2)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", 
    min_value=0.0, 
    value=float(recommended_total) if tenure > 0 else 0.0,
    help="Defaulted to Monthly Charges * Tenure. Modify if needed."
)

# Convert inputs into model payload format
payload = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Create Tabs
tab_predict, tab_performance, tab_eda = st.tabs([
    "🎯 Churn Prediction & SHAP Explanation",
    "📈 Model Performance & Evaluation",
    "📊 Dataset Exploratory Insights"
])

# ==================== TAB 1: PREDICT ====================
with tab_predict:
    st.subheader("Run Churn Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Customer Summary")
        summary_df = pd.DataFrame(list(payload.items()), columns=["Feature", "Input Value"])
        summary_df['Input Value'] = summary_df['Input Value'].astype(str)
        st.dataframe(summary_df, use_container_width=True, height=450)
        
        predict_btn = st.button("Predict Churn Probability", type="primary", use_container_width=True)
        
    with col2:
        if predict_btn:
            try:
                # 1. Run prediction
                res = predict_single_customer(payload)
                prob = res['probability']
                pred = res['prediction']
                
                # Show results in a clean layout
                st.markdown("### Model Predictions")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    if pred == 1:
                        st.error("⚠️ HIGH RISK OF CHURN")
                    else:
                        st.success("✅ LOW RISK OF CHURN")
                with metric_col2:
                    st.metric(label="Churn Probability", value=f"{prob*100:.2f}%")
                
                st.progress(prob)
                
                # 2. Get and plot SHAP
                st.markdown("### Local Explanation (SHAP Attribution)")
                st.info("The chart below explains how the customer's traits pushed the model's prediction higher (red) or lower (blue).")
                
                explanation, _, _ = get_shap_explanation(payload)
                
                fig = plot_shap_waterfall(explanation)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error executing prediction: {e}")
                st.warning("Please ensure the model training step (`train.py`) is successfully completed to save the pipeline.")
        else:
            st.info("👈 Enter customer details in the sidebar and click **Predict Churn Probability** to run the inference.")

# ==================== TAB 2: PERFORMANCE ====================
with tab_performance:
    st.subheader("Model Validation & Training Insights")
    
    comparison_path = os.path.join('models', 'model_comparison.csv')
    if os.path.exists(comparison_path):
        st.markdown("### Candidate Model Performance Comparison")
        comp_df = pd.read_csv(comparison_path).rename(columns={'Unnamed: 0': 'Model'})
        st.dataframe(comp_df, use_container_width=True)
        
        st.markdown("### Final Model Evaluation Plots")
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            cm_img_path = os.path.join('images', 'confusion_matrix.png')
            if os.path.exists(cm_img_path):
                st.image(Image.open(cm_img_path), caption="Confusion Matrix on Holdout Set", use_container_width=True)
            else:
                st.warning("Confusion matrix plot not found.")
                
        with plot_col2:
            roc_img_path = os.path.join('images', 'roc_curve.png')
            if os.path.exists(roc_img_path):
                st.image(Image.open(roc_img_path), caption="ROC Curve on Holdout Set", use_container_width=True)
            else:
                st.warning("ROC curve plot not found.")
                
        st.markdown("### Global Feature Importance")
        feat_img_path = os.path.join('images', 'feature_importance.png')
        if os.path.exists(feat_img_path):
            st.image(Image.open(feat_img_path), caption="Top 20 Important Features", use_container_width=True)
        else:
            st.warning("Feature importance plot not found.")
    else:
        st.warning("Model evaluation data not found. Please run the model training step first (`python src/train.py`).")

# ==================== TAB 3: EDA ====================
with tab_eda:
    st.subheader("Historical Dataset Analytics")
    if df_raw is not None:
        st.write(f"Total historical customer records: **{df_raw.shape[0]}**")
        
        # Show interactive metrics
        eda_col1, eda_col2, eda_col3 = st.columns(3)
        with eda_col1:
            avg_tenure = df_raw['tenure'].mean()
            st.metric("Average Customer Tenure", f"{avg_tenure:.1f} months")
        with eda_col2:
            avg_charges = df_raw['MonthlyCharges'].mean()
            st.metric("Average Monthly Charges", f"${avg_charges:.2f}")
        with eda_col3:
            churn_rate = (df_raw['Churn'] == 'Yes').mean() * 100
            st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
            
        # Dist plots
        st.markdown("### Key Factor Visualizations")
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            st.write("**Contract Type vs Churn**")
            # Create a simple contract-churn table
            contract_churn = pd.crosstab(df_raw['Contract'], df_raw['Churn'], normalize='index') * 100
            st.bar_chart(contract_churn)
            st.caption("Month-to-month contracts exhibit significantly higher churn rates compared to long-term commitments.")
            
        with factor_col2:
            st.write("**Tenure Distribution by Churn**")
            # Streamlit hist
            tenure_churn = df_raw.pivot(columns='Churn', values='tenure')
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df_raw, x='tenure', hue='Churn', multiple='stack', palette='coolwarm', ax=ax, bins=20)
            st.pyplot(fig)
            st.caption("Fresh sign-ups (0-10 months) make up the bulk of churned users.")
            
        # Charges plot
        st.write("**Monthly Charges Density by Churn Status**")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=df_raw, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, alpha=0.5, palette='coolwarm', ax=ax2)
        st.pyplot(fig2)
        st.caption("Users paying high monthly rates ($70-$100+) show higher densities of churn.")
    else:
        st.warning("Historical CSV dataset `data/customer_churn.csv` is missing.")
