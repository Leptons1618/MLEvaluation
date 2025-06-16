"""
Streamlit UI components for ML Evaluation App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Dict, Any, Optional
from ..utils.config import APP_TITLE, DATASETS, MODELS
from ..utils.logging_config import get_logger

logger = get_logger('ui')


def setup_page_config():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render the application header"""
    st.title("🤖 AI Explainer Pro - Model Insights & Evaluation")
    st.markdown("*A comprehensive tool for model interpretation and explainability assessment*")
    
    # Add navigation info
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        This application provides comprehensive model explainability tools including:
        - **SHAP** explanations with automatic fallback mechanisms
        - **LIME** local interpretable model explanations  
        - **Feature Importance** analysis
        - **Model Performance** evaluation and comparison
        - **Interactive Visualizations** for better understanding
        
        Choose a dataset and model from the sidebar to get started!
        """)


def render_sidebar() -> Dict[str, str]:
    """Render sidebar controls and return user selections"""
    st.sidebar.header("🎛️ Configuration")
    
    # Dataset selection
    st.sidebar.subheader("📊 Dataset Selection")
    dataset_options = {name: config['name'] for name, config in DATASETS.items()}
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        help="Select the dataset for model training and explanation"
    )
    
    # Model selection  
    st.sidebar.subheader("🤖 Model Selection")
    model_options = {name: config['name'] for name, config in MODELS.items()}
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        help="Select the machine learning model to train and explain"
    )
    
    # Display dataset info
    if selected_dataset:
        dataset_config = DATASETS[selected_dataset]
        st.sidebar.info(f"""
        **{dataset_config['name']}**
        - Type: {dataset_config['type'].title()}
        - Features: {dataset_config['features']}
        - Classes: {dataset_config['classes']}
        
        {dataset_config['description']}
        """)
    
    # Display model info
    if selected_model:
        model_config = MODELS[selected_model]
        st.sidebar.info(f"""
        **{model_config['name']}**
        - Type: {model_config['type'].title()}
        - Feature Importance: {'✓' if model_config['supports_feature_importance'] else '✗'}
        
        {model_config['description']}
        """)
    
    return {
        'dataset': selected_dataset,
        'model': selected_model
    }


def render_dataset_info(dataset_info: Dict[str, Any]):
    """Render dataset information"""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(dataset_info['X']))
    with col2:
        st.metric("Features", len(dataset_info['feature_names']))
    with col3:
        st.metric("Classes", len(dataset_info['target_names']))
    with col4:
        st.metric("Type", dataset_info['config']['type'].title())
    
    # Dataset preview
    with st.expander("🔍 Data Preview"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Features (first 5 rows):**")
            st.dataframe(dataset_info['X'].head())
        
        with col2:
            st.write("**Target Distribution:**")
            target_counts = pd.Series(dataset_info['y']).value_counts().sort_index()
            target_df = pd.DataFrame({
                'Class': [dataset_info['target_names'][i] for i in target_counts.index],
                'Count': target_counts.values
            })
            st.dataframe(target_df)


def render_model_performance(train_info: Dict[str, Any], eval_info: Dict[str, Any]):
    """Render model performance metrics"""
    st.subheader("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", f"{train_info['train_score']:.4f}")
    with col2:
        st.metric("Test Accuracy", f"{eval_info['test_score']:.4f}")
    
    # Detailed metrics
    with st.expander("📋 Detailed Performance Metrics"):
        st.text("Classification Report:")
        st.text(eval_info['classification_report'])
        
        st.write("Confusion Matrix:")
        st.write(eval_info['confusion_matrix'])


def render_prediction_section(model, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Render prediction section and return selected input"""
    st.subheader("🎯 Single Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Select or modify feature values:**")
        
        # Get a sample from test set
        sample_idx = st.slider(
            "Sample Index", 
            0, len(dataset_info['X_test']) - 1, 
            0,
            help="Choose a sample from the test set"
        )
        
        sample_data = dataset_info['X_test'].iloc[sample_idx]
        
        # Allow user to modify values
        user_input = {}
        for feature in dataset_info['feature_names']:
            user_input[feature] = st.number_input(
                feature,
                value=float(sample_data[feature]),
                format="%.4f",
                key=f"input_{feature}"
            )
        
        input_df = pd.DataFrame([user_input])
        
    with col2:
        st.write("**Prediction:**")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        predicted_class = dataset_info['target_names'][prediction]
        confidence = probabilities[prediction]
        
        st.success(f"**Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.4f}")
        
        # Show all probabilities
        st.write("**All Probabilities:**")
        prob_df = pd.DataFrame({
            'Class': dataset_info['target_names'],
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.dataframe(prob_df, use_container_width=True)
    
    return input_df


def render_explanation_tabs(
    model, 
    dataset_info: Dict[str, Any], 
    input_df: pd.DataFrame
):
    """Render explanation method tabs"""
    explanation_method = st.selectbox(
        "Choose Explanation Method",
        ["SHAP", "LIME", "Feature Importance", "All Methods"],
        help="Select the explanation method to understand model predictions"
    )
    
    return explanation_method


def show_loading_message(message: str):
    """Show a loading message"""
    return st.spinner(message)


def show_success_message(message: str):
    """Show a success message"""
    st.success(message)


def show_error_message(message: str):
    """Show an error message"""
    st.error(message)


def show_info_message(message: str):
    """Show an info message"""
    st.info(message)


def show_warning_message(message: str):
    """Show a warning message"""
    st.warning(message)
