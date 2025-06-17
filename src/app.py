"""
AI Explainer Pro - Model Insights & Evaluation Suite
A comprehensive tool for model interpretation and explainability assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os
import datetime
from pathlib import Path

# Add the src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import our modular components
from utils.logging_config import setup_logging
from utils.data_handler import load_dataset, get_available_datasets
from utils.model_handler import create_model, train_model, evaluate_model
from utils.explanation_handler import (
    generate_shap_explanation, 
    create_shap_waterfall_plot,
    generate_lime_explanation,
    get_feature_importance
)
from components.ui_components import (
    setup_page_config,
    render_header,
    render_sidebar,
    render_dataset_info,    render_model_performance,
    render_prediction_section,
    render_explanation_tabs,
    render_main_content_tabs,
    show_loading_message,
    show_success_message,
    show_error_message,
    show_info_message,
    show_warning_message,
    trigger_alert
)

warnings.filterwarnings('ignore', category=UserWarning)

# Initialize logging
app_logger, model_logger, explanation_logger, user_study_logger = setup_logging()
app_logger.info("AI Explainer Pro Application Starting")

# Set up page configuration
setup_page_config()

# Initialize session state for user feedback
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

if 'user_study_data' not in st.session_state:
    st.session_state.user_study_data = []

# Main application
def main():
    """Main application function"""
    app_logger.info("Main application started")
    
    # Render header
    render_header()
    
    # Render sidebar and get user selections
    selections = render_sidebar()
    selected_dataset = selections['dataset']
    selected_model = selections['model']
    
    if not selected_dataset or not selected_model:
        st.info("👈 Please select a dataset and model from the sidebar to begin.")
        return
        
    app_logger.info(f"User selected dataset: {selected_dataset}, model: {selected_model}")
    
    # Load dataset
    try:
        with show_loading_message("Loading dataset..."):
            dataset_info = load_dataset(selected_dataset)
        
    except Exception as e:
        app_logger.error(f"Error loading dataset {selected_dataset}: {str(e)}")
        trigger_alert()
        show_error_message(f"Error loading dataset: {str(e)}")
        return
        
    # Train model
    try:
        with show_loading_message("Training model..."):
            model = create_model(selected_model)
            train_info = train_model(model, dataset_info['X_train'], dataset_info['y_train'])
            eval_info = evaluate_model(model, dataset_info['X_test'], dataset_info['y_test'])
          
        # Use the new tabbed interface for main content
        st.divider()
        render_main_content_tabs(model, dataset_info, train_info, eval_info)
        
    except Exception as e:
        app_logger.error(f"Error training model {selected_model}: {str(e)}")
        trigger_alert()
        show_error_message(f"Error training model: {str(e)}")
        return

if __name__ == "__main__":
    main()
