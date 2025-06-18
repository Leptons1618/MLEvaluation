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
    render_dataset_info,
    render_model_performance,
    render_prediction_section,
    render_explanation_tabs,
    show_loading_message,
    show_success_message,
    show_error_message,
    show_info_message,
    show_warning_message,
    render_data_upload_page,
    render_navigation_sidebar
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
    
    # Navigation
    current_page = render_navigation_sidebar()
    
    if current_page == "data_upload":
        render_data_upload_page()
        return
    
    # Original model analysis page
    render_model_analysis_page()


def render_model_analysis_page():
    """Render the enhanced model analysis page"""
    # Render header
    render_header()
    
    # Dataset source selection with better UI
    st.markdown("### 🎯 Select Your Data Source")
    
    source_col1, source_col2 = st.columns(2)
    
    with source_col1:
        dataset_source = st.radio(
            "Choose dataset source:",
            options=["🗂️ Built-in Datasets", "📁 My Uploaded Dataset"],
            horizontal=False,
            help="Select whether to use built-in sample datasets or your own uploaded data"
        )
    
    with source_col2:
        if dataset_source == "📁 My Uploaded Dataset":
            if 'prepared_dataset' in st.session_state and st.session_state.get('use_uploaded_dataset', False):
                dataset_info = st.session_state.prepared_dataset
                st.success(f"✅ **Ready to use:** {dataset_info['config']['name']}")
                st.info(f"📊 {dataset_info['config']['features']} features, {dataset_info['config']['classes']} classes")
            else:
                st.warning("⚠️ **No dataset prepared**")
                st.info("👆 Go to 'Data Upload & Prep' to upload your data first")
        else:
            st.info("🗂️ **Built-in datasets available:**\n• Iris (flowers)\n• Wine (quality)\n• Breast Cancer (medical)")
    
    st.markdown("---")
    
    if dataset_source == "📁 My Uploaded Dataset":
        if 'prepared_dataset' not in st.session_state or not st.session_state.get('use_uploaded_dataset', False):
            st.error("❌ **No prepared dataset found**")
            st.markdown("""
            **Next Steps:**
            1. 👆 Navigate to **'📁 Data Upload & Prep'** page using the sidebar
            2. 📤 Upload your dataset (CSV or Excel)
            3. 🔧 Prepare it for machine learning
            4. 🔄 Return here to train models and generate explanations
            """)
            
            if st.button("🚀 Go to Data Upload Page", type="primary"):
                st.session_state.current_page = "data_upload"
                st.rerun()
            return
        
        # Use uploaded dataset
        dataset_info = st.session_state.prepared_dataset
        selected_dataset = "uploaded"
        
        st.success(f"🎯 **Using uploaded dataset:** {dataset_info['config']['name']}")
        
        # Model selection for uploaded dataset
        st.markdown("### 🤖 Model Selection")
        from utils.config import MODELS
        model_options = {name: config['name'] for name, config in MODELS.items()}
        
        model_col1, model_col2 = st.columns([2, 1])
        
        with model_col1:
            selected_model = st.selectbox(
                "Choose a machine learning model:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                help="Select the machine learning model to train and explain"
            )
        
        with model_col2:
            if selected_model:
                model_config = MODELS[selected_model]
                st.info(f"""
                **{model_config['name']}**
                Type: {model_config['type'].title()}
                Feature Importance: {'✓' if model_config['supports_feature_importance'] else '✗'}
                """)
        
        if not selected_model:
            st.info("� Please select a model to begin training and analysis.")
            return
        
    else:
        # Use built-in datasets with enhanced sidebar
        st.markdown("### ⚙️ Configuration")
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
            show_success_message(f"Dataset '{dataset_info['config']['name']}' loaded successfully!")
            
        except Exception as e:
            app_logger.error(f"Error loading dataset {selected_dataset}: {str(e)}")
            show_error_message(f"Error loading dataset: {str(e)}")
            return
    
    # Render dataset information
    render_dataset_info(dataset_info)
    
    # Train model
    try:
        with show_loading_message("Training model..."):
            model = create_model(selected_model)
            train_info = train_model(model, dataset_info['X_train'], dataset_info['y_train'])
            eval_info = evaluate_model(model, dataset_info['X_test'], dataset_info['y_test'])
        
        show_success_message(f"Model '{train_info['model_name']}' trained successfully!")
        
        # Render model performance
        render_model_performance(train_info, eval_info)
        
    except Exception as e:
        app_logger.error(f"Error training model {selected_model}: {str(e)}")
        show_error_message(f"Error training model: {str(e)}")
        return
    
    # Prediction section
    st.divider()
    input_df = render_prediction_section(model, dataset_info)
    
    if input_df is None:
        return
    
    # Explanation section
    st.divider()
    st.header("🧠 Model Explanations")
    
    explanation_method = render_explanation_tabs(model, dataset_info, input_df)
    
    # Generate explanations based on selection
    if explanation_method in ["SHAP", "All Methods"]:
        render_shap_explanation(model, dataset_info, input_df)
    
    if explanation_method in ["LIME", "All Methods"]:
        render_lime_explanation(model, dataset_info, input_df)
    
    if explanation_method in ["Feature Importance", "All Methods"]:
        render_feature_importance(model, dataset_info)
    
    # User feedback section
    render_user_feedback()


def render_shap_explanation(model, dataset_info, input_df):
    """Render SHAP explanation section"""
    st.subheader("🎯 SHAP Explanation")
    explanation_logger.info("Starting SHAP explanation generation")
    
    try:
        with show_loading_message("Generating SHAP explanation..."):
            shap_values, expected_value, success, error_msg = generate_shap_explanation(
                model, 
                dataset_info['X_train'], 
                input_df, 
                dataset_info['feature_names']
            )
        
        if success:
            show_success_message("SHAP explanation generated successfully!")
            
            # Create waterfall plot
            fig, plot_success, plot_error = create_shap_waterfall_plot(
                shap_values, expected_value, input_df.values[0], dataset_info['feature_names']
            )
            
            if plot_success:
                st.pyplot(fig)
            else:
                show_error_message(f"Error creating waterfall plot: {plot_error}")
            
            # Show feature contributions
            st.subheader("📊 Feature Contributions")
            contrib_df = pd.DataFrame({
                'Feature': dataset_info['feature_names'],
                'Value': input_df.values[0],
                'SHAP Value': shap_values,
                'Contribution': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in shap_values]
            })
            contrib_df['Abs SHAP Value'] = np.abs(contrib_df['SHAP Value'])
            contrib_df = contrib_df.sort_values('Abs SHAP Value', ascending=False)
            st.dataframe(contrib_df.round(4))
            
            explanation_logger.info(f"SHAP explanation completed successfully")
            
        else:
            show_error_message(f"SHAP explanation failed: {error_msg}")
            show_info_message("💡 **Suggestion**: Try using LIME explanations instead, which work with all model types.")
            
    except Exception as e:
        explanation_logger.error(f"Unexpected error in SHAP explanation: {str(e)}")
        show_error_message(f"Unexpected error: {str(e)}")


def render_lime_explanation(model, dataset_info, input_df):
    """Render LIME explanation section"""
    st.subheader("🧪 LIME Explanation")
    explanation_logger.info("Starting LIME explanation generation")
    
    try:
        with show_loading_message("Generating LIME explanation..."):
            exp, success, error_msg = generate_lime_explanation(
                model,
                dataset_info['X_train'],
                input_df.iloc[0],
                dataset_info['feature_names'],
                dataset_info['target_names']
            )
        
        if success:
            show_success_message("LIME explanation generated successfully!")
            
            # Display LIME results
            explanations = exp.as_list()
            
            lime_df = pd.DataFrame(explanations, columns=['Feature', 'Impact'])
            lime_df = lime_df.sort_values('Impact', key=abs, ascending=False)
            
            st.dataframe(lime_df)
            
            # Show LIME plot
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            
            explanation_logger.info("LIME explanation completed successfully")
            
        else:
            show_error_message(f"LIME explanation failed: {error_msg}")
            
    except Exception as e:
        explanation_logger.error(f"Unexpected error in LIME explanation: {str(e)}")
        show_error_message(f"Unexpected error: {str(e)}")


def render_feature_importance(model, dataset_info):
    """Render feature importance section"""
    st.subheader("📈 Feature Importance")
    
    try:
        importances, success, error_msg = get_feature_importance(model)
        
        if success:
            show_success_message("Feature importance extracted successfully!")
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': dataset_info['feature_names'],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display as dataframe
            st.dataframe(importance_df)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            st.pyplot(fig)
            
        else:
            show_info_message(f"Feature importance not available: {error_msg}")
            
    except Exception as e:
        show_error_message(f"Error getting feature importance: {str(e)}")


def render_user_feedback():
    """Render user feedback section"""
    st.divider()
    st.header("💬 Feedback")
    
    with st.expander("📝 Share Your Feedback"):
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General", "Bug Report", "Feature Request", "Explanation Quality"]
        )
        
        feedback_text = st.text_area(
            "Your feedback:",
            help="Help us improve the application by sharing your thoughts"
        )
        
        if st.button("Submit Feedback"):
            if feedback_text:
                feedback_entry = {
                    'type': feedback_type,
                    'text': feedback_text,
                    'timestamp': pd.Timestamp.now()
                }
                st.session_state.feedback_data.append(feedback_entry)
                user_study_logger.info(f"Feedback received: {feedback_type}")
                show_success_message("Thank you for your feedback!")
            else:
                show_warning_message("Please enter some feedback text.")


if __name__ == "__main__":
    main()
