"""
Streamlit UI components for ML Evaluation App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
import random
import datetime
from typing import Dict, Any, Optional
from utils.config import APP_TITLE, DATASETS, MODELS
from utils.logging_config import get_logger

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
    st.title("� AI Explainer Pro - Model Insights & Evaluation")
    st.markdown("*A comprehensive tool for AI model interpretation and explainability assessment*")
    st.markdown("---")


def render_sidebar() -> Dict[str, str]:
    """Render sidebar controls and return user selections"""
    st.sidebar.header("⚙️ Configuration")
    
    # About section in sidebar
    with st.sidebar.expander("📋 About this Application"):
        st.markdown("""
        This application provides comprehensive model explainability tools including:
        - **🔍 SHAP** explanations with automatic fallback mechanisms
        - **🎯 LIME** local interpretable model explanations  
        - **📊 Feature Importance** analysis
        - **📈 Model Performance** evaluation and comparison
        - **🎨 Interactive Visualizations** for better understanding
        
        Choose a dataset and model below to get started!
        """)
    
    st.sidebar.divider()
    
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
    st.sidebar.subheader("� Model Selection")
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
          {model_config['description']}""")

    # Track changes in dataset/model selection - only trigger alerts on actual user changes
    if 'previous_dataset' not in st.session_state:
        st.session_state.previous_dataset = selected_dataset  # Initialize with current value
    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = selected_model  # Initialize with current value
    if 'initial_load_done' not in st.session_state:
        st.session_state.initial_load_done = False
      # Only check for changes after initial load is done
    if st.session_state.initial_load_done:
        # Check if dataset or model actually changed from user interaction
        dataset_changed = st.session_state.previous_dataset != selected_dataset
        model_changed = st.session_state.previous_model != selected_model
        
        if dataset_changed or model_changed:
            # Update stored values
            st.session_state.previous_dataset = selected_dataset
            st.session_state.previous_model = selected_model
            # Set flag to allow alerts to show for this change
            trigger_alert()
            
            # Show success messages for what changed
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            if dataset_changed and selected_dataset:
                dataset_name = DATASETS[selected_dataset]['name']
                show_success_message(f"Dataset changed to '{dataset_name}' at {timestamp}!")
            if model_changed and selected_model:
                model_name = MODELS[selected_model]['name']
                show_success_message(f"Model changed to '{model_name}' at {timestamp}!")
    else:
        # Mark initial load as complete
        st.session_state.initial_load_done = True

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
    """Render enhanced prediction section with better UI/UX"""
    
    # Create tabs for different prediction modes
    pred_tab1, pred_tab2, pred_tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions", "🔄 Interactive Prediction"])
    
    with pred_tab1:
        st.markdown("### 🎯 Single Sample Prediction")
        st.markdown("Select a sample and modify its features to see how predictions change.")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**🔧 Feature Configuration**")
            
            # Sample selection with better UX
            sample_source = st.radio(
                "Choose sample source:",
                ["📖 Test Set Sample", "🎲 Random Sample", "✏️ Manual Input"],
                horizontal=True
            )
            
            if sample_source == "📖 Test Set Sample":
                sample_idx = st.slider(
                    "Sample Index", 
                    0, len(dataset_info['X_test']) - 1, 
                    0,
                    help="Choose a sample from the test set"
                )
                sample_data = dataset_info['X_test'].iloc[sample_idx]
                st.info(f"📍 Selected sample {sample_idx} from test set")
                
            elif sample_source == "🎲 Random Sample":
                if st.button("🎲 Generate Random Sample"):
                    sample_data = dataset_info['X_test'].sample(1).iloc[0]
                    st.success("🎲 Random sample generated!")
                else:
                    sample_data = dataset_info['X_test'].sample(1).iloc[0]
                    
            else:  # Manual Input
                sample_data = dataset_info['X_test'].iloc[0]  # Use as template
                st.info("✏️ Manually configure all feature values below")
            
            # Feature input with enhanced UI
            user_input = {}
            feature_cols = st.columns(2)
            
            for i, feature in enumerate(dataset_info['feature_names']):
                with feature_cols[i % 2]:
                    user_input[feature] = st.number_input(
                        f"🔹 {feature}",
                        value=float(sample_data[feature]),
                        format="%.4f",
                        key=f"single_input_{feature}",
                        help=f"Adjust {feature} value"
                    )
            
            input_df = pd.DataFrame([user_input])
            
        with col2:
            st.markdown("**🎯 Prediction Results**")
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            predicted_class = dataset_info['target_names'][prediction]
            confidence = probabilities[prediction]
            
            # Enhanced prediction display
            st.markdown("#### 🏆 Predicted Class")
            st.success(f"**{predicted_class}**")
            
            st.markdown("#### 📊 Confidence Score")
            st.progress(confidence, text=f"{confidence:.2%}")
            
            # Probability distribution with color coding
            st.markdown("#### 🎨 Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': dataset_info['target_names'],
                'Probability': probabilities,
                'Percentage': [f"{prob:.1%}" for prob in probabilities]
            }).sort_values('Probability', ascending=False)
            
            # Color-code the top prediction
            def color_rows(row):
                if row.name == 0:  # Top prediction
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            styled_df = prob_df.style.apply(color_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with pred_tab2:
        st.markdown("### 📊 Enhanced Batch Predictions")
        st.markdown("Generate predictions for multiple samples with advanced analysis.")
        
        # Batch prediction configuration
        batch_col1, batch_col2 = st.columns([2, 1])
        
        with batch_col1:
            st.markdown("**⚙️ Batch Configuration**")
            
            batch_mode = st.selectbox(
                "Select batch mode:",
                ["🎲 Random Samples", "📝 Custom Range", "📤 Upload Data"],
                help="Choose how to generate batch predictions"
            )
            
            if batch_mode == "🎲 Random Samples":
                n_samples = st.slider("Number of samples", 5, 50, 10, step=5)
                
                if st.button("🚀 Generate Batch Predictions", type="primary"):
                    with st.spinner("🔄 Generating predictions..."):
                        random_samples = dataset_info['X_test'].sample(n_samples)
                        batch_preds = model.predict(random_samples)
                        batch_probs = model.predict_proba(random_samples)
                        
                        # Store results in session state
                        st.session_state.batch_results = {
                            'samples': random_samples,
                            'predictions': batch_preds,
                            'probabilities': batch_probs,
                            'timestamp': datetime.datetime.now()
                        }
                        
                        trigger_alert()
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        show_success_message(f"✅ Generated {n_samples} batch predictions at {timestamp}!")
                        
            elif batch_mode == "📝 Custom Range":
                start_idx = st.number_input("Start index", 0, len(dataset_info['X_test'])-1, 0)
                end_idx = st.number_input("End index", start_idx+1, len(dataset_info['X_test']), min(start_idx+10, len(dataset_info['X_test'])))
                
                if st.button("🎯 Predict Range", type="primary"):
                    with st.spinner("🔄 Processing range..."):
                        range_samples = dataset_info['X_test'].iloc[start_idx:end_idx]
                        batch_preds = model.predict(range_samples)
                        batch_probs = model.predict_proba(range_samples)
                        
                        st.session_state.batch_results = {
                            'samples': range_samples,
                            'predictions': batch_preds,
                            'probabilities': batch_probs,
                            'timestamp': datetime.datetime.now()
                        }
                        
                        trigger_alert()
                        show_success_message(f"✅ Processed samples {start_idx} to {end_idx}!")
                        
            else:  # Upload Data
                uploaded_file = st.file_uploader(
                    "Upload CSV file", 
                    type=['csv'],
                    help="Upload a CSV file with the same features as the training data"
                )
                
                if uploaded_file is not None:
                    try:
                        upload_df = pd.read_csv(uploaded_file)
                        st.info(f"📁 Uploaded file with {len(upload_df)} samples")
                        
                        if st.button("🔍 Predict Uploaded Data", type="primary"):
                            # Validate columns
                            missing_cols = set(dataset_info['feature_names']) - set(upload_df.columns)
                            if missing_cols:
                                st.error(f"❌ Missing columns: {missing_cols}")
                            else:
                                with st.spinner("🔄 Processing uploaded data..."):
                                    upload_samples = upload_df[dataset_info['feature_names']]
                                    batch_preds = model.predict(upload_samples)
                                    batch_probs = model.predict_proba(upload_samples)
                                    
                                    st.session_state.batch_results = {
                                        'samples': upload_samples,
                                        'predictions': batch_preds,
                                        'probabilities': batch_probs,
                                        'timestamp': datetime.datetime.now()
                                    }
                                    
                                    trigger_alert()
                                    show_success_message(f"✅ Processed {len(upload_samples)} uploaded samples!")
                                    
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
        
        with batch_col2:
            st.markdown("**📈 Batch Analytics**")
            
            # Model confidence distribution
            if len(dataset_info['X_test']) > 0:
                all_probs = model.predict_proba(dataset_info['X_test'])
                max_probs = np.max(all_probs, axis=1)
                
                st.markdown("**🎯 Confidence Distribution**")
                fig_hist = px.histogram(
                    x=max_probs, 
                    title="Model Confidence",
                    labels={'x': 'Confidence', 'y': 'Count'},
                    nbins=15,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Quick stats
                st.markdown("**📊 Quick Stats**")
                st.metric("Avg Confidence", f"{np.mean(max_probs):.2%}")
                st.metric("Min Confidence", f"{np.min(max_probs):.2%}")
                st.metric("Max Confidence", f"{np.max(max_probs):.2%}")
        
        # Display batch results
        if 'batch_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 📋 Batch Results")
            
            results = st.session_state.batch_results
            
            # Create enhanced results dataframe
            results_df = pd.DataFrame({
                'Sample_ID': range(len(results['samples'])),
                'Predicted_Class': [dataset_info['target_names'][pred] for pred in results['predictions']],
                'Confidence': [f"{probs[pred]:.2%}" for pred, probs in zip(results['predictions'], results['probabilities'])],
                'Top_2_Classes': [f"{dataset_info['target_names'][np.argsort(probs)[-1]]} ({probs[np.argsort(probs)[-1]]:.2%}), {dataset_info['target_names'][np.argsort(probs)[-2]]} ({probs[np.argsort(probs)[-2]]:.2%})" for probs in results['probabilities']]
            })
            
            # Display options
            display_col1, display_col2 = st.columns([3, 1])
            
            with display_col1:
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
            with display_col2:
                st.markdown("**📊 Summary**")
                st.metric("Total Samples", len(results_df))
                st.metric("Generated At", results['timestamp'].strftime("%H:%M:%S"))
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with pred_tab3:
        st.markdown("### 🔄 Interactive Prediction Explorer")
        st.markdown("Explore how changing individual features affects predictions in real-time.")
        
        # Feature impact analysis
        if st.button("🔍 Analyze Feature Impact", type="primary"):
            with st.spinner("🔄 Analyzing feature impact..."):
                base_sample = dataset_info['X_test'].iloc[0]
                base_pred = model.predict([base_sample])[0]
                base_prob = model.predict_proba([base_sample])[0]
                
                st.markdown("#### 📈 Feature Impact Analysis")
                
                impact_data = []
                for feature in dataset_info['feature_names']:
                    # Test different variations of this feature
                    variations = []
                    feature_range = dataset_info['X_test'][feature].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                    
                    for percentile, value in feature_range.items():
                        modified_sample = base_sample.copy()
                        modified_sample[feature] = value
                        
                        new_pred = model.predict([modified_sample])[0]
                        new_prob = model.predict_proba([modified_sample])[0]
                        
                        variations.append({
                            'percentile': f"{percentile*100:.0f}th",
                            'value': value,
                            'prediction': dataset_info['target_names'][new_pred],
                            'confidence': new_prob[new_pred],
                            'changed': new_pred != base_pred
                        })
                    
                    impact_data.append({
                        'feature': feature,
                        'variations': variations,
                        'sensitivity': sum(1 for v in variations if v['changed']) / len(variations)
                    })
                
                # Sort by sensitivity
                impact_data.sort(key=lambda x: x['sensitivity'], reverse=True)
                
                # Display most sensitive features
                st.markdown("**🎯 Most Sensitive Features**")
                for i, feature_data in enumerate(impact_data[:3]):
                    with st.expander(f"🔹 {feature_data['feature']} (Sensitivity: {feature_data['sensitivity']:.0%})"):
                        sens_df = pd.DataFrame(feature_data['variations'])
                        sens_df['Impact'] = sens_df['changed'].apply(lambda x: '🔄 Changed' if x else '✅ Stable')
                        st.dataframe(sens_df[['percentile', 'value', 'prediction', 'confidence', 'Impact']], 
                                   use_container_width=True, hide_index=True)
                
                trigger_alert()
                show_success_message("🎯 Feature impact analysis completed!")
    
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


def show_floating_alert(message: str, alert_type: str = "success", duration: int = 3):
    """Show a floating alert using the best available method"""
    
    # Try to use Streamlit's toast feature if available (Streamlit 1.27+)
    try:
        if hasattr(st, 'toast'):
            icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            icon = icons.get(alert_type, "ℹ️")
            st.toast(f"{icon} {message}", icon=icon)
            return True
    except (AttributeError, TypeError, Exception):
        pass
    
    # Check if we should show this alert based on trigger flag
    if 'show_alert_flag' not in st.session_state:
        st.session_state.show_alert_flag = False
    
    # Only show alert if the flag is set (meaning something triggered it)
    if st.session_state.show_alert_flag:
        # Reset the flag immediately after showing
        st.session_state.show_alert_flag = False
        
        # Display immediately using standard Streamlit components
        icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
        icon = icons.get(alert_type, "ℹ️")
        formatted_message = f"{icon} {message}"
        
        if alert_type == "success":
            st.success(formatted_message)
        elif alert_type == "error":
            st.error(formatted_message)
        elif alert_type == "warning":
            st.warning(formatted_message)
        else:
            st.info(formatted_message)
    
    return True


def display_session_alerts():
    """Display active alerts from session state - simplified version"""
    # This function is now simplified since we're using immediate display
    # in show_floating_alert instead of session-based persistence
    pass


def show_success_message(message: str, duration: int = 3):
    """Show a floating success message"""
    show_floating_alert(message, "success", duration)


def show_error_message(message: str, duration: int = 0):
    """Show an error message"""
    if duration > 0:
        show_floating_alert(message, "error", duration)
    else:
        st.error(message)


def show_info_message(message: str, duration: int = 5):
    """Show a floating info message"""
    show_floating_alert(message, "info", duration)


def show_warning_message(message: str, duration: int = 4):
    """Show a floating warning message"""
    show_floating_alert(message, "warning", duration)


def init_and_display_alerts():
    """Initialize alert system - simplified version"""
    # Initialize the last alert tracker if not present
    if 'last_alert' not in st.session_state:
        st.session_state.last_alert = None


def trigger_alert():
    """Set flag to allow the next alert to be shown"""
    st.session_state.show_alert_flag = True


def clear_alerts():
    """Clear the alert trigger flag"""
    if 'show_alert_flag' in st.session_state:
        st.session_state.show_alert_flag = False


def render_main_content_tabs(
    model, 
    dataset_info: Dict[str, Any], 
    train_info: Dict[str, Any], 
    eval_info: Dict[str, Any]
):
    """Render main content area with organized tabs for better UX"""
    
    # Initialize and display any active alerts
    init_and_display_alerts()
      # Create main tabs with enhanced emojis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dataset Overview",
        "🎯 Smart Predictions", 
        "📈 Model Performance", 
        "🔍 AI Explanations", 
        "🔬 Feature Analysis",
        "� User Feedback", 
        "📁 Export & Share"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview & Insights")
        st.markdown("*Comprehensive analysis of your selected dataset*")
        st.markdown("---")
        
        # Render dataset information
        render_dataset_info(dataset_info)    
    with tab2:
        st.header("🎯 Smart Predictions & Analysis")
        st.markdown("*Advanced prediction tools with interactive exploration capabilities*")
        st.markdown("---")
        
        input_df = render_prediction_section(model, dataset_info)
        
        # Store input in session state for explanation tab
        if input_df is not None:
            st.session_state.current_input_df = input_df
    with tab3:
        st.header("📈 Comprehensive Model Performance")
        st.markdown("*Detailed analysis of model performance across different metrics*")
        st.markdown("---")
        
        # Performance metrics
        render_model_performance(train_info, eval_info)
        
        st.divider()
        
        # Additional performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Accuracy by Class")
            
            # Class-wise accuracy
            y_pred = model.predict(dataset_info['X_test'])
            y_true = dataset_info['y_test']
            
            class_accuracy = {}
            for i, class_name in enumerate(dataset_info['target_names']):
                class_mask = y_true == i
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
                    class_accuracy[class_name] = class_acc
            
            acc_df = pd.DataFrame(list(class_accuracy.items()), columns=['Class', 'Accuracy'])
            fig_acc = px.bar(acc_df, x='Class', y='Accuracy', title="Accuracy by Class")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            st.subheader("🔄 Cross-Validation Results")
            
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, dataset_info['X_train'], dataset_info['y_train'], cv=5)
            
            st.metric("CV Mean", f"{cv_scores.mean():.4f}")
            st.metric("CV Std", f"{cv_scores.std():.4f}")
            
            # CV scores visualization
            cv_df = pd.DataFrame({'Fold': range(1, 6), 'Score': cv_scores})
            fig_cv = px.line(cv_df, x='Fold', y='Score', title="Cross-Validation Scores", markers=True)
            st.plotly_chart(fig_cv, use_container_width=True)    
    with tab4:
        st.header("🔍 Model Explanations")
        st.markdown("Understand how the model makes decisions with various explanation techniques.")
        
        # Get input data for explanations from predictions tab
        # Use session state to share data between tabs
        if 'current_input_df' not in st.session_state:
            st.info("⚠️ Please make a prediction in the 'Predictions' tab first to generate explanations.")
            return None
        
        input_df = st.session_state.current_input_df
        
        # Explanation method selection
        explanation_method = st.selectbox(
            "Choose Explanation Method",
            ["SHAP", "LIME", "Feature Importance", "All Methods"],
            help="Select the explanation method to understand model predictions"
        )
        
        # Generate explanations based on selection
        if explanation_method in ["SHAP", "All Methods"]:
            render_shap_explanation_tab(model, dataset_info, input_df)
        
        if explanation_method in ["LIME", "All Methods"]:
            render_lime_explanation_tab(model, dataset_info, input_df)
        
        if explanation_method in ["Feature Importance", "All Methods"]:
            render_feature_importance_tab(model, dataset_info)
        
        # Explanation comparison section
        st.divider()
        st.subheader("🔍 Explanation Method Comparison")
        
        with st.expander("ℹ️ Method Comparison Guide"):
            st.markdown("""
            | Method | Best For | Advantages | Limitations |
            |--------|----------|------------|-------------|
            | **SHAP** | Tree-based models, global + local explanations | Theoretically grounded, consistent | Can be slow for large datasets |
            | **LIME** | Any model type, local explanations | Model-agnostic, intuitive | Instability across runs |
            | **Feature Importance** | Tree-based models, global understanding | Fast, simple to interpret | Only for compatible models |
            """)
    with tab5:
        st.header("� Feature Analysis")
        st.markdown("Deep dive into feature relationships and importance patterns.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔗 Feature Correlations")
            
            # Correlation heatmap
            corr_matrix = dataset_info['X'].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.subheader("📊 Feature Distributions")
            
            # Feature distribution plots
            selected_feature = st.selectbox(
                "Select feature to analyze:",
                dataset_info['feature_names']
            )
            
            if selected_feature:
                fig_dist = px.histogram(
                    dataset_info['X'], 
                    x=selected_feature,
                    color=dataset_info['y'],
                    title=f"Distribution of {selected_feature} by Class",
                    marginal="box"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature importance section (if available)
        st.divider()
        st.subheader("⭐ Global Feature Importance")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': dataset_info['feature_names'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance Ranking"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    with tab6:
        st.header("👥 User Feedback")
        st.markdown("Help us improve AI explainability by sharing your experience.")
        
        # User feedback section
        render_user_feedback_advanced()
        
        # Explainability metrics
        st.divider()
        st.subheader("📊 Explainability Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Complexity", f"{len(dataset_info['feature_names'])} features")
        
        with col2:
            if hasattr(model, 'n_estimators'):
                st.metric("Model Estimators", model.n_estimators)
            elif hasattr(model, 'max_iter'):
                st.metric("Max Iterations", model.max_iter)
            else:
                st.metric("Model Type", train_info['model_name'])
        
        with col3:
            prediction_consistency = eval_info['test_score']
            st.metric("Prediction Consistency", f"{prediction_consistency:.3f}")
    with tab7:
        st.header("📁 Export & Reports")
        st.markdown("Download your analysis results and generate comprehensive reports.")
        
        render_export_section(model, dataset_info, train_info, eval_info)
    
    return input_df


def render_user_feedback_advanced():
    """Advanced user feedback section for explainability studies"""
    
    with st.expander("📝 Explainability Feedback Form"):
        col1, col2 = st.columns(2)
        
        with col1:
            trust_rating = st.slider(
                "How much do you trust the model's predictions?",
                1, 10, 5,
                help="1 = Don't trust at all, 10 = Complete trust"
            )
            
            understanding_rating = st.slider(
                "How well do you understand the model's decisions?",
                1, 10, 5,
                help="1 = Don't understand at all, 10 = Complete understanding"
            )
            
            usefulness_rating = st.slider(
                "How useful are the explanations for your task?",
                1, 10, 5,
                help="1 = Not useful at all, 10 = Extremely useful"
            )
        
        with col2:
            explanation_preference = st.selectbox(
                "Which explanation method do you prefer?",
                ["SHAP", "LIME", "Feature Importance", "Multiple methods", "None"]
            )
            
            improvement_areas = st.multiselect(
                "What aspects need improvement?",
                ["Clarity", "Speed", "Accuracy", "Visual design", "Interactivity", "Coverage"]
            )
            
            feedback_text = st.text_area(
                "Additional feedback:",
                placeholder="Share any specific thoughts or suggestions..."
            )
        
        if st.button("Submit Feedback"):
            feedback_data = {
                'trust': trust_rating,
                'understanding': understanding_rating,
                'usefulness': usefulness_rating,
                'preference': explanation_preference,
                'improvements': improvement_areas,
                'text': feedback_text,
                'timestamp': pd.Timestamp.now()
            }
            
            if 'user_feedback' not in st.session_state:
                st.session_state.user_feedback = []
            
            st.session_state.user_feedback.append(feedback_data)
            st.success("Thank you for your feedback!")


def render_export_section(model, dataset_info, train_info, eval_info):
    """Render export and reporting section"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Quick Export Options")
        
        # Model performance summary
        if st.button("📈 Export Performance Report"):
            performance_data = {
                'model_name': train_info['model_name'],
                'dataset': dataset_info['config']['name'],
                'train_accuracy': train_info['train_score'],
                'test_accuracy': eval_info['test_score'],
                'classification_report': eval_info['classification_report']
            }
            
            report_json = pd.json_normalize(performance_data).to_json(indent=2)
            st.download_button(
                "Download Performance Report",
                report_json,
                file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Dataset summary
        if st.button("📋 Export Dataset Summary"):
            dataset_summary = {
                'name': dataset_info['config']['name'],
                'shape': dataset_info['X'].shape,
                'features': dataset_info['feature_names'].tolist(),
                'classes': dataset_info['target_names'].tolist(),
                'class_distribution': pd.Series(dataset_info['y']).value_counts().to_dict()
            }
            
            summary_json = pd.json_normalize(dataset_summary).to_json(indent=2)
            st.download_button(
                "Download Dataset Summary",
                summary_json,
                file_name=f"dataset_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.subheader("📄 Comprehensive Reports")
        
        if st.button("📋 Generate Full Analysis Report"):
            st.info("Generating comprehensive analysis report...")
            
            # Create comprehensive report
            report_content = f"""
# ML Model Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- **Dataset:** {dataset_info['config']['name']}
- **Model Type:** {train_info['model_name']}
- **Training Samples:** {len(dataset_info['X_train'])}
- **Test Samples:** {len(dataset_info['X_test'])}

## Performance Summary
- **Training Accuracy:** {train_info['train_score']:.4f}
- **Test Accuracy:** {eval_info['test_score']:.4f}

## Dataset Information
- **Total Features:** {len(dataset_info['feature_names'])}
- **Total Classes:** {len(dataset_info['target_names'])}
- **Feature Names:** {', '.join(dataset_info['feature_names'])}
- **Class Names:** {', '.join(dataset_info['target_names'])}

## Model Analysis
This report was generated using the AI Explainer Pro application.
The model shows {'good' if eval_info['test_score'] > 0.8 else 'moderate' if eval_info['test_score'] > 0.6 else 'limited'} performance on the test dataset.

---
*Report generated by AI Explainer Pro*
            """
            
            st.download_button(
                "📄 Download Full Report",
                report_content,
                file_name=f"full_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Session data export
        if st.button("💾 Export Session Data"):
            session_data = {
                'model_info': {
                    'name': train_info['model_name'],
                    'dataset': dataset_info['config']['name'],
                    'performance': {
                        'train_score': train_info['train_score'],
                        'test_score': eval_info['test_score']
                    }
                },
                'user_feedback': st.session_state.get('user_feedback', []),
                'session_timestamp': pd.Timestamp.now().isoformat()
            }
            
            session_json = pd.json_normalize(session_data).to_json(indent=2)
            st.download_button(
                "Download Session Data",
                session_json,
                file_name=f"session_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def render_shap_explanation_tab(model, dataset_info, input_df):
    """Render SHAP explanation in the explanations tab"""
    st.subheader("🎯 SHAP Explanation")
    
    # Import explanation handler functions
    from utils.explanation_handler import generate_shap_explanation, create_shap_waterfall_plot
    from utils.logging_config import get_logger
    
    explanation_logger = get_logger('explanation')
    explanation_logger.info("Starting SHAP explanation generation")
    
    try:
        with st.spinner("Generating SHAP explanation..."):
            shap_values, expected_value, success, error_msg = generate_shap_explanation(
                model, 
                dataset_info['X_train'], 
                input_df, 
                dataset_info['feature_names']
            )
        
        if success:
            st.success("SHAP explanation generated successfully!")
            
            # Create waterfall plot
            fig, plot_success, plot_error = create_shap_waterfall_plot(
                shap_values, expected_value, input_df.values[0], dataset_info['feature_names']
            )
            
            if plot_success:
                st.pyplot(fig)
            else:
                st.error(f"Error creating waterfall plot: {plot_error}")
              # Show feature contributions
            st.subheader("📊 Feature Contributions")
            if shap_values is not None:
                contrib_df = pd.DataFrame({
                    'Feature': dataset_info['feature_names'],
                    'Value': input_df.values[0],
                    'SHAP Value': shap_values,
                    'Contribution': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in shap_values]
                })
                contrib_df['Abs SHAP Value'] = np.abs(contrib_df['SHAP Value'])
                contrib_df = contrib_df.sort_values('Abs SHAP Value', ascending=False)
                st.dataframe(contrib_df.round(4))
            else:
                st.error("SHAP values could not be computed")
            
            explanation_logger.info(f"SHAP explanation completed successfully")
            
        else:
            st.error(f"SHAP explanation failed: {error_msg}")
            st.info("💡 **Suggestion**: Try using LIME explanations instead, which work with all model types.")
            
    except Exception as e:
        explanation_logger.error(f"Unexpected error in SHAP explanation: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")


def render_lime_explanation_tab(model, dataset_info, input_df):
    """Render LIME explanation in the explanations tab"""
    st.subheader("🧪 LIME Explanation")
    
    # Import explanation handler functions
    from utils.explanation_handler import generate_lime_explanation
    from utils.logging_config import get_logger
    
    explanation_logger = get_logger('explanation')
    explanation_logger.info("Starting LIME explanation generation")
    
    try:
        with st.spinner("Generating LIME explanation..."):
            exp, success, error_msg = generate_lime_explanation(
                model,
                dataset_info['X_train'],
                input_df.iloc[0],
                dataset_info['feature_names'],
                dataset_info['target_names']
            )
        
        if success and exp is not None:
            st.success("LIME explanation generated successfully!")
            
            # Display LIME results
            explanations = exp.as_list()
            
            lime_df = pd.DataFrame(explanations, columns=['Feature', 'Impact'])
            lime_df = lime_df.sort_values('Impact', key=abs, ascending=False)
            
            st.dataframe(lime_df)
            
            # Show LIME plot
            try:
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
            except Exception as plot_error:
                st.warning(f"Could not display LIME plot: {plot_error}")
            
            explanation_logger.info("LIME explanation completed successfully")
            
        else:
            st.error(f"LIME explanation failed: {error_msg}")
            
    except Exception as e:
        explanation_logger.error(f"Unexpected error in LIME explanation: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")


def render_feature_importance_tab(model, dataset_info):
    """Render feature importance in the explanations tab"""
    st.subheader("📈 Feature Importance")
    
    # Import explanation handler functions
    from utils.explanation_handler import get_feature_importance
    
    try:
        importances, success, error_msg = get_feature_importance(model)
        
        if success:
            st.success("Feature importance extracted successfully!")
            
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
            st.info(f"Feature importance not available: {error_msg}")
            
    except Exception as e:
        st.error(f"Error getting feature importance: {str(e)}")
