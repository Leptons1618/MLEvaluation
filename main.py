import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import json
import warnings
import logging
import sys
import os
import time
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
def setup_logging():
    """Set up comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f"ml_evaluation_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # File handler for errors only
    error_handler = logging.FileHandler(
        log_dir / f"ml_evaluation_errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Console handler for development (only warnings and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    app_logger = logging.getLogger('ml_evaluation')
    model_logger = logging.getLogger('ml_evaluation.model')
    explanation_logger = logging.getLogger('ml_evaluation.explanation')
    user_study_logger = logging.getLogger('ml_evaluation.user_study')
    
    return app_logger, model_logger, explanation_logger, user_study_logger

# Initialize logging
app_logger, model_logger, explanation_logger, user_study_logger = setup_logging()

# Log application startup
app_logger.info("ML Evaluation Application Starting")
app_logger.info(f"Python version: {sys.version}")
app_logger.info(f"Working directory: {os.getcwd()}")

# Set page config
st.set_page_config(page_title="AI Explainer Pro - Model Insights & Evaluation", layout="wide")

st.title("🔍 AI Explainer Pro - Model Insights & Evaluation")
st.markdown("*Making AI decisions transparent and understandable for everyone*")

# Initialize session state for user feedback
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

if 'user_study_data' not in st.session_state:
    st.session_state.user_study_data = []

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Iris", "Wine", "Breast Cancer"],
    help="Select the dataset for model training and explanation"
)

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"],
    help="Select the ML model to train and explain"
)

# Log user configuration choices
app_logger.info(f"User configuration - Dataset: {dataset_choice}, Model: {model_choice}")

# Load dataset based on selection
@st.cache_data
def load_dataset(dataset_name):
    """Load dataset with logging"""
    app_logger.info(f"Loading dataset: {dataset_name}")
    try:
        if dataset_name == "Iris":
            data = load_iris()
            app_logger.debug(f"Iris dataset loaded - samples: {data.data.shape[0]}, features: {data.data.shape[1]}")
            return data.data, data.target, data.feature_names, data.target_names
        elif dataset_name == "Wine":
            data = load_wine()
            app_logger.debug(f"Wine dataset loaded - samples: {data.data.shape[0]}, features: {data.data.shape[1]}")
            return data.data, data.target, data.feature_names, data.target_names
        elif dataset_name == "Breast Cancer":
            data = load_breast_cancer()
            app_logger.debug(f"Breast Cancer dataset loaded - samples: {data.data.shape[0]}, features: {data.data.shape[1]}")
            return data.data, data.target, data.feature_names, data.target_names
        else:
            app_logger.error(f"Unknown dataset: {dataset_name}")
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        app_logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        raise

# Get model based on selection
@st.cache_resource
def get_model(model_name):
    """Get model with logging"""
    model_logger.info(f"Creating model: {model_name}")
    try:
        if model_name == "Random Forest":
            # Random Forest doesn't need scaling
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_logger.debug(f"Random Forest created with 100 estimators")
            return model
        elif model_name == "Gradient Boosting":
            # Gradient Boosting doesn't need scaling
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model_logger.debug(f"Gradient Boosting created with 100 estimators")
            return model
        elif model_name == "Logistic Regression":
            # Logistic Regression benefits from scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42, 
                    max_iter=5000,  # Increased iterations
                    solver='lbfgs',  # Good for small datasets
                    C=1.0  # Default regularization
                ))
            ])
            model_logger.debug(f"Logistic Regression pipeline created with StandardScaler and max_iter=5000")
            return model
        elif model_name == "SVM":
            # SVM also benefits from scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    probability=True, 
                    random_state=42,
                    kernel='rbf',  # Good default kernel
                    C=1.0  # Default regularization
                ))
            ])
            model_logger.debug(f"SVM pipeline created with StandardScaler and RBF kernel")
            return model
        else:
            model_logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        model_logger.error(f"Error creating model {model_name}: {str(e)}")
        raise

# Load selected dataset
try:
    X_raw, y, feature_names, class_names = load_dataset(dataset_choice)
    X = pd.DataFrame(X_raw, columns=feature_names)
    app_logger.info(f"Dataset loaded successfully: {dataset_choice}")
    app_logger.debug(f"Dataset shape: {X.shape}, Classes: {list(class_names)}")
except Exception as e:
    app_logger.error(f"Failed to load dataset {dataset_choice}: {str(e)}")
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    app_logger.info(f"Data split completed - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
except Exception as e:
    app_logger.error(f"Failed to split data: {str(e)}")
    st.error(f"Error splitting data: {str(e)}")
    st.stop()

# Train model
try:
    model = get_model(model_choice)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    model_logger.info(f"Model {model_choice} trained successfully in {training_time:.2f} seconds")
    
    # Log model performance quickly
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    model_logger.info(f"Model performance - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
except Exception as e:
    model_logger.error(f"Failed to train model {model_choice}: {str(e)}")
    st.error(f"Error training model: {str(e)}")
    st.stop()

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Prediction", 
    "📊 Model Performance", 
    "🧠 Explanations", 
    "👤 User Study", 
    "📈 Explainability Metrics",
    "💾 Export Results"
])

# Helper function to ensure proper DataFrame format for predictions
def ensure_dataframe(data, feature_names):
    """Ensure data is in DataFrame format with proper feature names"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return pd.DataFrame(data, columns=feature_names)
    else:
        return pd.DataFrame(data, columns=feature_names)

with tab1:
    st.header("🎯 Model Prediction")
    app_logger.info("User accessed Prediction tab")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌿 Input Features")
        input_data = []
        for i, feature in enumerate(feature_names):
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            val = float(X[feature].mean())
            input_data.append(st.slider(
                label=f"{feature}", 
                min_value=min_val, 
                max_value=max_val, 
                value=val, 
                step=(max_val-min_val)/100,
                key=f"slider_{i}"            ))
        
        input_array = np.array(input_data).reshape(1, -1)
        input_df = ensure_dataframe(input_array, feature_names)
        app_logger.debug(f"User input features: {dict(zip(feature_names, input_data))}")
    
    with col2:
        st.subheader("🌟 Prediction Results")
        try:
            pred_class = model.predict(input_df)[0]
            pred_prob = model.predict_proba(input_df)[0]
            
            app_logger.info(f"Prediction made: {class_names[pred_class]} with confidence {pred_prob[pred_class]:.3f}")
            
            st.markdown(f"### Predicted Class: `{class_names[pred_class]}`")
            st.markdown(f"### Confidence: `{pred_prob[pred_class]:.3f}`")
            
            # Create probability chart
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': pred_prob
            })
            fig_prob = px.bar(prob_df, x='Class', y='Probability', 
                             title="Prediction Probabilities",
                             color='Probability',
                             color_continuous_scale='viridis')
            st.plotly_chart(fig_prob, use_container_width=True)
            
        except Exception as e:
            app_logger.error(f"Error making prediction: {str(e)}")
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")

with tab2:
    st.header("📊 Model Performance Analysis")
    app_logger.info("User accessed Model Performance tab")
    
    try:
        # Model performance metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        model_logger.info(f"Performance calculated - Accuracy: {accuracy:.3f}, CV Mean: {cv_scores.mean():.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("CV Mean", f"{cv_scores.mean():.3f}")
        with col3:
            st.metric("CV Std", f"{cv_scores.std():.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                           x=class_names, 
                           y=class_names,
                           aspect="auto",
                           title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("📋 Classification Report")
        st.dataframe(report_df.round(3))
        
        model_logger.debug(f"Classification report generated successfully")
        
    except Exception as e:
        model_logger.error(f"Error calculating model performance: {str(e)}")
        st.error(f"Error calculating performance metrics: {str(e)}")
        st.info("There was an issue analyzing model performance. Please try again.")

with tab3:
    st.header("🧠 Model Explanations")
    app_logger.info("User accessed Explanations tab")
    
    explanation_method = st.selectbox(
        "Choose Explanation Method",
        ["SHAP", "LIME", "Feature Importance", "All Methods"]
    )
    explanation_logger.info(f"User selected explanation method: {explanation_method}")
    
    if explanation_method in ["SHAP", "All Methods"]:
        st.subheader("🎯 SHAP Explanation")
        explanation_logger.info("Starting SHAP explanation generation")
        
        try:
            # SHAP explainer - handle different model types and pipelines
            if model_choice in ["Random Forest", "Gradient Boosting"]:
                # Tree-based models - try TreeExplainer first, fallback to others
                explanation_logger.debug(f"Using TreeExplainer for {model_choice}")
                try:
                    explainer_shap = shap.TreeExplainer(model)
                    shap_values = explainer_shap.shap_values(input_df)
                    expected_value = explainer_shap.expected_value
                    use_old_format = True
                    explanation_logger.debug("TreeExplainer succeeded")
                except Exception as tree_error:
                    explanation_logger.warning(f"TreeExplainer failed for {model_choice}: {str(tree_error)}")
                    if "only supported for binary classification" in str(tree_error):
                        explanation_logger.info(f"Multi-class classification detected for {model_choice}, using PermutationExplainer instead")
                        st.info("ℹ️ Using PermutationExplainer for multi-class classification (explains probabilities)")
                    else:
                        explanation_logger.info(f"TreeExplainer incompatible with current {model_choice} configuration, falling back to PermutationExplainer")
                        st.info("ℹ️ Using PermutationExplainer as fallback method")
                    
                    # Fallback to PermutationExplainer for multi-class Gradient Boosting
                    explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train.sample(50, random_state=42))
                    shap_values = explainer_shap(input_df)
                    expected_value = None  # PermutationExplainer doesn't have expected_value
                    use_old_format = False
                    explanation_logger.info("PermutationExplainer successfully configured for probability explanation")
            else:
                # Pipeline models (Logistic Regression, SVM) - need to handle differently
                explanation_logger.debug(f"Using general Explainer for {model_choice}")
                if hasattr(model, 'named_steps'):
                    # This is a pipeline - try different approaches
                    classifier = model.named_steps['classifier']
                    scaler = model.named_steps['scaler']
                    
                    try:
                        # Approach 1: Extract classifier and use transformed data
                        X_train_scaled = scaler.transform(X_train)
                        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
                        
                        input_scaled = scaler.transform(input_df)
                        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
                        
                        explainer_shap = shap.Explainer(classifier, X_train_scaled_df.sample(100))
                        shap_values = explainer_shap(input_scaled_df)
                        expected_value = explainer_shap.expected_value
                        explanation_logger.debug("Pipeline model SHAP values calculated with extracted classifier")
                        
                    except Exception as pipeline_error:
                        explanation_logger.warning(f"Pipeline extraction failed for {model_choice}: {str(pipeline_error)}")
                        if "not callable" in str(pipeline_error):
                            explanation_logger.info(f"Pipeline model {model_choice} requires PermutationExplainer for SHAP analysis")
                            st.info(f"ℹ️ Using PermutationExplainer for {model_choice} pipeline (explains probabilities)")
                        else:
                            explanation_logger.info(f"Pipeline approach incompatible, falling back to PermutationExplainer")
                            st.info("ℹ️ Using PermutationExplainer as fallback method")
                        
                        # Fallback to PermutationExplainer
                        explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train.sample(50, random_state=42))
                        shap_values = explainer_shap(input_df)
                        expected_value = None  # PermutationExplainer doesn't have expected_value
                        explanation_logger.info("PermutationExplainer successfully configured for pipeline probability explanation")
                else:
                    # Not a pipeline - use original approach
                    explainer_shap = shap.Explainer(model, X_train.sample(100))
                    shap_values = explainer_shap(input_df)
                    expected_value = explainer_shap.expected_value
                    explanation_logger.debug("Non-pipeline model SHAP values calculated")
                use_old_format = False
              # Get prediction for the current input
            pred_class_idx = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            explanation_logger.debug(f"Model prediction: class {pred_class_idx}, probability: {pred_proba[pred_class_idx]:.4f}")
            
            # Handle different SHAP value formats with more robust logic
            if use_old_format:
                # Old format from TreeExplainer
                if isinstance(shap_values, list):
                    # Multi-class case - shap_values is a list of arrays
                    single_shap_values = shap_values[pred_class_idx][0]
                    single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    # Multi-class case - shap_values is array with shape (n_samples, n_features, n_classes)
                    single_shap_values = shap_values[0, :, pred_class_idx]
                    single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                else:
                    # Binary classification case
                    single_shap_values = shap_values[0]
                    single_expected_value = expected_value
            else:                # New format from general Explainer
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        # Multi-class case: shape is (n_samples, n_features, n_classes)
                        single_shap_values = shap_values.values[0, :, pred_class_idx]
                        if hasattr(shap_values, 'base_values'):
                            if shap_values.base_values.ndim > 1:
                                single_expected_value = shap_values.base_values[0, pred_class_idx]
                            else:
                                single_expected_value = shap_values.base_values[0]
                        else:
                            single_expected_value = expected_value if expected_value is not None else 0.0
                    else:
                        # Binary or regression case: shape is (n_samples, n_features)
                        single_shap_values = shap_values.values[0]
                        single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else (expected_value if expected_value is not None else 0.0)
                else:
                    # Fallback case
                    single_shap_values = shap_values[0]
                    single_expected_value = expected_value if expected_value is not None else 0.0
              # SHAP waterfall plot
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            
            # Create a clean explanation object for waterfall plot with proper data types
            # Ensure all values are properly converted to float to avoid format issues
            safe_shap_values = np.array(single_shap_values, dtype=float)
            safe_expected_value = float(single_expected_value) if single_expected_value is not None else 0.0
            safe_input_data = np.array(input_df.values[0], dtype=float)
            
            # Log SHAP value validation
            shap_sum = np.sum(safe_shap_values) + safe_expected_value
            explanation_logger.debug(f"SHAP validation - Values sum: {np.sum(safe_shap_values):.4f}, Base: {safe_expected_value:.4f}, Total: {shap_sum:.4f}")
            explanation_logger.debug(f"Model probability for class {pred_class_idx}: {pred_proba[pred_class_idx]:.4f}")
            
            waterfall_explanation = shap.Explanation(
                values=safe_shap_values,
                base_values=safe_expected_value,
                data=safe_input_data,
                feature_names=list(feature_names)  # Ensure it's a list
            )
              # Generate the waterfall plot
            shap.plots.waterfall(waterfall_explanation, show=False)
            st.pyplot(fig_shap)
            
            explanation_logger.info(f"SHAP waterfall plot generated successfully for {model_choice}")
            explanation_logger.debug(f"Waterfall plot shows {len(safe_shap_values)} feature contributions")
            
            # Show feature values and contributions
            st.subheader("📊 Feature Contributions")
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': input_df.values[0],
                'SHAP Value': single_shap_values,
                'Contribution': ['Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral' for x in single_shap_values]
            })
            contrib_df['Abs SHAP Value'] = np.abs(contrib_df['SHAP Value'])
            contrib_df = contrib_df.sort_values('Abs SHAP Value', ascending=False)
            st.dataframe(contrib_df.round(4))
            
            explanation_logger.info(f"Feature contributions table generated with {len(feature_names)} features")
            
        except Exception as e:
            explanation_logger.error(f"SHAP explanation failed for {model_choice}: {str(e)}")
            
            # Provide specific error guidance
            if "unsupported format string" in str(e):
                st.error("❌ SHAP waterfall plot format error")
                st.info("This is usually due to data type issues. Our system attempts to handle this automatically.")
                explanation_logger.error("Waterfall plot format error - data type conversion failed")
            elif "not callable" in str(e):
                st.error("❌ SHAP explainer compatibility error")
                st.info(f"The {model_choice} model structure is not directly compatible with standard SHAP explainers.")
                explanation_logger.error(f"Model {model_choice} is not callable for SHAP analysis")
            elif "only supported for binary classification" in str(e):
                st.error("❌ SHAP TreeExplainer limitation")
                st.info("TreeExplainer for Gradient Boosting only supports binary classification. Try with a binary dataset.")
                explanation_logger.error("TreeExplainer multi-class limitation encountered")
            else:
                st.error(f"❌ SHAP explanation error: {str(e)}")
                explanation_logger.error(f"Unexpected SHAP error: {str(e)}")
            
            st.info("💡 **Suggestion**: Try using LIME explanations instead, which work with all model types.")
            
            # Debug information
            with st.expander("Debug Information"):
                st.write("Error details:", str(e))
                try:
                    st.write("SHAP values type:", type(shap_values))
                    if hasattr(shap_values, 'values'):
                        st.write("SHAP values shape:", shap_values.values.shape)
                    elif isinstance(shap_values, list):
                        st.write("SHAP values list length:", len(shap_values))
                        st.write("First element shape:", np.array(shap_values[0]).shape)
                    else:
                        st.write("SHAP values shape:", np.array(shap_values).shape)
                except:                    st.write("Could not determine SHAP values structure")
        
        # SHAP summary plot for feature importance
        if st.checkbox("Show Feature Importance (SHAP)"):
            explanation_logger.info("User requested SHAP feature importance plot")
            try:
                with st.spinner("Generating SHAP summary plot..."):
                    sample_size = min(100, len(X_test))
                    X_sample = X_test.iloc[:sample_size]
                    explanation_logger.debug(f"Using {sample_size} samples for SHAP summary plot")
                    
                    if model_choice in ["Random Forest", "Gradient Boosting"]:
                        shap_values_all = explainer_shap.shap_values(X_sample)
                        if isinstance(shap_values_all, list):
                            # Multi-class case - show all classes
                            fig_summary, axes = plt.subplots(1, len(class_names), figsize=(15, 6))
                            if len(class_names) == 1:
                                axes = [axes]
                            for i, class_name in enumerate(class_names):
                                plt.sca(axes[i])
                                shap.summary_plot(shap_values_all[i], X_sample, 
                                                feature_names=feature_names, 
                                                show=False, title=f"Class: {class_name}")
                            explanation_logger.debug(f"Multi-class SHAP summary plot generated for {len(class_names)} classes")
                        else:
                            fig_summary, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values_all, X_sample, 
                                            feature_names=feature_names, show=False)
                            explanation_logger.debug("Binary SHAP summary plot generated")
                    else:
                        shap_values_all = explainer_shap(X_sample)
                        fig_summary, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.beeswarm(shap_values_all, show=False)
                        explanation_logger.debug("SHAP beeswarm plot generated for pipeline model")                    
                    st.pyplot(fig_summary)
                    explanation_logger.info("SHAP summary plot displayed successfully")
            except Exception as e:
                explanation_logger.error(f"Error generating SHAP summary plot: {str(e)}")
                st.error(f"Error generating SHAP summary plot: {str(e)}")
                st.info("Try reducing the sample size or using a different model.")
    
    if explanation_method in ["LIME", "All Methods"]:
        st.subheader("🧪 LIME Explanation")
        explanation_logger.info("Starting LIME explanation generation")
        
        try:
            # Create a wrapper function to ensure proper DataFrame format for LIME
            def model_predict_proba_wrapper(X):
                """Wrapper to ensure X is DataFrame with proper feature names"""
                X_df = ensure_dataframe(X, feature_names)
                return model.predict_proba(X_df)
            
            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, 
                feature_names=feature_names,
                class_names=class_names, 
                discretize_continuous=True
            )
            exp = explainer_lime.explain_instance(
                input_array[0], 
                model_predict_proba_wrapper, 
                num_features=len(feature_names)
            )
            
            explanation_logger.debug(f"LIME explanation generated for {len(feature_names)} features")            
            fig_lime = exp.as_pyplot_figure()
            st.pyplot(fig_lime)
            
            explanation_logger.info("LIME explanation displayed successfully")
            
        except Exception as e:
            explanation_logger.error(f"Error generating LIME explanation: {str(e)}")
            st.error(f"Error generating LIME explanation: {str(e)}")
            st.info("LIME explanations may not be available for this model/dataset combination.")
    
    if explanation_method in ["Feature Importance", "All Methods"]:
        st.subheader("📊 Feature Importance")
        explanation_logger.info("Generating feature importance explanation")
        
        if hasattr(model, 'feature_importances_'):
            try:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                explanation_logger.debug(f"Feature importance calculated for {len(feature_names)} features")
                
                fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title="Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True)
                
                explanation_logger.info("Feature importance plot displayed successfully")
                
            except Exception as e:
                explanation_logger.error(f"Error generating feature importance plot: {str(e)}")
                st.error(f"Error generating feature importance plot: {str(e)}")
        else:
            explanation_logger.warning(f"Feature importance not available for model type: {model_choice}")
            st.info("Feature importance not available for this model type")

with tab4:
    st.header("👤 User-Centric Explainability Study")
    app_logger.info("User accessed User Study tab")
    user_study_logger.info("User study session started")
    
    st.markdown("""
    This section evaluates the quality of explanations from a user perspective.
    Please interact with the explanations above and provide your feedback.
    """)
    
    # User comprehension test
    st.subheader("🧩 Comprehension Test")
    
    # Generate a test question based on current prediction
    current_explanation = f"The model predicted '{class_names[pred_class]}' with {pred_prob[pred_class]:.2f} confidence."
    user_study_logger.debug(f"Current prediction for user study: {current_explanation}")
    
    with st.form("comprehension_test"):
        st.write("Based on the explanations provided, please answer:")
        
        q1 = st.radio(
            "Which feature contributed MOST to the current prediction?",
            options=feature_names,
            key="q1"
        )
        
        q2 = st.slider(
            "How confident are you in your understanding of the model's decision? (1=Not confident, 10=Very confident)",
            min_value=1, max_value=10, value=5, key="q2"
        )
        
        q3 = st.text_area(
            "In your own words, explain why the model made this prediction:",
            key="q3"
        )
        
        q4 = st.selectbox(
            "Which explanation method did you find most helpful?",
            ["SHAP", "LIME", "Feature Importance", "None were clear"],
            key="q4"
        )
        
        q5 = st.slider(            "Rate the overall usefulness of the explanations (1=Not useful, 10=Very useful):",
            min_value=1, max_value=10, value=5, key="q5"
        )
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            user_study_logger.info("User submitted feedback form")
            feedback = {
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_choice,
                'model': model_choice,
                'prediction': class_names[pred_class],
                'confidence': float(pred_prob[pred_class]),
                'feature_choice': q1,
                'user_confidence': q2,
                'explanation': q3,
                'preferred_method': q4,
                'usefulness_rating': q5,
                'input_features': dict(zip(feature_names, input_data))
            }
            st.session_state.feedback_data.append(feedback)
            user_study_logger.info(f"Feedback recorded - User confidence: {q2}/10, Usefulness: {q5}/10, Preferred method: {q4}")
            user_study_logger.debug(f"Full feedback data: {feedback}")
            st.success("Thank you for your feedback!")
    
    # Task-based evaluation
    st.subheader("🎯 Task-Based Evaluation")
    
    with st.expander("Counterfactual Analysis Task"):
        st.write("**Task**: What would need to change to get a different prediction?")
        
        target_class = st.selectbox(
            "What class would you like the model to predict instead?",
            [c for c in class_names if c != class_names[pred_class]]
        )        
        user_changes = st.text_area(
            f"Describe what changes you would make to get '{target_class}' prediction:",
            placeholder="e.g., Increase feature X by 20%, decrease feature Y..."
        )
        
        if st.button("Record Counterfactual Response"):
            user_study_logger.info("User submitted counterfactual response")
            cf_data = {
                'timestamp': datetime.now().isoformat(),
                'original_prediction': class_names[pred_class],
                'target_prediction': target_class,
                'user_strategy': user_changes,
                'model': model_choice
            }
            st.session_state.user_study_data.append(cf_data)
            user_study_logger.info(f"Counterfactual response recorded - Original: {class_names[pred_class]}, Target: {target_class}")
            user_study_logger.debug(f"User strategy: {user_changes}")
            st.success("Counterfactual response recorded!")

with tab5:
    st.header("📈 Explainability Quality Metrics")
    app_logger.info("User accessed Explainability Metrics tab")
    
    if len(st.session_state.feedback_data) > 0:
        app_logger.info(f"Displaying metrics for {len(st.session_state.feedback_data)} feedback entries")
        feedback_df = pd.DataFrame(st.session_state.feedback_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 User Feedback Summary")
            
            # Average ratings
            avg_confidence = feedback_df['user_confidence'].mean()
            avg_usefulness = feedback_df['usefulness_rating'].mean()
            
            st.metric("Avg User Confidence", f"{avg_confidence:.2f}/10")
            st.metric("Avg Usefulness Rating", f"{avg_usefulness:.2f}/10")
            
            # Method preferences
            method_counts = feedback_df['preferred_method'].value_counts()
            fig_methods = px.pie(values=method_counts.values, 
                               names=method_counts.index,
                               title="Preferred Explanation Methods")
            st.plotly_chart(fig_methods, use_container_width=True)
        
        with col2:
            st.subheader("📈 Trends Over Time")
            
            if len(feedback_df) > 1:
                fig_trends = px.line(
                    feedback_df.reset_index(), 
                    x='index', 
                    y=['user_confidence', 'usefulness_rating'],
                    title="User Ratings Over Time"
                )
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Correlation analysis
            if len(feedback_df) > 5:
                corr_matrix = feedback_df[['user_confidence', 'usefulness_rating', 'confidence']].corr()
                fig_corr = px.imshow(corr_matrix, 
                                   title="Correlation Matrix",
                                   aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)
    
    else:
        st.info("No user feedback collected yet. Please complete the user study in the previous tab.")
    
    # Computational metrics
    st.subheader("💻 Computational Explainability Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:        # Explanation consistency
        if st.button("Calculate Consistency"):
            app_logger.info("User requested explanation consistency calculation")
            explanation_logger.info("Starting consistency calculation")
            try:
                with st.spinner("Calculating explanation consistency..."):                    # Generate explanations for similar inputs
                    consistency_scores = []
                    base_input = input_array[0]
                    explanation_logger.debug(f"Base input for consistency: {base_input}")
                    
                    for i in range(5):
                        # Add small noise to input
                        noisy_input = base_input + np.random.normal(0, 0.05, base_input.shape)  # Smaller noise
                        noisy_df = ensure_dataframe(noisy_input, feature_names)
                        
                        # Get SHAP values
                        try:
                            if model_choice in ["Random Forest", "Gradient Boosting"]:
                                explainer = shap.TreeExplainer(model)
                                shap_vals = explainer.shap_values(noisy_df)
                                if isinstance(shap_vals, list):
                                    # Multi-class case - use predicted class
                                    pred_class_idx = model.predict(noisy_df)[0]
                                    shap_vals = shap_vals[pred_class_idx]
                                consistency_scores.append(shap_vals[0])
                            else:                                # Handle pipeline models
                                if hasattr(model, 'named_steps'):
                                    # Pipeline model - use scaled data
                                    classifier = model.named_steps['classifier']
                                    scaler = model.named_steps['scaler']
                                    noisy_scaled = scaler.transform(noisy_df)
                                    noisy_scaled_df = pd.DataFrame(noisy_scaled, columns=feature_names)
                                    X_train_scaled = scaler.transform(X_train.sample(50))
                                    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
                                    explainer = shap.Explainer(classifier, X_train_scaled_df)
                                    shap_vals = explainer(noisy_scaled_df)
                                else:
                                    # Regular model
                                    explainer = shap.Explainer(model, X_train.sample(50))
                                    shap_vals = explainer(noisy_df)
                                
                                if hasattr(shap_vals, 'values'):
                                    if len(shap_vals.values.shape) == 3:
                                        pred_class_idx = model.predict(noisy_df)[0]
                                        consistency_scores.append(shap_vals.values[0, :, pred_class_idx])
                                    else:
                                        consistency_scores.append(shap_vals.values[0])
                        except Exception as e:
                            explanation_logger.warning(f"Skipping consistency calculation for sample {i+1}: {str(e)}")
                            st.warning(f"Skipping consistency calculation for sample {i+1}: {str(e)}")
                            continue
                    
                    if len(consistency_scores) >= 2:
                        explanation_logger.info(f"Calculating consistency metrics from {len(consistency_scores)} samples")
                        try:
                            # Flatten and normalize SHAP values for consistency calculation
                            flattened_scores = []
                            for score in consistency_scores:
                                # Convert to numpy array and flatten
                                score_array = np.array(score)
                                flattened_score = score_array.flatten()
                                flattened_scores.append(flattened_score)
                            
                            # Ensure all scores have the same length
                            min_length = min(len(score) for score in flattened_scores)
                            normalized_scores = [score[:min_length] for score in flattened_scores]
                            
                            # Convert to 2D array for correlation calculation
                            scores_matrix = np.array(normalized_scores)
                            
                            # Calculate consistency using multiple methods
                            consistency_metrics = {}
                            
                            # Method 1: Pearson correlation
                            if scores_matrix.shape[0] >= 2 and scores_matrix.shape[1] >= 2:
                                try:
                                    correlation_matrix = np.corrcoef(scores_matrix)
                                    if correlation_matrix.size > 1:
                                        # Get upper triangular part (excluding diagonal)
                                        upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
                                        if len(upper_tri_indices[0]) > 0:
                                            avg_correlation = np.mean(correlation_matrix[upper_tri_indices])
                                            consistency_metrics['Correlation'] = avg_correlation
                                except Exception as e:
                                    st.warning(f"Correlation calculation failed: {str(e)}")
                            
                            # Method 2: Cosine similarity
                            try:
                                from sklearn.metrics.pairwise import cosine_similarity
                                cosine_sim_matrix = cosine_similarity(scores_matrix)
                                if cosine_sim_matrix.size > 1:
                                    upper_tri_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
                                    if len(upper_tri_indices[0]) > 0:
                                        avg_cosine_sim = np.mean(cosine_sim_matrix[upper_tri_indices])
                                        consistency_metrics['Cosine Similarity'] = avg_cosine_sim
                            except Exception as e:
                                st.warning(f"Cosine similarity calculation failed: {str(e)}")
                            
                            # Method 3: Euclidean distance-based consistency
                            try:
                                from sklearn.metrics.pairwise import euclidean_distances
                                distances = euclidean_distances(scores_matrix)
                                if distances.size > 1:
                                    upper_tri_indices = np.triu_indices_from(distances, k=1)
                                    if len(upper_tri_indices[0]) > 0:
                                        avg_distance = np.mean(distances[upper_tri_indices])
                                        # Convert distance to similarity (higher = more consistent)
                                        max_distance = np.max(distances[upper_tri_indices])
                                        if max_distance > 0:
                                            distance_consistency = 1 - (avg_distance / max_distance)
                                            consistency_metrics['Distance-based'] = distance_consistency
                            except Exception as e:
                                st.warning(f"Distance calculation failed: {str(e)}")
                            
                            # Method 4: Standard deviation-based consistency
                            try:
                                # Calculate coefficient of variation for each feature
                                feature_std = np.std(scores_matrix, axis=0)
                                feature_mean = np.mean(np.abs(scores_matrix), axis=0)
                                # Avoid division by zero
                                cv_scores = np.divide(feature_std, feature_mean, 
                                                    out=np.zeros_like(feature_std), 
                                                    where=feature_mean!=0)
                                avg_cv = np.mean(cv_scores)
                                # Convert to consistency (lower CV = higher consistency)
                                cv_consistency = 1 / (1 + avg_cv) if avg_cv > 0 else 1.0
                                consistency_metrics['Stability'] = cv_consistency
                            except Exception as e:
                                st.warning(f"Stability calculation failed: {str(e)}")
                            
                            # Display results
                            if consistency_metrics:
                                st.subheader("🔍 Consistency Metrics")
                                
                                # Create columns for metrics
                                metric_cols = st.columns(len(consistency_metrics))
                                for i, (metric_name, value) in enumerate(consistency_metrics.items()):
                                    with metric_cols[i]:
                                        st.metric(f"{metric_name}", f"{value:.3f}")
                                
                                # Show interpretation
                                with st.expander("📊 Interpretation Guide"):
                                    st.markdown("""
                                    **Consistency Metrics Interpretation:**
                                    - **Correlation**: Measures linear relationship between explanations (-1 to 1, higher is better)
                                    - **Cosine Similarity**: Measures directional similarity (0 to 1, higher is better)
                                    - **Distance-based**: Measures spatial consistency (0 to 1, higher is better)
                                    - **Stability**: Measures variation consistency (0 to 1, higher is better)
                                    
                                    **Good Consistency**: Values > 0.7
                                    **Moderate Consistency**: Values 0.4-0.7
                                    **Low Consistency**: Values < 0.4
                                    """)
                                  # Calculate overall consistency score
                                overall_consistency = np.mean(list(consistency_metrics.values()))
                                st.metric("Overall Consistency", f"{overall_consistency:.3f}")
                                
                                explanation_logger.info(f"Consistency calculation completed - Overall score: {overall_consistency:.3f}")
                                explanation_logger.debug(f"Individual metrics: {consistency_metrics}")
                                
                                # Show diagnostic information
                                with st.expander("🔧 Diagnostic Information"):
                                    st.write(f"**Samples analyzed**: {len(consistency_scores)}")
                                    st.write(f"**Feature dimensions**: {scores_matrix.shape}")
                                    st.write(f"**Score ranges**: {np.min(scores_matrix):.3f} to {np.max(scores_matrix):.3f}")
                                    
                                    # Show sample correlations
                                    if 'Correlation' in consistency_metrics:
                                        st.write("**Sample Pairwise Correlations**:")
                                        if 'correlation_matrix' in locals():
                                            correlation_df = pd.DataFrame(correlation_matrix, 
                                                                        columns=[f"Sample {i+1}" for i in range(len(consistency_scores))],
                                                                        index=[f"Sample {i+1}" for i in range(len(consistency_scores))])
                                            st.dataframe(correlation_df.round(3))
                            else:
                                st.error("Could not calculate any consistency metrics")
                                
                        except Exception as e:
                            st.error(f"Error in consistency calculation: {str(e)}")
                              # Show debug information
                            with st.expander("🐛 Debug Information"):
                                st.write("**Error details**:", str(e))
                                st.write("**Number of samples**:", len(consistency_scores))
                                try:
                                    st.write("**Sample shapes**:", [np.array(score).shape for score in consistency_scores])
                                    st.write("**Sample types**:", [type(score) for score in consistency_scores])
                                except:
                                    st.write("Could not determine sample characteristics")
                    else:
                        explanation_logger.warning(f"Insufficient consistency samples: {len(consistency_scores)} (need at least 2)")
                        st.error("Could not calculate consistency - insufficient valid samples")
                        st.info(f"Only {len(consistency_scores)} valid samples collected (need at least 2)")
            except Exception as e:
                explanation_logger.error(f"Error in overall consistency calculation: {str(e)}")
                st.error(f"Error calculating consistency: {str(e)}")
    
    with col2:
        # Explanation complexity
        st.metric("Model Complexity", f"{len(feature_names)} features")
        if hasattr(model, 'n_estimators'):
            st.metric("Model Trees/Estimators", model.n_estimators)
    
    with col3:
        # Prediction confidence vs explanation clarity
        explanation_clarity = pred_prob[pred_class] * len([f for f in feature_names])  # Simple heuristic
        st.metric("Explanation Clarity Score", f"{explanation_clarity:.3f}")

with tab6:
    st.header("💾 Export Results & Data")
    app_logger.info("User accessed Export Results tab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Current Session Data")
        
        session_data = {
            'model_info': {
                'dataset': dataset_choice,
                'model_type': model_choice,
                'accuracy': float(accuracy),
                'features': list(feature_names),
                'classes': list(class_names)
            },
            'current_prediction': {
                'input': dict(zip(feature_names, input_data)),
                'prediction': class_names[pred_class],
                'probabilities': pred_prob.tolist(),
                'timestamp': datetime.now().isoformat()
            },
            'user_feedback': st.session_state.feedback_data,
            'user_study': st.session_state.user_study_data
        }
        
        if st.button("📥 Download Session Data (JSON)"):
            app_logger.info("User requested session data download")
            st.download_button(
                label="Download JSON",
                data=json.dumps(session_data, indent=2),
                file_name=f"ml_evaluation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"            )
            app_logger.info("Session data download prepared")
    
    with col2:
        st.subheader("📊 Generate Report")
        
        if st.button("📋 Generate Summary Report"):
            app_logger.info("User requested summary report generation")
            report = f"""
            # ML Model Explainability Evaluation Report
            
            **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ## Model Configuration
            - **Dataset**: {dataset_choice}
            - **Model**: {model_choice}
            - **Test Accuracy**: {accuracy:.3f}
            - **Cross-validation Mean**: {cv_scores.mean():.3f}
            
            ## Current Prediction
            - **Input**: {dict(zip(feature_names, input_data))}
            - **Prediction**: {class_names[pred_class]}
            - **Confidence**: {pred_prob[pred_class]:.3f}
            
            ## User Feedback Summary
            - **Total Feedback Entries**: {len(st.session_state.feedback_data)}
            """
            
            if len(st.session_state.feedback_data) > 0:
                feedback_df = pd.DataFrame(st.session_state.feedback_data)
                report += f"""
            - **Average User Confidence**: {feedback_df['user_confidence'].mean():.2f}/10
            - **Average Usefulness Rating**: {feedback_df['usefulness_rating'].mean():.2f}/10
            - **Most Preferred Method**: {feedback_df['preferred_method'].mode().iloc[0] if len(feedback_df) > 0 else 'N/A'}
                """
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            app_logger.info("Summary report generated and download prepared")
      # Clear session data
    if st.button("🗑️ Clear All Session Data", type="secondary"):
        app_logger.info("User requested to clear all session data")
        feedback_count = len(st.session_state.feedback_data)
        study_count = len(st.session_state.user_study_data)
        
        st.session_state.feedback_data = []
        st.session_state.user_study_data = []
        
        app_logger.info(f"Session data cleared - {feedback_count} feedback entries and {study_count} user study entries removed")
        st.success("Session data cleared!")

# Footer
st.markdown("---")
st.markdown("*🧠 Developed by Leptons*")
