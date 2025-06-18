"""
Streamlit UI components for ML Evaluation App
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from utils.config import APP_TITLE, DATASETS, MODELS
from utils.logging_config import get_logger
from utils.data_preparation import (
    load_uploaded_file, 
    analyze_dataset, 
    paginate_dataframe,
    detect_target_column,
    prepare_dataset_for_ml,
    get_data_quality_report
)
# Enhanced data preparation imports
try:
    from utils.data_preparation_enhanced import data_prep_tools
    ENHANCED_PREP_AVAILABLE = True
except ImportError:
    ENHANCED_PREP_AVAILABLE = False

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


def render_data_upload_page():
    """Render the enhanced data upload and preparation page"""
    st.title("📁 Dataset Upload & Preparation")
    st.markdown("*Upload your own dataset and prepare it for machine learning analysis*")
    
    # Enhanced imports
    try:
        from components.enhanced_ui_components import (
            render_editable_column_types, 
            render_enhanced_quality_report,
            render_operation_tracker,
            render_advanced_feature_selection
        )
        enhanced_features_available = True
    except ImportError:
        enhanced_features_available = False
        st.warning("⚠️ Enhanced features not available. Some advanced options may be limited.")
    
    # Progress indicator
    progress_col1, progress_col2, progress_col3, progress_col4, progress_col5 = st.columns(5)
    
    with progress_col1:
        upload_status = "✅" if 'uploaded_df' in st.session_state else "⏳"
        st.markdown(f"**{upload_status} 1. Upload**")
    
    with progress_col2:
        analyze_status = "✅" if 'dataset_analysis' in st.session_state else "⏳"
        st.markdown(f"**{analyze_status} 2. Analyze**")
    
    with progress_col3:
        quality_status = "✅" if 'enhanced_quality_done' in st.session_state else "⏳"
        st.markdown(f"**{quality_status} 3. Quality & Types**")
    
    with progress_col4:
        feature_status = "✅" if 'feature_selection_done' in st.session_state else "⏳"
        st.markdown(f"**{feature_status} 4. Features**")
    
    with progress_col5:
        prep_status = "✅" if 'prepared_dataset' in st.session_state else "⏳"
        st.markdown(f"**{prep_status} 5. Prepare**")
    
    st.markdown("---")
    
    # File upload section
    st.header("1. 📤 Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing your dataset. Supported formats: .csv, .xlsx, .xls"
        )
    
    with col2:
        st.info("""
        **Supported Formats:**
        • CSV files (.csv)
        • Excel files (.xlsx, .xls)
        
        **Enhanced Features:**        • Editable column types
        • Advanced quality analysis
        • Smart feature selection
        • Operation tracking
        """)
    
    # Sample datasets info
    if st.expander("📋 Sample Datasets Available", expanded=False):
        st.markdown("""
        **🧪 Test with sample datasets:**
        1. `problematic_dataset.csv` - Multiple issues for testing auto-preparation
        2. `missing_values_dataset.csv` - Heavy missing data scenarios  
        3. `single_class_dataset.csv` - Stratification error examples
        
        **Requirements:**
        • First row should contain column headers
        • Data should be clean and structured
        """)
    
    if uploaded_file is not None:
        # Load and analyze the dataset
        with st.spinner("🔄 Loading and analyzing dataset..."):
            df, error_msg = load_uploaded_file(uploaded_file)
        
        if error_msg:
            show_error_message(error_msg)
            return None
        
        if df is not None:
            show_success_message(f"✅ Successfully loaded **{uploaded_file.name}**")            # Store in session state and analyze
            st.session_state.uploaded_df = df
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Analyze dataset
            with st.spinner("🔍 Analyzing dataset structure..."):
                analysis = analyze_dataset(df)
                st.session_state.dataset_analysis = analysis
            
            # Store original dataset for tracking
            if 'original_df' not in st.session_state:
                st.session_state.original_df = df.copy()
                st.session_state.modified_df = df.copy()
            
            # Create enhanced tabs for better organization
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", 
                "🔧 Column Types", 
                "📋 Quality Report", 
                "📄 Data Preview",
                "🎯 Feature Selection", 
                "🚀 ML Preparation"
            ])
            
            with tab1:
                render_dataset_overview(df, analysis)
            
            with tab2:
                if enhanced_features_available:
                    # Use current modified dataframe
                    current_df = st.session_state.get('modified_df', df)
                    current_analysis = analyze_dataset(current_df)
                    df_modified = render_editable_column_types(current_df, current_analysis)
            
            with tab3:
                if enhanced_features_available:
                    current_df = st.session_state.get('modified_df', df)
                    current_analysis = analyze_dataset(current_df)
                    render_enhanced_quality_report(current_df, current_analysis)
                    render_operation_tracker()
                else:
                    render_data_quality_report(df, analysis)
            
            with tab4:
                current_df = st.session_state.get('modified_df', df)
                render_paginated_data(current_df)
            
            with tab5:
                if enhanced_features_available:
                    current_df = st.session_state.get('modified_df', df)
                    
                    # Target column selection first
                    suggested_targets = detect_target_column(current_df, analyze_dataset(current_df))
                    
                    target_column = st.selectbox(
                        "🎯 Select Target Column",
                        options=current_df.columns.tolist(),
                        index=current_df.columns.tolist().index(suggested_targets[0]) if suggested_targets else 0,
                        help="Choose the column you want to predict"
                    )
                    
                    if suggested_targets:
                        st.info(f"💡 Suggested target columns: {', '.join(suggested_targets)}")
                    
                    if target_column:
                        render_advanced_feature_selection(current_df, target_column)
                        st.session_state.target_column = target_column
                        st.session_state.feature_selection_done = True
                else:
                    st.info("⚠️ Enhanced feature selection requires advanced components.")
            
            with tab6:
                current_df = st.session_state.get('modified_df', df)
                current_analysis = analyze_dataset(current_df)
                render_dataset_preparation(current_df, current_analysis)
            
            return df
    
    else:
        # Show helpful tips when no file is uploaded
        st.markdown("### 💡 Getting Started")
        
        tip_col1, tip_col2 = st.columns(2)
        
        with tip_col1:
            st.info("""
            **📝 Preparing Your Data:**
            1. Ensure your data is in CSV or Excel format
            2. Include column headers in the first row
            3. Make sure there's a clear target/label column
            4. Remove or handle any completely empty rows/columns
            """)
        
        with tip_col2:
            st.success("""
            **🎯 What You'll Get:**
            • Automatic data analysis and statistics
            • Interactive data preview with pagination
            • Data quality assessment and recommendations
            • ML-ready dataset preparation
            • Seamless integration with model analysis
            """)
        
        # Sample data option
        st.markdown("### 📚 Don't Have Data? Try Our Samples!")
        if st.button("🎲 Generate Sample Datasets", type="secondary"):
            st.info("� Run `python scripts/create_sample_data.py` in your terminal to create sample datasets for testing!")
        
        return None


def render_dataset_overview(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Render enhanced dataset overview section"""
    # Basic metrics with better visual hierarchy
    st.markdown("### � Dataset Statistics")
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("📋 Total Rows", f"{analysis['shape'][0]:,}")
    with metric_col2:
        st.metric("📊 Columns", analysis['shape'][1])
    with metric_col3:
        missing_count = sum(analysis['missing_values'].values())
        st.metric("❌ Missing Values", f"{missing_count:,}", 
                 delta=f"{(missing_count/(analysis['shape'][0]*analysis['shape'][1])*100):.1f}%")
    with metric_col4:
        st.metric("🔄 Duplicates", f"{analysis['duplicates']:,}")
    with metric_col5:
        memory_mb = analysis['memory_usage'] / (1024 * 1024)
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
    
    # Column type breakdown
    st.markdown("### 📋 Column Types")
    type_col1, type_col2, type_col3 = st.columns(3)
    
    with type_col1:
        st.info(f"🔢 **Numeric Columns**: {len(analysis['numeric_columns'])}")
        if analysis['numeric_columns']:
            st.caption(", ".join(analysis['numeric_columns'][:3]) + 
                      (f" (+{len(analysis['numeric_columns'])-3} more)" if len(analysis['numeric_columns']) > 3 else ""))
    
    with type_col2:
        st.info(f"📝 **Categorical Columns**: {len(analysis['categorical_columns'])}")
        if analysis['categorical_columns']:
            st.caption(", ".join(analysis['categorical_columns'][:3]) + 
                      (f" (+{len(analysis['categorical_columns'])-3} more)" if len(analysis['categorical_columns']) > 3 else ""))
    
    with type_col3:
        st.info(f"📅 **DateTime Columns**: {len(analysis['datetime_columns'])}")
        if analysis['datetime_columns']:
            st.caption(", ".join(analysis['datetime_columns'][:3]))
        else:
            st.caption("None detected")
    
    # Detailed column information in expandable section
    with st.expander("� Detailed Column Information", expanded=False):
        col_info = []
        for col in analysis['columns']:
            missing_pct = (analysis['missing_values'][col] / analysis['shape'][0]) * 100
            col_type = str(analysis['dtypes'][col])
            
            # Determine column category
            if col in analysis['numeric_columns']:
                category = "🔢 Numeric"
            elif col in analysis['categorical_columns']:
                category = "📝 Categorical"
            elif col in analysis['datetime_columns']:
                category = "📅 DateTime"
            else:
                category = "❓ Other"
            
            col_info.append({
                'Column Name': col,
                'Category': category,
                'Data Type': col_type,
                'Missing Count': analysis['missing_values'][col],
                'Missing %': f"{missing_pct:.1f}%",
                'Unique Values': df[col].nunique() if col in df.columns else 'N/A'
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, height=300)


def render_paginated_data(df: pd.DataFrame, page_size: int = 50):
    """Render paginated data display with controls below table"""
    st.header("3. 🔍 Data Preview")
    
    # Top controls - page size selector and info
    top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
    
    with top_col1:
        page_size = st.selectbox(
            "📄 Rows per page",
            options=[25, 50, 100, 200],
            index=1,
            key="page_size_selector"
        )
    
    # Initialize page number in session state
    if 'data_page_number' not in st.session_state:
        st.session_state.data_page_number = 1
    
    # Get paginated data
    paginated_df, pagination_info = paginate_dataframe(
        df, page_size, st.session_state.data_page_number
    )
    
    with top_col2:
        st.info(f"📊 Showing rows **{pagination_info['start_row']:,} - {pagination_info['end_row']:,}** of **{pagination_info['total_rows']:,}** total records")
    
    with top_col3:
        st.metric("📑 Total Pages", f"{pagination_info['total_pages']:,}")
    
    # Display paginated data
    st.dataframe(paginated_df, use_container_width=True, height=400)
    
    # Navigation controls below the table
    st.markdown("---")
    
    # Navigation buttons in a more compact layout
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6, nav_col7 = st.columns([1, 1, 1, 1.5, 1, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", disabled=pagination_info['current_page'] == 1, key="first_btn", use_container_width=True):
            st.session_state.data_page_number = 1
            st.rerun()
    
    with nav_col2:
        if st.button("⬅️ Prev", disabled=pagination_info['current_page'] == 1, key="prev_btn", use_container_width=True):
            st.session_state.data_page_number = max(1, st.session_state.data_page_number - 1)
            st.rerun()
    
    with nav_col3:
        st.write("")  # Empty column for spacing
    
    with nav_col4:
        # Page number input with better styling
        new_page = st.number_input(
            "🔢 Go to page",
            min_value=1,
            max_value=pagination_info['total_pages'],
            value=pagination_info['current_page'],
            key="page_input",
            help=f"Enter page number (1-{pagination_info['total_pages']})"
        )
        if new_page != pagination_info['current_page']:
            st.session_state.data_page_number = new_page
            st.rerun()
    
    with nav_col5:
        st.write("")  # Empty column for spacing
    
    with nav_col6:
        if st.button("Next ➡️", disabled=pagination_info['current_page'] == pagination_info['total_pages'], key="next_btn", use_container_width=True):
            st.session_state.data_page_number = min(pagination_info['total_pages'], st.session_state.data_page_number + 1)
            st.rerun()
    
    with nav_col7:
        if st.button("Last ⏭️", disabled=pagination_info['current_page'] == pagination_info['total_pages'], key="last_btn", use_container_width=True):
            st.session_state.data_page_number = pagination_info['total_pages']
            st.rerun()
    
    # Page info below navigation
    st.caption(f"📍 Page **{pagination_info['current_page']}** of **{pagination_info['total_pages']}** | "
              f"📋 Displaying **{pagination_info['showing_rows']}** rows | "
              f"📊 Total dataset: **{pagination_info['total_rows']:,}** rows")


def render_data_quality_report(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Render data quality report"""
    st.header("4. 🎯 Data Quality Report")
    
    quality_report = get_data_quality_report(df, analysis)
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        completeness = quality_report['completeness']['complete_percentage']
        st.metric(
            "Data Completeness", 
            f"{completeness:.1f}%",
            delta=f"{completeness - 100:.1f}%" if completeness < 100 else None
        )
    
    with col2:
        duplicates = quality_report['consistency']['duplicate_percentage']
        st.metric(
            "Duplicate Rows", 
            f"{duplicates:.1f}%",
            delta=f"+{duplicates:.1f}%" if duplicates > 0 else None
        )
    
    with col3:
        missing_pct = quality_report['completeness']['missing_percentage']
        st.metric(
            "Missing Data", 
            f"{missing_pct:.1f}%",
            delta=f"+{missing_pct:.1f}%" if missing_pct > 0 else None
        )
    
    # Recommendations
    if quality_report['recommendations']:
        st.subheader("💡 Recommendations")
        for rec in quality_report['recommendations']:
            st.warning(f"⚠️ {rec}")


def render_dataset_preparation(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Render enhanced dataset preparation section with auto-preparation tools"""
    # Target column selection
    suggested_targets = detect_target_column(df, analysis)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox(
            "Select Target Column",
            options=analysis['columns'],
            index=analysis['columns'].index(suggested_targets[0]) if suggested_targets else 0,
            help="Choose the column you want to predict"
        )
    
    with col2:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
    
    if suggested_targets:
        st.info(f"💡 Suggested target columns: {', '.join(suggested_targets)}")
    
    # Enhanced preparation section
    if ENHANCED_PREP_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🔧 Data Preparation Assistant")
        
        # Analyze preparation needs
        if st.button("🔍 Analyze Preparation Needs", type="secondary"):
            with st.spinner("Analyzing dataset preparation needs..."):
                prep_analysis = data_prep_tools.analyze_preparation_needs(df, target_column)
                st.session_state.prep_analysis = prep_analysis
        
        # Show analysis results if available
        if 'prep_analysis' in st.session_state:
            prep_analysis = st.session_state.prep_analysis
            
            # Show issues and recommendations
            if prep_analysis['issues']:
                st.markdown("#### ⚠️ Issues Detected")
                for i, issue in enumerate(prep_analysis['issues'], 1):
                    st.warning(f"{i}. {issue}")
            
            # Show recommendations
            recommendations = data_prep_tools.get_preparation_recommendations(prep_analysis)
            
            if recommendations:
                st.markdown("#### 💡 Preparation Recommendations")
                
                # Group by priority
                critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
                high_recs = [r for r in recommendations if r['priority'] == 'High']
                medium_recs = [r for r in recommendations if r['priority'] == 'Medium']
                low_recs = [r for r in recommendations if r['priority'] == 'Low']
                
                # Critical recommendations
                if critical_recs:
                    st.markdown("**🚨 Critical (Required):**")
                    for rec in critical_recs:
                        st.error(f"**{rec['title']}**: {rec['description']}")
                
                # High priority recommendations
                if high_recs:
                    st.markdown("**⚡ High Priority (Recommended):**")
                    for rec in high_recs:
                        st.warning(f"**{rec['title']}**: {rec['description']}")
                
                # Medium and low priority
                if medium_recs or low_recs:
                    with st.expander("📋 Additional Recommendations"):
                        if medium_recs:
                            st.markdown("**Medium Priority:**")
                            for rec in medium_recs:
                                st.info(f"**{rec['title']}**: {rec['description']}")
                        
                        if low_recs:
                            st.markdown("**Low Priority:**")
                            for rec in low_recs:
                                st.info(f"**{rec['title']}**: {rec['description']}")
                
                # Auto-preparation options
                st.markdown("#### 🚀 Auto-Preparation Options")
                
                auto_col1, auto_col2 = st.columns(2)
                
                with auto_col1:
                    if st.button("🔧 Auto-Fix Critical Issues", type="primary"):
                        try:
                            with st.spinner("Auto-fixing critical issues..."):
                                critical_fixes = [rec['action'] for rec in critical_recs if rec['auto_fixable']]
                                prepared_data = data_prep_tools.auto_prepare_dataset(
                                    df, target_column, critical_fixes, test_size
                                )
                            
                            st.session_state.prepared_dataset = prepared_data
                            st.session_state.use_uploaded_dataset = True
                            
                            show_success_message("Critical issues fixed and dataset prepared!")
                            render_preparation_summary(prepared_data)
                            
                        except Exception as e:
                            show_error_message(f"Auto-preparation failed: {str(e)}")
                
                with auto_col2:
                    if st.button("⚡ Auto-Fix All Recommended", type="secondary"):
                        try:
                            with st.spinner("Applying all recommended fixes..."):
                                all_fixes = [rec['action'] for rec in recommendations if rec['auto_fixable']]
                                prepared_data = data_prep_tools.auto_prepare_dataset(
                                    df, target_column, all_fixes, test_size
                                )
                            
                            st.session_state.prepared_dataset = prepared_data
                            st.session_state.use_uploaded_dataset = True
                            
                            show_success_message("All recommended fixes applied and dataset prepared!")
                            render_preparation_summary(prepared_data)
                            
                        except Exception as e:
                            show_error_message(f"Auto-preparation failed: {str(e)}")
                
                # Custom fix selection
                with st.expander("🎛️ Custom Fix Selection"):
                    st.markdown("Select specific fixes to apply:")
                    
                    fix_options = {}
                    for rec in recommendations:
                        if rec['auto_fixable']:
                            fix_options[rec['action']] = st.checkbox(
                                f"{rec['title']} ({rec['priority']} priority)",
                                value=rec['priority'] in ['Critical', 'High'],
                                help=rec['description']
                            )
                    
                    if st.button("🛠️ Apply Selected Fixes"):
                        try:
                            selected_fixes = [action for action, selected in fix_options.items() if selected]
                            if selected_fixes:
                                with st.spinner("Applying selected fixes..."):
                                    prepared_data = data_prep_tools.auto_prepare_dataset(
                                        df, target_column, selected_fixes, test_size
                                    )
                                
                                st.session_state.prepared_dataset = prepared_data
                                st.session_state.use_uploaded_dataset = True
                                
                                show_success_message("Selected fixes applied and dataset prepared!")
                                render_preparation_summary(prepared_data)
                            else:
                                show_warning_message("Please select at least one fix to apply.")
                        except Exception as e:
                            show_error_message(f"Fix application failed: {str(e)}")
    
    # Basic preparation (fallback)
    st.markdown("---")
    st.markdown("### 🎯 Basic Preparation")
    
    if st.button("🚀 Prepare Dataset for ML (Basic)", type="primary"):
        try:
            with st.spinner("Preparing dataset for machine learning..."):
                prepared_data = prepare_dataset_for_ml(df, target_column, test_size)
            
            # Store prepared data in session state
            st.session_state.prepared_dataset = prepared_data
            st.session_state.use_uploaded_dataset = True
            
            show_success_message("Dataset prepared successfully! You can now use it in the main application.")
            render_preparation_summary(prepared_data)
                
        except Exception as e:
            show_error_message(f"Error preparing dataset: {str(e)}")
            
            # Show helpful error information
            if "least populated class" in str(e):
                st.markdown("""
                **💡 This error occurs when some classes have too few samples for stratified splitting.**
                
                **Solutions:**
                1. 🔍 Use the 'Analyze Preparation Needs' button above for automated fixes
                2. 🗑️ Remove classes with very few samples
                3. 📊 Collect more data for minority classes
                4. ⚖️ Use class balancing techniques
                """)


def render_preparation_summary(prepared_data: Dict[str, Any]):
    """Render preparation summary with detailed information"""
    st.markdown("### 📋 Preparation Summary")
    
    # Basic metrics
    prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)
    
    with prep_col1:
        st.metric("Training Samples", len(prepared_data['X_train']))
    with prep_col2:
        st.metric("Test Samples", len(prepared_data['X_test']))
    with prep_col3:
        st.metric("Features", len(prepared_data['feature_names']))
    with prep_col4:
        st.metric("Classes", len(prepared_data['target_names']))
    
    # Additional info
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        if 'stratified' in prepared_data:
            stratified_status = "✅ Yes" if prepared_data['stratified'] else "⚠️ No (due to small classes)"
            st.info(f"**Stratified Split:** {stratified_status}")
    
    with info_col2:
        if 'preparation_log' in prepared_data:
            st.info(f"**Preparation Steps:** {len(prepared_data['preparation_log'])} applied")
    
    # Show preparation log if available
    if 'preparation_log' in prepared_data:
        with st.expander("📝 Preparation Log"):
            for i, log_entry in enumerate(prepared_data['preparation_log'], 1):
                status_icon = "✅" if log_entry['status'] == 'success' else "❌"
                st.write(f"{i}. {status_icon} **{log_entry['action']}**: {log_entry['details']}")
    
    # Feature information
    with st.expander("🔍 Feature Information"):
        feature_info = pd.DataFrame({
            'Feature': prepared_data['feature_names'],
            'Type': [str(prepared_data['X'][col].dtype) for col in prepared_data['feature_names']]
        })
        st.dataframe(feature_info, use_container_width=True)
    
    # Target information with improved display
    with st.expander("🎯 Target Information"):
        target_counts = pd.Series(prepared_data['y']).value_counts().sort_index()
        target_info = pd.DataFrame({
            'Class': [prepared_data['target_names'][i] for i in target_counts.index],
            'Count': target_counts.values,
            'Percentage': (target_counts.values / len(prepared_data['y']) * 100).round(1),
            'Train Count': [len(prepared_data['y_train'][prepared_data['y_train'] == i]) for i in target_counts.index],
            'Test Count': [len(prepared_data['y_test'][prepared_data['y_test'] == i]) for i in target_counts.index]
        })
        st.dataframe(target_info, use_container_width=True)


def render_navigation_sidebar():
    """Render enhanced navigation sidebar for multi-page app"""
    st.sidebar.markdown("# 🧭 Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "🤖 Model Analysis": {
            "key": "model_analysis",
            "description": "Train models and analyze predictions with built-in or uploaded datasets"
        },
        "📁 Data Upload & Prep": {
            "key": "data_upload", 
            "description": "Upload your own datasets and prepare them for machine learning"
        }
    }
    
    # Initialize page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "model_analysis"
    
    # Create navigation buttons with better styling
    for page_name, page_info in pages.items():
        is_current = st.session_state.current_page == page_info["key"]
        
        # Create a container for each page option
        with st.sidebar.container():
            if st.button(
                page_name, 
                key=f"nav_{page_info['key']}", 
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.current_page = page_info["key"]
                st.rerun()
            
            # Show description for current page
            if is_current:
                st.sidebar.caption(f"📝 {page_info['description']}")
    
    st.sidebar.markdown("---")
    
    # Show status information based on current page
    if st.session_state.current_page == "data_upload":
        st.sidebar.markdown("### 📤 Upload Status")
        if 'uploaded_df' in st.session_state:
            df = st.session_state.uploaded_df
            st.sidebar.success(f"✅ Dataset loaded: {st.session_state.get('uploaded_filename', 'Unknown')}")
            st.sidebar.info(f"📊 {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            if 'prepared_dataset' in st.session_state:
                st.sidebar.success("🎯 Dataset prepared for ML")
        else:
            st.sidebar.info("📁 No dataset uploaded yet")
    
    elif st.session_state.current_page == "model_analysis":
        st.sidebar.markdown("### 🎯 Analysis Status")
        
        # Show uploaded dataset status
        if 'prepared_dataset' in st.session_state and st.session_state.get('use_uploaded_dataset', False):
            dataset_info = st.session_state.prepared_dataset
            st.sidebar.success(f"📊 Using uploaded dataset")
            st.sidebar.info(f"🎯 {dataset_info['config']['features']} features, {dataset_info['config']['classes']} classes")
        else:
            st.sidebar.info("🔄 Using built-in datasets")
    
    # Add quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Reset Session", use_container_width=True):
        # Clear session state for fresh start
        keys_to_keep = ['current_page']  # Keep navigation state
        keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_clear:
            del st.session_state[key]
        st.sidebar.success("Session reset!")
        st.rerun()
    
    return st.session_state.current_page
