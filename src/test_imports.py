"""
Quick test of the modular app structure
"""

import streamlit as st
import sys
from pathlib import Path

# Add the src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from utils.config import APP_TITLE
    st.write(f"✅ Config loaded: {APP_TITLE}")
    
    from utils.logging_config import setup_logging
    st.write("✅ Logging config imported successfully")
    
    from utils.data_handler import get_available_datasets
    datasets = get_available_datasets()
    st.write(f"✅ Data handler loaded: {len(datasets)} datasets available")
    
    from utils.model_handler import get_available_models
    models = get_available_models()
    st.write(f"✅ Model handler loaded: {len(models)} models available")
    
    from components.ui_components import setup_page_config
    st.write("✅ UI components imported successfully")
    
    st.success("🎉 All imports working correctly!")
    st.info("The modular structure is properly set up!")
    
except Exception as e:
    st.error(f"❌ Import error: {str(e)}")
    st.write("Debug info:")
    st.write(f"Current directory: {Path.cwd()}")
    st.write(f"Src directory: {src_dir}")
    st.write(f"Python path: {sys.path[:3]}")  # Show first 3 entries
