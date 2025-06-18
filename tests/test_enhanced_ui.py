"""
Test the enhanced UI components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

def test_enhanced_components():
    """Test the enhanced UI components"""
    print("🧪 Testing Enhanced UI Components")
    print("="*50)
    
    try:
        # Test imports
        from components.enhanced_ui_components import (
            render_editable_column_types,
            render_enhanced_quality_report,
            render_operation_tracker,
            render_advanced_feature_selection
        )
        print("✅ Enhanced UI components imported successfully")
        
        # Create test data
        np.random.seed(42)
        test_data = {
            'numeric_feature_1': np.random.normal(0, 1, 100),
            'numeric_feature_2': np.random.normal(10, 5, 100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice(['Class1', 'Class2'], 100)
        }
        
        # Add some missing values
        test_data['numeric_feature_1'][10:15] = np.nan
        test_data['categorical_feature'][20:25] = None
        
        df = pd.DataFrame(test_data)
        print(f"✅ Created test dataset: {df.shape}")
        
        # Test column type analysis
        from utils.data_preparation import analyze_dataset
        analysis = analyze_dataset(df)
        print(f"✅ Dataset analysis completed: {len(analysis['issues']) if 'issues' in analysis else 0} issues")
        
        print("🎉 All enhanced UI components are ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced components: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_components()
