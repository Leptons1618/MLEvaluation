"""
Simple test of new auto-preparation features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

def test_simple():
    """Simple test of enhanced features"""
    print("Testing enhanced auto-preparation...")
    
    try:
        from utils.data_preparation_enhanced import data_prep_tools
        print("✅ Enhanced preparation module loaded")
        
        # Create test data
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(100, 50, 50),  # Different scale
            'target': ['A'] * 30 + ['B'] * 15 + ['C'] * 5  # Imbalanced
        }
        df = pd.DataFrame(data)
        
        print(f"Created test dataset: {df.shape}")
        
        # Test analysis
        analysis = data_prep_tools.analyze_preparation_needs(df, 'target')
        print(f"✅ Analysis completed: {len(analysis['issues'])} issues detected")
        
        # Test recommendations
        recommendations = data_prep_tools.get_preparation_recommendations(analysis)
        print(f"✅ Recommendations generated: {len(recommendations)} recommendations")
        
        # Test basic auto-preparation
        prepared_data = data_prep_tools.auto_prepare_dataset(df, 'target', ['scale_features'])
        print(f"✅ Auto-preparation successful: {prepared_data['X'].shape}")
        
        print("🎉 All enhanced features are working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
