#!/usr/bin/env python3
"""
Run script for AI Explainer Pro application
"""

import sys
import os
import subprocess
from pathlib import Path

def run_streamlit_app():
    """Run the Streamlit application"""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    app_file = src_dir / "app.py"
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run streamlit with the app.py file
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    
    print("🚀 Starting AI Explainer Pro...")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 Running: {' '.join(cmd)}")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
