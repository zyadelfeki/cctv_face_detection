#!/usr/bin/env python
"""
Launch script for the CCTV Face Detection Dashboard.
Run with: python run_dashboard.py
"""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    required = ['streamlit', 'plotly', 'opencv-python']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', *missing
        ])


def main():
    """Launch the dashboard."""
    # Set up paths
    project_root = Path(__file__).parent
    dashboard_path = project_root / 'src' / 'web' / 'dashboard.py'
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Check requirements
    check_requirements()
    
    # Launch Streamlit
    print("\n" + "="*50)
    print("🎥 CCTV Face Detection Dashboard")
    print("="*50)
    print(f"\nStarting dashboard...")
    print("Open http://localhost:8501 in your browser\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', '8501',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ])


if __name__ == '__main__':
    main()
