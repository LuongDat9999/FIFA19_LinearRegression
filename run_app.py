#!/usr/bin/env python3
"""
Simple script to run the FIFA 19 Streamlit app
==============================================

This script checks if the required data files exist and runs the Streamlit app.
If data files don't exist, it will run the data processing first.

Usage:
    python run_app.py
"""

import os
import subprocess
import sys
from pathlib import Path

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'processed_data.csv',
        'app_data.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def run_data_processing():
    """Run the data processing script"""
    print("🔧 Data files not found. Running data processing...")
    
    try:
        # Check if fifa_optimized.py exists
        if os.path.exists('fifa_optimized.py'):
            print("📊 Running FIFA data processing...")
            result = subprocess.run([sys.executable, 'fifa_optimized.py'], 
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Data processing completed successfully!")
                return True
            else:
                print(f"❌ Data processing failed:")
                print(result.stderr)
                return False
        else:
            print("❌ fifa_optimized.py not found!")
            print("Please run the data processing manually first.")
            return False
            
    except Exception as e:
        print(f"❌ Error running data processing: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    print("🚀 Starting Streamlit app...")
    
    try:
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print("❌ app.py not found!")
            return False
        
        # Check if streamlit is installed
        try:
            import streamlit
            print("✅ Streamlit is installed")
        except ImportError:
            print("❌ Streamlit not installed. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit'])
        
        # Run the app
        print("🌐 Starting Streamlit app on http://localhost:8501")
        print("Press Ctrl+C to stop the app")
        
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False

def main():
    """Main function"""
    print("⚽ FIFA 19 Player Analysis & Prediction App")
    print("=" * 50)
    
    # Check if data files exist
    missing_files = check_data_files()
    
    if missing_files:
        print(f"📁 Missing data files: {', '.join(missing_files)}")
        
        # Try to run data processing
        if not run_data_processing():
            print("\n❌ Cannot proceed without data files.")
            print("Please ensure you have:")
            print("1. The original FIFA dataset")
            print("2. Run fifa_optimized.py to process the data")
            print("3. Or manually create the required CSV files")
            return
        
        # Check again after processing
        missing_files = check_data_files()
        if missing_files:
            print(f"❌ Still missing files: {', '.join(missing_files)}")
            return
    
    print("✅ All required data files found!")
    
    # Run the Streamlit app
    run_streamlit_app()

if __name__ == "__main__":
    main()
