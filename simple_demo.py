#!/usr/bin/env python3
"""
Simple Demo Script for Food Supply Chain Resilience Analyzer
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print demo banner"""
    print("\n" + "=" * 60)
    print("=" * 60)
    print("    ╔══════════════════════════════════════════════════════════════╗")
    print("    ║                                                              ║")
    print("    ║    🌾 Food Supply Chain Resilience Analyzer - DEMO          ║")
    print("    ║                                                              ║")
    print("    ║    A comprehensive predictive analytics system for          ║")
    print("    ║    identifying food supply chain disruptions                ║")
    print("    ║                                                              ║")
    print("    ╚══════════════════════════════════════════════════════════════╝")
    print("=" * 60)
    print("=" * 60)

def main():
    """Main demo function"""
    print_banner()
    
    print("This demo will showcase the Food Supply Chain Resilience Analyzer:")
    print("1. 📊 Data Generation (Synthetic data with 25+ features)")
    print("2. 🔄 ETL Pipeline (Extract, Transform, Load)")
    print("3. 📈 Data Analysis (Key insights and metrics)")
    print("4. 📊 Dashboard Preview (Interactive features)")
    print("\nStarting demo...\n")
    
    # Step 1: Data Generation
    print("=" * 60)
    print("STEP 1: Data Generation")
    print("=" * 60)
    
    try:
        # Check if data already exists
        if os.path.exists('data/food_supply_chain_data.csv'):
            print("✅ Data already exists - loading existing data...")
            data = pd.read_csv('data/food_supply_chain_data.csv')
            print(f"✅ Loaded {len(data):,} data points with {len(data.columns)} features")
        else:
            print("❌ Data not found. Please run the data generator first.")
            return
            
        print(f"📊 Date range: {data['date'].min()} to {data['date'].max()}")
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return
    
    # Step 2: Data Analysis
    print("\n" + "=" * 60)
    print("STEP 2: Data Analysis")
    print("=" * 60)
    
    try:
        # Basic statistics
        print("📈 Key Statistics:")
        print(f"   📅 Total records: {len(data):,}")
        print(f"   🌡️  Avg temperature: {data['temperature_celsius'].mean():.1f}°C")
        print(f"   🌧️  Avg precipitation: {data['precipitation_mm'].mean():.1f}mm")
        print(f"   📦 Avg production: {data['production_volume_tonnes'].mean():.0f} tonnes")
        print(f"   ⚠️  Disruption rate: {(data['supply_chain_disruption'].sum() / len(data) * 100):.1f}%")
        print(f"   🏆 Avg quality score: {data['quality_score_percent'].mean():.1f}%")
        
        # Check for processed data
        if os.path.exists('data/feature_engineered_data.csv'):
            print("\n✅ Feature engineered data found!")
            feature_data = pd.read_csv('data/feature_engineered_data.csv')
            print(f"   🔧 Engineered features: {len(feature_data.columns)}")
        else:
            print("\n⚠️  Feature engineered data not found. Run ETL pipeline to create it.")
            
    except Exception as e:
        print(f"❌ Data analysis failed: {str(e)}")
        return
    
    # Step 3: Model Status
    print("\n" + "=" * 60)
    print("STEP 3: Model Status")
    print("=" * 60)
    
    try:
        # Check for trained models
        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            print("🤖 Trained Models:")
            for file in model_files:
                if file.endswith('.h5') or file.endswith('.pkl'):
                    print(f"   ✅ {file}")
        else:
            print("❌ No models directory found")
            
        # Check for plots
        plots_dir = 'plots'
        if os.path.exists(plots_dir):
            plot_files = os.listdir(plots_dir)
            print(f"\n📊 Generated Plots: {len(plot_files)} files")
        else:
            print("\n❌ No plots directory found")
            
    except Exception as e:
        print(f"❌ Model status check failed: {str(e)}")
        return
    
    # Step 4: Dashboard Status
    print("\n" + "=" * 60)
    print("STEP 4: Dashboard Status")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        if '8501' in result.stdout:
            print("✅ Dashboard is running on http://localhost:8501")
            print("🌐 Open your browser to access the interactive dashboard!")
        else:
            print("❌ Dashboard not running. Start it with: streamlit run src/dashboard/main.py")
            
    except Exception as e:
        print(f"❌ Dashboard status check failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("🎉 Your Food Supply Chain Resilience Analyzer is ready!")
    print("🌐 Access the dashboard at: http://localhost:8501")
    print("📊 Explore the data, models, and predictions!")

if __name__ == "__main__":
    main() 