#!/usr/bin/env python3
"""
Project Summary for Food Supply Chain Resilience Analyzer
Shows the current status and generated data
"""

import os
import pandas as pd
from datetime import datetime

def print_project_summary():
    """Print a summary of the project status"""
    
    print("=" * 80)
    print("🌾 FOOD SUPPLY CHAIN RESILIENCE ANALYZER - PROJECT SUMMARY")
    print("=" * 80)
    
    # Check data files
    print("\n📊 DATA STATUS:")
    data_files = []
    if os.path.exists('data/food_supply_chain_data.csv'):
        df = pd.read_csv('data/food_supply_chain_data.csv')
        data_files.append(f"✅ Raw data: {len(df):,} records, {len(df.columns)} features")
    
    if os.path.exists('data/processed_food_supply_chain_data.csv'):
        df = pd.read_csv('data/processed_food_supply_chain_data.csv')
        data_files.append(f"✅ Processed data: {len(df):,} records, {len(df.columns)} features")
    
    if os.path.exists('data/feature_engineered_data.csv'):
        df = pd.read_csv('data/feature_engineered_data.csv')
        data_files.append(f"✅ Feature engineered data: {len(df):,} records, {len(df.columns)} features")
    
    for file_status in data_files:
        print(f"   {file_status}")
    
    # Check components
    print("\n🔧 COMPONENTS STATUS:")
    components = [
        ("ETL Pipeline", "src/etl/pipeline.py"),
        ("Data Generator", "src/etl/data_generator.py"),
        ("Data Validator", "src/validation/data_validator.py"),
        ("LSTM Model", "src/models/lstm_model.py"),
        ("ARIMA Model", "src/models/arima_model.py"),
        ("Dashboard", "src/dashboard/main.py"),
        ("Configuration", "config/config.py")
    ]
    
    for component_name, file_path in components:
        if os.path.exists(file_path):
            print(f"   ✅ {component_name}: Available")
        else:
            print(f"   ❌ {component_name}: Missing")
    
    # Show sample data insights
    print("\n📈 SAMPLE DATA INSIGHTS:")
    if os.path.exists('data/feature_engineered_data.csv'):
        df = pd.read_csv('data/feature_engineered_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"   📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"   🌡️  Avg temperature: {df['temperature_celsius'].mean():.1f}°C")
        print(f"   🌧️  Avg precipitation: {df['precipitation_mm'].mean():.1f}mm")
        print(f"   📦 Avg production: {df['production_volume_tonnes'].mean():.0f} tonnes")
        print(f"   ⚠️  Disruption rate: {df['supply_chain_disruption'].mean()*100:.1f}%")
        print(f"   🏆 Avg quality score: {df['quality_score_percent'].mean():.1f}%")
    
    # Show next steps
    print("\n🚀 NEXT STEPS:")
    print("   1. 🌐 Dashboard is running at: http://localhost:8501")
    print("   2. 🧠 Train ML models: python src/models/train_models.py")
    print("   3. 📊 Explore data in the interactive dashboard")
    print("   4. 🔍 Run data validation: python src/validation/data_validator.py")
    
    print("\n" + "=" * 80)
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    print_project_summary() 