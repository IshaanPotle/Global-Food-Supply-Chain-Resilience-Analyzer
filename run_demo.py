#!/usr/bin/env python3
"""
Comprehensive Demo Script for Food Supply Chain Resilience Analyzer
Demonstrates the complete pipeline from data generation to model predictions
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.etl.data_generator import FoodSupplyChainDataGenerator
from src.etl.pipeline import ETLPipeline
from src.validation.data_validator import DataValidator
from src.models.lstm_model import LSTMDisruptionPredictor
from src.models.arima_model import ARIMAForecaster
from src.models.train_models import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print demo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🌾 Food Supply Chain Resilience Analyzer - DEMO          ║
    ║                                                              ║
    ║    A comprehensive predictive analytics system for          ║
    ║    identifying food supply chain disruptions                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def step_1_data_generation():
    """Step 1: Generate synthetic data"""
    print("\n" + "="*60)
    print("STEP 1: Data Generation")
    print("="*60)
    
    logger.info("Starting data generation...")
    
    try:
        # Create data generator
        generator = FoodSupplyChainDataGenerator()
        
        # Generate all data
        data = generator.generate_all_data()
        
        # Save data
        generator.save_data(data)
        
        print(f"✅ Generated {len(data):,} data points with {len(data.columns)} features")
        print(f"📊 Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        print(f"❌ Data generation failed: {str(e)}")
        return False

def step_2_etl_pipeline():
    """Step 2: Run ETL pipeline"""
    print("\n" + "="*60)
    print("STEP 2: ETL Pipeline")
    print("="*60)
    
    logger.info("Starting ETL pipeline...")
    
    try:
        # Initialize ETL pipeline
        pipeline = ETLPipeline()
        
        # Run complete pipeline
        results = pipeline.run_pipeline()
        
        print(f"✅ ETL pipeline completed successfully!")
        print(f"📈 Raw data: {len(results['raw_data']):,} records")
        print(f"🔄 Processed data: {len(results['processed_data']):,} records")
        print(f"⚙️  Engineered data: {len(results['engineered_data']):,} records")
        print(f"✅ Data validation: {'PASSED' if results['validation_results']['is_valid'] else 'FAILED'}")
        
        return True
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        print(f"❌ ETL pipeline failed: {str(e)}")
        return False

def step_3_data_validation():
    """Step 3: Data validation"""
    print("\n" + "="*60)
    print("STEP 3: Data Validation")
    print("="*60)
    
    logger.info("Starting data validation...")
    
    try:
        # Load data
        data = pd.read_csv('data/feature_engineered_data.csv')
        data['date'] = pd.to_datetime(data['date'])
        
        # Initialize validator
        validator = DataValidator()
        
        # Run validation
        validation_report = validator.validate_dataset(data)
        
        # Save validation report
        validator.save_validation_report(validation_report, 'validation_report.json')
        
        print(f"✅ Data validation completed!")
        print(f"📊 Quality score: {validation_report['summary']['quality_score']:.2f}")
        print(f"🔍 Schema valid: {'YES' if validation_report['summary']['schema_valid'] else 'NO'}")
        print(f"📋 Business rules valid: {'YES' if validation_report['summary']['business_rules_valid'] else 'NO'}")
        print(f"⏰ Time series consistent: {'YES' if validation_report['summary']['time_series_consistent'] else 'NO'}")
        
        if validation_report['recommendations']:
            print(f"💡 Recommendations:")
            for rec in validation_report['recommendations']:
                print(f"   - {rec}")
        
        return validation_report['overall_valid']
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        print(f"❌ Data validation failed: {str(e)}")
        return False

def step_4_model_training():
    """Step 4: Model training"""
    print("\n" + "="*60)
    print("STEP 4: Model Training")
    print("="*60)
    
    logger.info("Starting model training...")
    
    try:
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Run complete training
        results = trainer.run_complete_training(hyperparameter_tuning=True)
        
        print(f"✅ Model training completed successfully!")
        
        # LSTM results
        if 'lstm' in results['lstm_results']:
            lstm_perf = results['lstm_results']['evaluation_results']
            print(f"🧠 LSTM Model:")
            print(f"   - Overall Accuracy: {lstm_perf['overall_accuracy']:.4f}")
            print(f"   - Overall Precision: {lstm_perf['overall_precision']:.4f}")
            print(f"   - Overall Recall: {lstm_perf['overall_recall']:.4f}")
        
        # ARIMA results
        if 'arima' in results['arima_results']:
            print(f"📈 ARIMA Models:")
            for target_col, arima_result in results['arima_results'].items():
                eval_result = arima_result['evaluation_results']
                print(f"   - {target_col}: MAPE {eval_result['mape']:.2f}%, RMSE {eval_result['rmse']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        print(f"❌ Model training failed: {str(e)}")
        return False

def step_5_dashboard():
    """Step 5: Launch dashboard"""
    print("\n" + "="*60)
    print("STEP 5: Dashboard Launch")
    print("="*60)
    
    logger.info("Preparing to launch dashboard...")
    
    try:
        print("🚀 Launching Streamlit dashboard...")
        print("📊 Dashboard will open in your browser")
        print("🔄 To stop the dashboard, press Ctrl+C in the terminal")
        print("\n" + "="*60)
        
        # Launch dashboard
        os.system("streamlit run src/dashboard/main.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Dashboard launch failed: {str(e)}")
        print(f"❌ Dashboard launch failed: {str(e)}")
        return False

def main():
    """Main demo function"""
    print_banner()
    
    print("This demo will run the complete Food Supply Chain Resilience Analyzer pipeline:")
    print("1. 📊 Data Generation (300,000+ synthetic data points)")
    print("2. 🔄 ETL Pipeline (Extract, Transform, Load)")
    print("3. ✅ Data Validation (Quality assurance)")
    print("4. 🧠 Model Training (LSTM + ARIMA with hyperparameter tuning)")
    print("5. 📈 Dashboard Launch (Interactive visualization)")
    
    print("\nStarting demo...")
    start_time = time.time()
    
    # Step 1: Data Generation
    if not step_1_data_generation():
        print("\n❌ Demo failed at Step 1")
        return
    
    # Step 2: ETL Pipeline
    if not step_2_etl_pipeline():
        print("\n❌ Demo failed at Step 2")
        return
    
    # Step 3: Data Validation
    if not step_3_data_validation():
        print("\n⚠️  Data validation issues detected, but continuing...")
    
    # Step 4: Model Training
    if not step_4_model_training():
        print("\n❌ Demo failed at Step 4")
        return
    
    # Step 5: Dashboard
    print("\n🎉 All steps completed successfully!")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("📊 Your Food Supply Chain Resilience Analyzer is ready!")
    print("🌐 The dashboard should open automatically in your browser")
    print("📁 Check the following directories for outputs:")
    print("   - data/ : Processed datasets")
    print("   - models/ : Trained ML models")
    print("   - results/ : Training results and reports")
    print("   - plots/ : Generated visualizations")
    print("   - logs/ : Application logs")
    
    # Launch dashboard
    step_5_dashboard()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        print("👋 Thanks for trying the Food Supply Chain Resilience Analyzer!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(f"Demo failed with unexpected error: {str(e)}") 