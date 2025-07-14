"""
ETL Pipeline for Food Supply Chain Resilience Analyzer
Handles data extraction, transformation, and loading
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from .data_generator import FoodSupplyChainDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETLPipeline:
    """Main ETL pipeline for food supply chain data"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_data_file = os.path.join(data_dir, 'food_supply_chain_data.csv')
        self.processed_data_file = os.path.join(data_dir, 'processed_food_supply_chain_data.csv')
        self.feature_engineered_file = os.path.join(data_dir, 'feature_engineered_data.csv')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def extract_data(self):
        """Extract data from various sources"""
        logger.info("Starting data extraction...")
        
        try:
            # Check if raw data exists, if not generate it
            if not os.path.exists(self.raw_data_file):
                logger.info("Raw data not found. Generating synthetic data...")
                generator = FoodSupplyChainDataGenerator()
                data = generator.generate_all_data()
                generator.save_data(data, 'food_supply_chain_data.csv')
            
            # Load the data
            data = pd.read_csv(self.raw_data_file)
            data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"Successfully extracted {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            raise
    
    def transform_data(self, data):
        """Transform and clean the data"""
        logger.info("Starting data transformation...")
        
        try:
            # Create a copy to avoid modifying original data
            transformed_data = data.copy()
            
            # Handle missing values
            numeric_columns = transformed_data.select_dtypes(include=[np.number]).columns
            transformed_data[numeric_columns] = transformed_data[numeric_columns].fillna(
                transformed_data[numeric_columns].median()
            )
            
            # Handle categorical missing values
            categorical_columns = transformed_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                transformed_data[col] = transformed_data[col].fillna('Unknown')
            
            # Remove outliers using IQR method for key numeric columns
            outlier_columns = ['temperature_celsius', 'precipitation_mm', 'production_volume_tonnes']
            for col in outlier_columns:
                if col in transformed_data.columns:
                    Q1 = transformed_data[col].quantile(0.25)
                    Q3 = transformed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    transformed_data[col] = np.clip(transformed_data[col], lower_bound, upper_bound)
            
            # Add derived features
            transformed_data['month'] = transformed_data['date'].dt.month
            transformed_data['year'] = transformed_data['date'].dt.year
            transformed_data['day_of_week'] = transformed_data['date'].dt.dayofweek
            transformed_data['quarter'] = transformed_data['date'].dt.quarter
            
            # Create seasonal indicators
            transformed_data['is_summer'] = transformed_data['month'].isin([6, 7, 8]).astype(int)
            transformed_data['is_winter'] = transformed_data['month'].isin([12, 1, 2]).astype(int)
            
            logger.info(f"Successfully transformed {len(transformed_data)} records")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise
    
    def feature_engineering(self, data):
        """Perform feature engineering"""
        logger.info("Starting feature engineering...")
        
        try:
            engineered_data = data.copy()
            
            # Create lag features for time series analysis
            lag_features = ['production_volume_tonnes', 'temperature_celsius', 'precipitation_mm']
            for feature in lag_features:
                if feature in engineered_data.columns:
                    # 1-day lag
                    engineered_data[f'{feature}_lag_1'] = engineered_data[feature].shift(1)
                    # 7-day lag
                    engineered_data[f'{feature}_lag_7'] = engineered_data[feature].shift(7)
                    # 30-day lag
                    engineered_data[f'{feature}_lag_30'] = engineered_data[feature].shift(30)
            
            # Create rolling averages
            for feature in lag_features:
                if feature in engineered_data.columns:
                    # 7-day rolling average
                    engineered_data[f'{feature}_rolling_7'] = engineered_data[feature].rolling(window=7).mean()
                    # 30-day rolling average
                    engineered_data[f'{feature}_rolling_30'] = engineered_data[feature].rolling(window=30).mean()
            
            # Create composite risk score
            risk_factors = [
                'extreme_weather_event', 'drought_indicator', 'trade_restrictions',
                'sanctions_active', 'conflict_indicator', 'supply_chain_disruption'
            ]
            
            available_risk_factors = [f for f in risk_factors if f in engineered_data.columns]
            if available_risk_factors:
                engineered_data['composite_risk_score'] = engineered_data[available_risk_factors].sum(axis=1)
            
            # Create supply chain efficiency score
            efficiency_factors = ['quality_score_percent', 'supplier_reliability_score']
            available_efficiency_factors = [f for f in efficiency_factors if f in engineered_data.columns]
            if available_efficiency_factors:
                engineered_data['supply_chain_efficiency'] = engineered_data[available_efficiency_factors].mean(axis=1)
            
            # Create climate stress index
            if 'temperature_celsius' in engineered_data.columns and 'precipitation_mm' in engineered_data.columns:
                # Normalize temperature and precipitation
                temp_normalized = (engineered_data['temperature_celsius'] - engineered_data['temperature_celsius'].mean()) / engineered_data['temperature_celsius'].std()
                precip_normalized = (engineered_data['precipitation_mm'] - engineered_data['precipitation_mm'].mean()) / engineered_data['precipitation_mm'].std()
                
                engineered_data['climate_stress_index'] = np.sqrt(temp_normalized**2 + precip_normalized**2)

            # Add disruption_occurred column: 1 if disruption_risk > 0 or supply_chain_disruption > 0, else 0
            if 'disruption_risk' in engineered_data.columns and 'supply_chain_disruption' in engineered_data.columns:
                engineered_data['disruption_occurred'] = ((engineered_data['disruption_risk'] > 0) | (engineered_data['supply_chain_disruption'] > 0)).astype(int)
            
            # Fill NaN values created by lag features
            engineered_data = engineered_data.fillna(method='bfill')
            
            logger.info(f"Successfully engineered features for {len(engineered_data)} records")
            return engineered_data
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def load_data(self, data, filename):
        """Load processed data to storage"""
        logger.info(f"Loading data to {filename}...")
        
        try:
            data.to_csv(filename, index=False)
            logger.info(f"Successfully loaded {len(data)} records to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}")
            raise
    
    def validate_data(self, data):
        """Validate data quality"""
        logger.info("Starting data validation...")
        
        validation_results = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_records': data.duplicated().sum(),
            'date_range': {
                'start': data['date'].min(),
                'end': data['date'].max()
            },
            'columns': list(data.columns)
        }
        
        # Check for data quality issues
        quality_issues = []
        
        # Check for missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            quality_issues.append(f"Missing values in columns: {missing_cols}")
        
        # Check for negative values in positive-only columns
        positive_only_cols = ['production_volume_tonnes', 'inventory_level_tonnes', 'quality_score_percent']
        for col in positive_only_cols:
            if col in data.columns and (data[col] < 0).any():
                quality_issues.append(f"Negative values found in {col}")
        
        validation_results['quality_issues'] = quality_issues
        validation_results['is_valid'] = len(quality_issues) == 0
        
        logger.info(f"Data validation completed. Issues found: {len(quality_issues)}")
        return validation_results
    
    def run_pipeline(self):
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        
        try:
            # Extract
            raw_data = self.extract_data()
            
            # Transform
            transformed_data = self.transform_data(raw_data)
            self.load_data(transformed_data, self.processed_data_file)
            
            # Feature Engineering
            engineered_data = self.feature_engineering(transformed_data)
            self.load_data(engineered_data, self.feature_engineered_file)
            
            # Validate
            validation_results = self.validate_data(engineered_data)
            
            logger.info("ETL pipeline completed successfully!")
            logger.info(f"Final dataset: {validation_results['total_records']} records, {len(validation_results['columns'])} features")
            
            return {
                'raw_data': raw_data,
                'processed_data': transformed_data,
                'engineered_data': engineered_data,
                'validation_results': validation_results
            }
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Run the ETL pipeline
    pipeline = ETLPipeline()
    results = pipeline.run_pipeline()
    
    print("\nETL Pipeline Summary:")
    print(f"Raw data records: {len(results['raw_data'])}")
    print(f"Processed data records: {len(results['processed_data'])}")
    print(f"Engineered data records: {len(results['engineered_data'])}")
    print(f"Data validation passed: {results['validation_results']['is_valid']}") 