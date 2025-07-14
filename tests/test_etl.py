"""
Unit tests for ETL pipeline components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_generator import FoodSupplyChainDataGenerator
from etl.pipeline import ETLPipeline

class TestDataGenerator:
    """Test cases for data generator"""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization"""
        generator = FoodSupplyChainDataGenerator()
        assert generator.start_date == pd.to_datetime('2020-01-01')
        assert generator.end_date == pd.to_datetime('2024-12-31')
        assert len(generator.date_range) > 0
    
    def test_climate_data_generation(self):
        """Test climate data generation"""
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        climate_data = generator.generate_climate_data()
        
        assert isinstance(climate_data, pd.DataFrame)
        assert len(climate_data) == 31  # January has 31 days
        assert 'temperature_celsius' in climate_data.columns
        assert 'precipitation_mm' in climate_data.columns
        assert 'extreme_weather_event' in climate_data.columns
        assert climate_data['temperature_celsius'].dtype in [np.float64, np.float32]
        assert climate_data['precipitation_mm'].min() >= 0
    
    def test_government_data_generation(self):
        """Test government data generation"""
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        gov_data = generator.generate_government_data()
        
        assert isinstance(gov_data, pd.DataFrame)
        assert len(gov_data) == 31
        assert 'gdp_growth_percent' in gov_data.columns
        assert 'inflation_rate_percent' in gov_data.columns
        assert 'trade_restrictions' in gov_data.columns
        assert gov_data['trade_restrictions'].isin([0, 1]).all()
    
    def test_geopolitical_data_generation(self):
        """Test geopolitical data generation"""
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        geo_data = generator.generate_geopolitical_data()
        
        assert isinstance(geo_data, pd.DataFrame)
        assert len(geo_data) == 31
        assert 'political_stability_index' in geo_data.columns
        assert 'trade_relations_index' in geo_data.columns
        assert geo_data['political_stability_index'].min() >= 0
        assert geo_data['political_stability_index'].max() <= 100
    
    def test_supply_chain_data_generation(self):
        """Test supply chain data generation"""
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        supply_data = generator.generate_supply_chain_data()
        
        assert isinstance(supply_data, pd.DataFrame)
        assert len(supply_data) == 31
        assert 'production_volume_tonnes' in supply_data.columns
        assert 'transportation_cost_per_ton' in supply_data.columns
        assert 'quality_score_percent' in supply_data.columns
        assert supply_data['production_volume_tonnes'].min() >= 0
        assert supply_data['quality_score_percent'].min() >= 0
        assert supply_data['quality_score_percent'].max() <= 100
    
    def test_all_data_generation(self):
        """Test complete data generation"""
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        all_data = generator.generate_all_data()
        
        assert isinstance(all_data, pd.DataFrame)
        assert len(all_data) == 31
        assert len(all_data.columns) >= 20  # Should have many features after merging
        assert 'date' in all_data.columns
        assert not all_data['date'].isnull().any()

class TestETLPipeline:
    """Test cases for ETL pipeline"""
    
    def setup_method(self):
        """Setup test data"""
        # Create temporary data directory
        os.makedirs('test_data', exist_ok=True)
        
        # Generate test data
        generator = FoodSupplyChainDataGenerator('2023-01-01', '2023-01-31')
        test_data = generator.generate_all_data()
        test_data.to_csv('test_data/test_food_supply_chain_data.csv', index=False)
        
        self.pipeline = ETLPipeline('test_data')
    
    def teardown_method(self):
        """Cleanup test data"""
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.data_dir == 'test_data'
        assert 'food_supply_chain_data.csv' in self.pipeline.raw_data_file
    
    def test_data_extraction(self):
        """Test data extraction"""
        data = self.pipeline.extract_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'date' in data.columns
        assert pd.api.types.is_datetime64_any_dtype(data['date'])
    
    def test_data_transformation(self):
        """Test data transformation"""
        raw_data = self.pipeline.extract_data()
        transformed_data = self.pipeline.transform_data(raw_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(raw_data)
        assert 'month' in transformed_data.columns
        assert 'year' in transformed_data.columns
        assert 'is_summer' in transformed_data.columns
        assert 'is_winter' in transformed_data.columns
        
        # Check that derived features are binary
        assert transformed_data['is_summer'].isin([0, 1]).all()
        assert transformed_data['is_winter'].isin([0, 1]).all()
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        raw_data = self.pipeline.extract_data()
        transformed_data = self.pipeline.transform_data(raw_data)
        engineered_data = self.pipeline.feature_engineering(transformed_data)
        
        assert isinstance(engineered_data, pd.DataFrame)
        assert len(engineered_data) == len(transformed_data)
        
        # Check for lag features
        lag_features = [col for col in engineered_data.columns if 'lag_' in col]
        assert len(lag_features) > 0
        
        # Check for rolling features
        rolling_features = [col for col in engineered_data.columns if 'rolling_' in col]
        assert len(rolling_features) > 0
        
        # Check for composite features
        assert 'composite_risk_score' in engineered_data.columns or 'supply_chain_efficiency' in engineered_data.columns
    
    def test_data_validation(self):
        """Test data validation"""
        raw_data = self.pipeline.extract_data()
        transformed_data = self.pipeline.transform_data(raw_data)
        engineered_data = self.pipeline.feature_engineering(transformed_data)
        
        validation_results = self.pipeline.validate_data(engineered_data)
        
        assert isinstance(validation_results, dict)
        assert 'total_records' in validation_results
        assert 'missing_values' in validation_results
        assert 'duplicate_records' in validation_results
        assert 'date_range' in validation_results
        assert 'columns' in validation_results
        assert 'quality_issues' in validation_results
        assert 'is_valid' in validation_results
        
        assert validation_results['total_records'] == len(engineered_data)
        assert isinstance(validation_results['is_valid'], bool)
    
    def test_complete_pipeline(self):
        """Test complete ETL pipeline"""
        results = self.pipeline.run_pipeline()
        
        assert isinstance(results, dict)
        assert 'raw_data' in results
        assert 'processed_data' in results
        assert 'engineered_data' in results
        assert 'validation_results' in results
        
        # Check that files were created
        assert os.path.exists(self.pipeline.processed_data_file)
        assert os.path.exists(self.pipeline.feature_engineered_file)
        
        # Check data integrity
        assert len(results['raw_data']) == len(results['processed_data'])
        assert len(results['processed_data']) == len(results['engineered_data'])

if __name__ == "__main__":
    pytest.main([__file__]) 