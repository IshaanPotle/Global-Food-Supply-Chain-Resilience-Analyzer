"""
Configuration file for Food Supply Chain Resilience Analyzer
Centralized settings and parameters
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the project"""
    
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
    
    # Data files
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'food_supply_chain_data.csv')
    PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_food_supply_chain_data.csv')
    FEATURE_ENGINEERED_FILE = os.path.join(DATA_DIR, 'feature_engineered_data.csv')
    
    # Model files
    LSTM_MODEL_FILE = os.path.join(MODELS_DIR, 'lstm_disruption_predictor.h5')
    ARIMA_MODEL_FILES = {
        'production': os.path.join(MODELS_DIR, 'arima_production_volume_tonnes_forecaster.pkl'),
        'temperature': os.path.join(MODELS_DIR, 'arima_temperature_celsius_forecaster.pkl'),
        'precipitation': os.path.join(MODELS_DIR, 'arima_precipitation_mm_forecaster.pkl'),
        'transportation': os.path.join(MODELS_DIR, 'arima_transportation_cost_per_ton_forecaster.pkl')
    }
    
    # ETL Configuration
    ETL_CONFIG = {
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'data_points': 300000,
        'batch_size': 1000
    }
    
    # LSTM Model Configuration
    LSTM_CONFIG = {
        'sequence_length': 30,
        'prediction_horizon': 7,
        'default_units': 50,
        'default_dropout': 0.2,
        'default_learning_rate': 0.001,
        'default_batch_size': 32,
        'max_epochs': 100,
        'patience': 10,
        'validation_split': 0.2
    }
    
    # ARIMA Model Configuration
    ARIMA_CONFIG = {
        'max_p': 5,
        'max_d': 2,
        'max_q': 5,
        'forecast_steps': 30,
        'seasonal_periods': 12
    }
    
    # Feature Engineering Configuration
    FEATURE_ENGINEERING_CONFIG = {
        'lag_features': [1, 7, 30],
        'rolling_windows': [7, 30],
        'risk_thresholds': {
            'temperature_high': 35,
            'temperature_low': 0,
            'precipitation_high': 100,
            'production_min': 0,
            'quality_min': 70
        }
    }
    
    # Data Validation Configuration
    VALIDATION_CONFIG = {
        'quality_threshold': 0.8,
        'missing_value_threshold': 0.1,
        'outlier_threshold': 3.0,
        'business_rules': {
            'production_positive_when_no_disruption': True,
            'quality_high_when_reliability_high': True,
            'stability_consistent': True
        }
    }
    
    # Dashboard Configuration
    DASHBOARD_CONFIG = {
        'refresh_interval': 300,  # 5 minutes
        'max_data_points': 10000,
        'chart_height': 400,
        'risk_thresholds': {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': os.path.join(PROJECT_ROOT, 'logs', 'app.log')
    }
    
    # Performance Metrics
    PERFORMANCE_TARGETS = {
        'prediction_accuracy': 0.85,
        'response_time_seconds': 300,
        'data_quality_score': 0.95,
        'system_uptime': 0.99
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR,
            os.path.dirname(cls.LOGGING_CONFIG['file'])
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        if model_type.lower() == 'lstm':
            return cls.LSTM_CONFIG
        elif model_type.lower() == 'arima':
            return cls.ARIMA_CONFIG
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_feature_columns(cls) -> list:
        """Get list of feature columns for models"""
        return [
            'temperature_celsius',
            'precipitation_mm',
            'extreme_weather_event',
            'drought_indicator',
            'humidity_percent',
            'gdp_growth_percent',
            'inflation_rate_percent',
            'trade_restrictions',
            'subsidies_active',
            'regulatory_changes',
            'food_safety_incidents',
            'political_stability_index',
            'trade_relations_index',
            'sanctions_active',
            'conflict_indicator',
            'supply_chain_disruption',
            'border_closure_days',
            'production_volume_tonnes',
            'transportation_cost_per_ton',
            'delivery_time_days',
            'inventory_level_tonnes',
            'quality_score_percent',
            'disruption_risk',
            'supplier_reliability_score'
        ]
    
    @classmethod
    def get_target_columns(cls) -> list:
        """Get list of target columns for forecasting"""
        return [
            'supply_chain_disruption',
            'production_volume_tonnes',
            'temperature_celsius',
            'precipitation_mm',
            'transportation_cost_per_ton'
        ]
    
    @classmethod
    def get_validation_schema(cls) -> Dict[str, Dict]:
        """Get data validation schema"""
        return {
            'date': {
                'required': True,
                'data_type': 'datetime64[ns]',
                'range_check': {
                    'min_date': '2020-01-01',
                    'max_date': '2024-12-31'
                }
            },
            'temperature_celsius': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': -50,
                    'max': 60
                }
            },
            'precipitation_mm': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 1000
                }
            },
            'production_volume_tonnes': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 10000
                }
            },
            'quality_score_percent': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 100
                }
            }
        }

# Create directories on import
Config.create_directories() 