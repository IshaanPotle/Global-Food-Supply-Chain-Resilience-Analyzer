"""
Unit tests for machine learning models
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import LSTMDisruptionPredictor
from models.arima_model import ARIMAForecaster

class TestLSTMModel:
    """Test cases for LSTM model"""
    
    def setup_method(self):
        """Setup test data"""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'temperature_celsius': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 2, len(dates)),
            'precipitation_mm': np.maximum(0, 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 + np.pi/2) + np.random.normal(0, 10, len(dates))),
            'extreme_weather_event': np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05]),
            'gdp_growth_percent': 2.5 + np.random.normal(0, 0.5, len(dates)),
            'inflation_rate_percent': 2.0 + np.random.normal(0, 0.3, len(dates)),
            'trade_restrictions': np.random.choice([0, 1], size=len(dates), p=[0.85, 0.15]),
            'political_stability_index': 70 + np.random.normal(0, 10, len(dates)),
            'trade_relations_index': 75 + np.random.normal(0, 8, len(dates)),
            'sanctions_active': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1]),
            'production_volume_tonnes': 1000 + np.random.normal(0, 100, len(dates)),
            'transportation_cost_per_ton': 50 + np.random.normal(0, 10, len(dates)),
            'inventory_level_tonnes': 500 + np.random.normal(0, 50, len(dates)),
            'quality_score_percent': 85 + np.random.normal(0, 5, len(dates)),
            'supplier_reliability_score': 80 + np.random.normal(0, 8, len(dates)),
            'supply_chain_disruption': np.random.choice([0, 1], size=len(dates), p=[0.92, 0.08])
        })
        
        # Ensure positive values
        self.test_data['production_volume_tonnes'] = np.maximum(0, self.test_data['production_volume_tonnes'])
        self.test_data['inventory_level_tonnes'] = np.maximum(0, self.test_data['inventory_level_tonnes'])
        self.test_data['quality_score_percent'] = np.clip(self.test_data['quality_score_percent'], 0, 100)
        self.test_data['supplier_reliability_score'] = np.clip(self.test_data['supplier_reliability_score'], 0, 100)
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        
        assert lstm_model.sequence_length == 30
        assert lstm_model.prediction_horizon == 7
        assert lstm_model.model is None
        assert lstm_model.scaler is not None
    
    def test_data_preparation(self):
        """Test data preparation for LSTM"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        X, y = lstm_model.prepare_data(self.test_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[1] == 30  # sequence_length
        assert X.shape[2] > 0  # number of features
        assert y.shape[1] == 7  # prediction_horizon
        
        # Check that X and y have the same number of samples
        assert len(X) == len(y)
    
    def test_model_building(self):
        """Test LSTM model building"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        X, y = lstm_model.prepare_data(self.test_data)
        
        model = lstm_model.build_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001)
        
        assert model is not None
        assert hasattr(model, 'layers')
        assert len(model.layers) > 0
        
        # Check output shape
        expected_output_shape = (None, 7)  # (batch_size, prediction_horizon)
        assert model.output_shape == expected_output_shape
    
    def test_model_training(self):
        """Test LSTM model training"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        X, y = lstm_model.prepare_data(self.test_data)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model with fewer epochs for testing
        training_results = lstm_model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=5, batch_size=16, patience=3
        )
        
        assert isinstance(training_results, dict)
        assert 'val_loss' in training_results
        assert 'val_accuracy' in training_results
        assert 'val_precision' in training_results
        assert 'val_recall' in training_results
        assert 'epochs_trained' in training_results
        assert 'training_history' in training_results
        
        # Check that model was trained
        assert lstm_model.model is not None
        assert lstm_model.history is not None
    
    def test_model_prediction(self):
        """Test LSTM model prediction"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        X, y = lstm_model.prepare_data(self.test_data)
        
        # Train model first
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        lstm_model.train_model(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
        
        # Make predictions
        predictions = lstm_model.predict(X_val[:5])  # Predict on 5 samples
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 5
        assert predictions.shape[1] == 7  # prediction_horizon
        assert np.all((predictions >= 0) & (predictions <= 1))  # Probabilities should be between 0 and 1
    
    def test_model_evaluation(self):
        """Test LSTM model evaluation"""
        lstm_model = LSTMDisruptionPredictor(sequence_length=30, prediction_horizon=7)
        X, y = lstm_model.prepare_data(self.test_data)
        
        # Train model first
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        val_split_idx = int(0.8 * len(X_train))
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        lstm_model.train_model(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
        
        # Evaluate model
        evaluation_results = lstm_model.evaluate_model(X_test, y_test)
        
        assert isinstance(evaluation_results, dict)
        assert 'mse' in evaluation_results
        assert 'mae' in evaluation_results
        assert 'horizon_metrics' in evaluation_results
        assert 'overall_accuracy' in evaluation_results
        assert 'overall_precision' in evaluation_results
        assert 'overall_recall' in evaluation_results
        
        # Check that metrics are reasonable
        assert evaluation_results['mse'] >= 0
        assert evaluation_results['mae'] >= 0
        assert 0 <= evaluation_results['overall_accuracy'] <= 1
        assert 0 <= evaluation_results['overall_precision'] <= 1
        assert 0 <= evaluation_results['overall_recall'] <= 1

class TestARIMAModel:
    """Test cases for ARIMA model"""
    
    def setup_method(self):
        """Setup test data"""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        # Create time series with trend and seasonality
        trend = np.linspace(1000, 1200, len(dates))
        seasonality = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 20, len(dates))
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'production_volume_tonnes': trend + seasonality + noise
        })
        
        # Ensure positive values
        self.test_data['production_volume_tonnes'] = np.maximum(0, self.test_data['production_volume_tonnes'])
    
    def test_arima_initialization(self):
        """Test ARIMA model initialization"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        
        assert arima_model.target_column == 'production_volume_tonnes'
        assert arima_model.model is None
        assert arima_model.fitted_model is None
        assert arima_model.order is None
    
    def test_stationarity_check(self):
        """Test stationarity checking"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        series = self.test_data['production_volume_tonnes']
        
        stationarity_result = arima_model.check_stationarity(series)
        
        assert isinstance(stationarity_result, dict)
        assert 'is_stationary' in stationarity_result
        assert 'adf_statistic' in stationarity_result
        assert 'adf_pvalue' in stationarity_result
        assert 'kpss_statistic' in stationarity_result
        assert 'kpss_pvalue' in stationarity_result
        assert isinstance(stationarity_result['is_stationary'], bool)
    
    def test_making_stationary(self):
        """Test making series stationary"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        series = self.test_data['production_volume_tonnes']
        
        stationary_series, diff_order = arima_model.make_stationary(series)
        
        assert isinstance(stationary_series, pd.Series)
        assert isinstance(diff_order, int)
        assert diff_order >= 0
        assert len(stationary_series) <= len(series)
    
    def test_optimal_order_finding(self):
        """Test finding optimal ARIMA order"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        series = self.test_data['production_volume_tonnes']
        
        # Make series stationary first
        stationary_series, _ = arima_model.make_stationary(series)
        
        optimal_order = arima_model.find_optimal_order(stationary_series, max_p=3, max_d=1, max_q=3)
        
        assert isinstance(optimal_order, tuple)
        assert len(optimal_order) == 3
        assert all(isinstance(x, int) for x in optimal_order)
        assert all(x >= 0 for x in optimal_order)
    
    def test_model_fitting(self):
        """Test ARIMA model fitting"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        
        fitting_results = arima_model.fit_model(self.test_data, auto_order=True)
        
        assert isinstance(fitting_results, dict)
        assert 'order' in fitting_results
        assert 'aic' in fitting_results
        assert 'bic' in fitting_results
        assert 'hqic' in fitting_results
        assert 'is_stationary' in fitting_results
        assert 'differencing_order' in fitting_results
        assert 'model_summary' in fitting_results
        
        # Check that model was fitted
        assert arima_model.fitted_model is not None
        assert arima_model.order is not None
    
    def test_forecasting(self):
        """Test ARIMA forecasting"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        
        # Fit model first
        arima_model.fit_model(self.test_data, auto_order=True)
        
        # Make forecast
        forecast, lower_ci, upper_ci = arima_model.forecast(steps=30)
        
        assert isinstance(forecast, np.ndarray)
        assert isinstance(lower_ci, np.ndarray)
        assert isinstance(upper_ci, np.ndarray)
        assert len(forecast) == 30
        assert len(lower_ci) == 30
        assert len(upper_ci) == 30
        
        # Check that confidence intervals make sense
        assert np.all(lower_ci <= forecast)
        assert np.all(forecast <= upper_ci)
    
    def test_model_evaluation(self):
        """Test ARIMA model evaluation"""
        arima_model = ARIMAForecaster(target_column='production_volume_tonnes')
        
        # Fit model first
        arima_model.fit_model(self.test_data, auto_order=True)
        
        # Make forecast
        forecast, _, _ = arima_model.forecast(steps=30)
        
        # Use last 30 days as test set
        test_data = self.test_data['production_volume_tonnes'].tail(30)
        
        evaluation_results = arima_model.evaluate_forecast(test_data.values, forecast)
        
        assert isinstance(evaluation_results, dict)
        assert 'mse' in evaluation_results
        assert 'rmse' in evaluation_results
        assert 'mae' in evaluation_results
        assert 'mape' in evaluation_results
        
        # Check that metrics are reasonable
        assert evaluation_results['mse'] >= 0
        assert evaluation_results['rmse'] >= 0
        assert evaluation_results['mae'] >= 0
        assert evaluation_results['mape'] >= 0

if __name__ == "__main__":
    pytest.main([__file__]) 