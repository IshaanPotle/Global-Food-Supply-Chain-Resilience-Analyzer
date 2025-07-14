"""
Data Validation Framework for Food Supply Chain Resilience Analyzer
Ensures data quality and model reliability
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation framework"""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        self.validation_results = {}
    
    def _define_validation_rules(self) -> Dict[str, Dict]:
        """Define validation rules for different data types"""
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
                },
                'null_allowed': False
            },
            'precipitation_mm': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 1000
                },
                'null_allowed': False
            },
            'production_volume_tonnes': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 10000
                },
                'null_allowed': False
            },
            'quality_score_percent': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 100
                },
                'null_allowed': False
            },
            'political_stability_index': {
                'required': True,
                'data_type': 'float64',
                'range_check': {
                    'min': 0,
                    'max': 100
                },
                'null_allowed': False
            }
        }
    
    def validate_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema and structure"""
        logger.info("Validating data schema...")
        
        validation_result = {
            'schema_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'details': {}
        }
        
        # Check required columns
        required_columns = [col for col, rules in self.validation_rules.items() if rules.get('required', False)]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_result['schema_valid'] = False
            validation_result['missing_columns'] = missing_columns
            logger.warning(f"Missing required columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = [col for col in data.columns if col not in self.validation_rules]
        if extra_columns:
            validation_result['extra_columns'] = extra_columns
            logger.info(f"Extra columns found: {extra_columns}")
        
        # Check data types
        for col, rules in self.validation_rules.items():
            if col in data.columns:
                expected_type = rules.get('data_type')
                actual_type = str(data[col].dtype)
                
                if expected_type and actual_type != expected_type:
                    validation_result['type_mismatches'].append({
                        'column': col,
                        'expected': expected_type,
                        'actual': actual_type
                    })
                    validation_result['schema_valid'] = False
        
        self.validation_results['schema'] = validation_result
        return validation_result
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics"""
        logger.info("Validating data quality...")
        
        quality_result = {
            'quality_score': 0.0,
            'missing_values': {},
            'duplicates': 0,
            'outliers': {},
            'range_violations': {},
            'details': {}
        }
        
        # Check missing values
        missing_values = data.isnull().sum()
        quality_result['missing_values'] = missing_values.to_dict()
        
        # Check duplicates
        duplicates = data.duplicated().sum()
        quality_result['duplicates'] = int(duplicates)
        
        # Check range violations
        for col, rules in self.validation_rules.items():
            if col in data.columns and 'range_check' in rules:
                range_check = rules['range_check']
                
                if 'min' in range_check and 'max' in range_check:
                    min_val = range_check['min']
                    max_val = range_check['max']
                    
                    violations = ((data[col] < min_val) | (data[col] > max_val)).sum()
                    if violations > 0:
                        quality_result['range_violations'][col] = int(violations)
        
        # Calculate quality score
        total_cells = len(data) * len(data.columns)
        missing_cells = sum(quality_result['missing_values'].values())
        duplicate_rows = quality_result['duplicates']
        range_violations = sum(quality_result['range_violations'].values())
        
        quality_score = 1.0 - (missing_cells + duplicate_rows + range_violations) / total_cells
        quality_result['quality_score'] = max(0.0, quality_score)
        
        self.validation_results['quality'] = quality_result
        return quality_result
    
    def validate_business_rules(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate business-specific rules"""
        logger.info("Validating business rules...")
        
        business_result = {
            'business_rules_valid': True,
            'violations': [],
            'details': {}
        }
        
        # Rule 1: Production volume should be positive when no disruption
        if 'production_volume_tonnes' in data.columns and 'supply_chain_disruption' in data.columns:
            violation_mask = (data['production_volume_tonnes'] <= 0) & (data['supply_chain_disruption'] == 0)
            violations = violation_mask.sum()
            if violations > 0:
                business_result['violations'].append({
                    'rule': 'Production volume should be positive when no disruption',
                    'violations': int(violations)
                })
        
        # Rule 2: Quality score should be high when supplier reliability is high
        if 'quality_score_percent' in data.columns and 'supplier_reliability_score' in data.columns:
            violation_mask = (data['quality_score_percent'] < 70) & (data['supplier_reliability_score'] > 90)
            violations = violation_mask.sum()
            if violations > 0:
                business_result['violations'].append({
                    'rule': 'Quality score should be high when supplier reliability is high',
                    'violations': int(violations)
                })
        
        # Rule 3: Political stability should be consistent (no sudden drops)
        if 'political_stability_index' in data.columns:
            stability_diff = data['political_stability_index'].diff().abs()
            sudden_drops = (stability_diff > 20).sum()
            if sudden_drops > 0:
                business_result['violations'].append({
                    'rule': 'Political stability should not have sudden drops',
                    'violations': int(sudden_drops)
                })
        
        if business_result['violations']:
            business_result['business_rules_valid'] = False
        
        self.validation_results['business_rules'] = business_result
        return business_result
    
    def validate_time_series_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate time series data consistency"""
        logger.info("Validating time series consistency...")
        
        ts_result = {
            'ts_consistent': True,
            'gaps': [],
            'duplicates': [],
            'ordering': True,
            'details': {}
        }
        
        if 'date' not in data.columns:
            ts_result['ts_consistent'] = False
            ts_result['details']['error'] = 'Date column not found'
            return ts_result
        
        # Check for date ordering
        if not data['date'].is_monotonic_increasing:
            ts_result['ordering'] = False
            ts_result['ts_consistent'] = False
        
        # Check for duplicate dates
        duplicate_dates = data['date'].duplicated().sum()
        if duplicate_dates > 0:
            ts_result['duplicates'] = int(duplicate_dates)
            ts_result['ts_consistent'] = False
        
        # Check for gaps in time series
        expected_dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
        actual_dates = data['date'].unique()
        missing_dates = set(expected_dates) - set(actual_dates)
        
        if missing_dates:
            ts_result['gaps'] = len(missing_dates)
            ts_result['ts_consistent'] = False
        
        self.validation_results['time_series'] = ts_result
        return ts_result
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_valid': True,
            'summary': {
                'schema_valid': False,
                'quality_score': 0.0,
                'business_rules_valid': False,
                'time_series_consistent': False
            },
            'details': self.validation_results,
            'recommendations': []
        }
        
        # Update summary
        if 'schema' in self.validation_results:
            report['summary']['schema_valid'] = self.validation_results['schema']['schema_valid']
        
        if 'quality' in self.validation_results:
            report['summary']['quality_score'] = self.validation_results['quality']['quality_score']
        
        if 'business_rules' in self.validation_results:
            report['summary']['business_rules_valid'] = self.validation_results['business_rules']['business_rules_valid']
        
        if 'time_series' in self.validation_results:
            report['summary']['time_series_consistent'] = self.validation_results['time_series']['ts_consistent']
        
        # Determine overall validity
        report['overall_valid'] = all([
            report['summary']['schema_valid'],
            report['summary']['quality_score'] > 0.8,
            report['summary']['business_rules_valid'],
            report['summary']['time_series_consistent']
        ])
        
        # Generate recommendations
        if not report['summary']['schema_valid']:
            report['recommendations'].append("Fix schema violations before proceeding")
        
        if report['summary']['quality_score'] < 0.9:
            report['recommendations'].append("Data quality needs improvement")
        
        if not report['summary']['business_rules_valid']:
            report['recommendations'].append("Review and fix business rule violations")
        
        if not report['summary']['time_series_consistent']:
            report['recommendations'].append("Fix time series consistency issues")
        
        return report
    
    def validate_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        logger.info("Starting complete data validation...")
        
        # Reset validation results
        self.validation_results = {}
        
        # Run all validation checks
        self.validate_schema(data)
        self.validate_data_quality(data)
        self.validate_business_rules(data)
        self.validate_time_series_consistency(data)
        
        # Generate final report
        report = self.generate_validation_report()
        
        logger.info(f"Validation completed. Overall valid: {report['overall_valid']}")
        logger.info(f"Quality score: {report['summary']['quality_score']:.2f}")
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], filename: str = 'validation_report.json'):
        """Save validation report to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")

if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Load sample data for testing
    try:
        data = pd.read_csv('data/food_supply_chain_data.csv')
        data['date'] = pd.to_datetime(data['date'])
        
        # Run validation
        report = validator.validate_dataset(data)
        
        # Save report
        validator.save_validation_report(report)
        
        print("\nValidation Report Summary:")
        print(f"Overall Valid: {report['overall_valid']}")
        print(f"Quality Score: {report['summary']['quality_score']:.2f}")
        print(f"Schema Valid: {report['summary']['schema_valid']}")
        print(f"Business Rules Valid: {report['summary']['business_rules_valid']}")
        print(f"Time Series Consistent: {report['summary']['time_series_consistent']}")
        
    except FileNotFoundError:
        print("Data file not found. Please run the ETL pipeline first.") 