"""
Data Generator for Food Supply Chain Resilience Analyzer
Simulates real-world data from climate, government, and geopolitical sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class FoodSupplyChainDataGenerator:
    """Generates synthetic data for food supply chain analysis"""
    
    def __init__(self, start_date='2020-01-01', end_date='2024-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
    def generate_climate_data(self):
        """Generate climate data with seasonal patterns and extreme events"""
        np.random.seed(42)
        
        # Base climate patterns
        base_temp = 15 + 10 * np.sin(2 * np.pi * np.arange(len(self.date_range)) / 365.25)
        base_precip = 50 + 30 * np.sin(2 * np.pi * np.arange(len(self.date_range)) / 365.25 + np.pi/2)
        
        # Add noise and extreme events
        temperature = base_temp + np.random.normal(0, 3, len(self.date_range))
        precipitation = np.maximum(0, base_precip + np.random.normal(0, 10, len(self.date_range)))
        
        # Add extreme weather events (droughts, floods)
        extreme_events = np.random.choice([0, 1], size=len(self.date_range), p=[0.95, 0.05])
        drought_periods = np.random.choice([0, 1], size=len(self.date_range), p=[0.98, 0.02])
        
        climate_data = pd.DataFrame({
            'date': self.date_range,
            'temperature_celsius': temperature,
            'precipitation_mm': precipitation,
            'extreme_weather_event': extreme_events,
            'drought_indicator': drought_periods,
            'humidity_percent': 60 + 20 * np.random.random(len(self.date_range))
        })
        
        return climate_data
    
    def generate_government_data(self):
        """Generate government policy and economic data"""
        np.random.seed(43)
        
        # Economic indicators
        gdp_growth = 2.5 + np.random.normal(0, 0.5, len(self.date_range))
        inflation_rate = 2.0 + np.random.normal(0, 0.3, len(self.date_range))
        
        # Trade policies (binary indicators)
        trade_restrictions = np.random.choice([0, 1], size=len(self.date_range), p=[0.85, 0.15])
        subsidies_active = np.random.choice([0, 1], size=len(self.date_range), p=[0.7, 0.3])
        
        # Regulatory changes
        regulatory_changes = np.random.choice([0, 1], size=len(self.date_range), p=[0.95, 0.05])
        
        gov_data = pd.DataFrame({
            'date': self.date_range,
            'gdp_growth_percent': gdp_growth,
            'inflation_rate_percent': inflation_rate,
            'trade_restrictions': trade_restrictions,
            'subsidies_active': subsidies_active,
            'regulatory_changes': regulatory_changes,
            'food_safety_incidents': np.random.poisson(0.1, len(self.date_range))
        })
        
        return gov_data
    
    def generate_geopolitical_data(self):
        """Generate geopolitical stability and trade relation data"""
        np.random.seed(44)
        
        # Political stability index (0-100)
        political_stability = 70 + np.random.normal(0, 10, len(self.date_range))
        political_stability = np.clip(political_stability, 0, 100)
        
        # Trade relations (0-100)
        trade_relations = 75 + np.random.normal(0, 8, len(self.date_range))
        trade_relations = np.clip(trade_relations, 0, 100)
        
        # Sanctions and conflicts
        sanctions_active = np.random.choice([0, 1], size=len(self.date_range), p=[0.9, 0.1])
        conflict_indicator = np.random.choice([0, 1], size=len(self.date_range), p=[0.95, 0.05])
        
        # Supply chain disruptions
        supply_chain_disruption = np.random.choice([0, 1], size=len(self.date_range), p=[0.92, 0.08])
        
        geo_data = pd.DataFrame({
            'date': self.date_range,
            'political_stability_index': political_stability,
            'trade_relations_index': trade_relations,
            'sanctions_active': sanctions_active,
            'conflict_indicator': conflict_indicator,
            'supply_chain_disruption': supply_chain_disruption,
            'border_closure_days': np.random.poisson(0.05, len(self.date_range))
        })
        
        return geo_data
    
    def generate_supply_chain_data(self):
        """Generate supply chain operational data"""
        np.random.seed(45)
        
        # Production metrics
        production_volume = 1000 + np.random.normal(0, 100, len(self.date_range))
        production_volume = np.maximum(0, production_volume)
        
        # Transportation metrics
        transportation_cost = 50 + np.random.normal(0, 10, len(self.date_range))
        delivery_time_days = 3 + np.random.exponential(1, len(self.date_range))
        
        # Inventory levels
        inventory_level = 500 + np.random.normal(0, 50, len(self.date_range))
        inventory_level = np.maximum(0, inventory_level)
        
        # Quality metrics
        quality_score = 85 + np.random.normal(0, 5, len(self.date_range))
        quality_score = np.clip(quality_score, 0, 100)
        
        # Disruption indicators
        disruption_risk = np.random.choice([0, 1], size=len(self.date_range), p=[0.88, 0.12])
        
        supply_data = pd.DataFrame({
            'date': self.date_range,
            'production_volume_tonnes': production_volume,
            'transportation_cost_per_ton': transportation_cost,
            'delivery_time_days': delivery_time_days,
            'inventory_level_tonnes': inventory_level,
            'quality_score_percent': quality_score,
            'disruption_risk': disruption_risk,
            'supplier_reliability_score': 80 + np.random.normal(0, 8, len(self.date_range))
        })
        
        return supply_data
    
    def generate_all_data(self):
        """Generate all datasets and merge them"""
        print("Generating climate data...")
        climate_data = self.generate_climate_data()
        
        print("Generating government data...")
        gov_data = self.generate_government_data()
        
        print("Generating geopolitical data...")
        geo_data = self.generate_geopolitical_data()
        
        print("Generating supply chain data...")
        supply_data = self.generate_supply_chain_data()
        
        # Merge all datasets
        print("Merging datasets...")
        merged_data = climate_data.merge(gov_data, on='date', how='inner')
        merged_data = merged_data.merge(geo_data, on='date', how='inner')
        merged_data = merged_data.merge(supply_data, on='date', how='inner')
        
        print(f"Generated {len(merged_data)} data points with {len(merged_data.columns)} features")
        return merged_data
    
    def save_data(self, data, filename='food_supply_chain_data.csv'):
        """Save generated data to CSV"""
        data.to_csv(f'data/{filename}', index=False)
        print(f"Data saved to data/{filename}")
        return filename

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    generator = FoodSupplyChainDataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data) 