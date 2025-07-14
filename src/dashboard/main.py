"""
Streamlit Dashboard for Food Supply Chain Resilience Analyzer
Real-time monitoring and interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Food Supply Chain Resilience Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    """Main dashboard application"""
    
    def __init__(self):
        self.data = None
        self.models_loaded = False
        self.predictions = None
        
    def load_data(self):
        """Load data for the dashboard"""
        try:
            # Load processed data
            if os.path.exists('data/feature_engineered_data.csv'):
                self.data = pd.read_csv('data/feature_engineered_data.csv')
                self.data['date'] = pd.to_datetime(self.data['date'])
                return True
            else:
                st.error("Data file not found. Please run the ETL pipeline first.")
                return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def load_models(self):
        """Load trained models"""
        try:
            # Check if models exist
            lstm_exists = os.path.exists('models/lstm_disruption_predictor.h5')
            arima_exists = os.path.exists('models/arima_production_volume_tonnes_forecaster.pkl')
            
            if lstm_exists and arima_exists:
                self.models_loaded = True
                return True
            else:
                st.warning("Trained models not found. Please run the model training first.")
                return False
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def generate_dummy_predictions(self):
        """Generate dummy predictions for demonstration"""
        if self.data is None:
            return None
        
        # Generate dummy disruption predictions
        last_date = self.data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        # Simulate predictions with some randomness
        np.random.seed(42)
        disruption_prob = np.random.beta(2, 8, len(future_dates))  # Low probability of disruption
        disruption_pred = (disruption_prob > 0.3).astype(int)
        
        # Simulate production forecasts
        base_production = self.data['production_volume_tonnes'].iloc[-1]
        production_forecast = base_production + np.random.normal(0, 50, len(future_dates))
        production_forecast = np.maximum(0, production_forecast)
        
        self.predictions = {
            'dates': future_dates,
            'disruption_probability': disruption_prob,
            'disruption_prediction': disruption_pred,
            'production_forecast': production_forecast
        }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🌾 Food Supply Chain Resilience Analyzer</h1>', unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", f"{len(self.data):,}" if self.data is not None else "N/A")
        
        with col2:
            st.metric("Date Range", f"{self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}" if self.data is not None else "N/A")
        
        with col3:
            st.metric("Models Status", "✅ Loaded" if self.models_loaded else "❌ Not Loaded")
        
        with col4:
            st.metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    def render_overview_metrics(self):
        """Render overview metrics"""
        st.subheader("📊 Overview Metrics")
        
        if self.data is None:
            st.warning("No data available")
            return
        
        # Calculate key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_production = self.data['production_volume_tonnes'].mean()
            st.metric("Avg Production (tonnes)", f"{avg_production:.0f}")
        
        with col2:
            avg_temp = self.data['temperature_celsius'].mean()
            st.metric("Avg Temperature (°C)", f"{avg_temp:.1f}")
        
        with col3:
            disruption_rate = self.data['supply_chain_disruption'].mean() * 100
            st.metric("Disruption Rate (%)", f"{disruption_rate:.1f}")
        
        with col4:
            avg_quality = self.data['quality_score_percent'].mean()
            st.metric("Avg Quality Score (%)", f"{avg_quality:.1f}")
    
    def render_time_series_charts(self):
        """Render time series charts"""
        st.subheader("📈 Time Series Analysis")
        
        if self.data is None:
            st.warning("No data available")
            return
        
        # Create tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs(["Production", "Climate", "Economic", "Geopolitical"])
        
        with tab1:
            # Production metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Production Volume', 'Transportation Cost'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['production_volume_tonnes'],
                          mode='lines', name='Production Volume'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['transportation_cost_per_ton'],
                          mode='lines', name='Transportation Cost'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Production Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Climate metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature', 'Precipitation'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['temperature_celsius'],
                          mode='lines', name='Temperature'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['precipitation_mm'],
                          mode='lines', name='Precipitation'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Climate Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Economic metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('GDP Growth', 'Inflation Rate'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['gdp_growth_percent'],
                          mode='lines', name='GDP Growth'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['inflation_rate_percent'],
                          mode='lines', name='Inflation Rate'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Economic Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Geopolitical metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Political Stability', 'Trade Relations'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['political_stability_index'],
                          mode='lines', name='Political Stability'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.data['date'], y=self.data['trade_relations_index'],
                          mode='lines', name='Trade Relations'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Geopolitical Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis"""
        st.subheader("🔗 Correlation Analysis")
        
        if self.data is None:
            st.warning("No data available")
            return
        
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_data = self.data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    def render_predictions(self):
        """Render prediction results"""
        st.subheader("🔮 Predictive Analytics")
        
        if self.predictions is None:
            self.generate_dummy_predictions()
        
        if self.predictions is None:
            st.warning("No predictions available")
            return
        
        # Create tabs for different predictions
        tab1, tab2 = st.tabs(["Disruption Risk", "Production Forecast"])
        
        with tab1:
            # Disruption risk prediction
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.predictions['dates'],
                y=self.predictions['disruption_probability'],
                mode='lines+markers',
                name='Disruption Probability',
                line=dict(color='red', width=2)
            ))
            
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                         annotation_text="Risk Threshold")
            
            fig.update_layout(
                title="Supply Chain Disruption Risk Forecast (30 Days)",
                xaxis_title="Date",
                yaxis_title="Disruption Probability",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk alerts
            high_risk_dates = self.predictions['dates'][self.predictions['disruption_probability'] > 0.5]
            if len(high_risk_dates) > 0:
                st.markdown('<div class="metric-card alert-high">', unsafe_allow_html=True)
                st.warning(f"⚠️ High disruption risk detected on {len(high_risk_dates)} dates")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Production forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=self.data['date'],
                y=self.data['production_volume_tonnes'],
                mode='lines',
                name='Historical Production',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=self.predictions['dates'],
                y=self.predictions['production_forecast'],
                mode='lines+markers',
                name='Production Forecast',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Production Volume Forecast (30 Days)",
                xaxis_title="Date",
                yaxis_title="Production Volume (tonnes)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_assessment(self):
        """Render risk assessment dashboard"""
        st.subheader("⚠️ Risk Assessment")
        
        if self.data is None:
            st.warning("No data available")
            return
        
        # Calculate current risk factors
        latest_data = self.data.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Climate risk
            temp_risk = "High" if latest_data['temperature_celsius'] > 35 else "Medium" if latest_data['temperature_celsius'] > 25 else "Low"
            precip_risk = "High" if latest_data['precipitation_mm'] > 100 else "Medium" if latest_data['precipitation_mm'] > 50 else "Low"
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Climate Risk**")
            st.write(f"Temperature: {temp_risk}")
            st.write(f"Precipitation: {precip_risk}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Economic risk
            gdp_risk = "High" if latest_data['gdp_growth_percent'] < 1 else "Medium" if latest_data['gdp_growth_percent'] < 2 else "Low"
            inflation_risk = "High" if latest_data['inflation_rate_percent'] > 5 else "Medium" if latest_data['inflation_rate_percent'] > 3 else "Low"
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Economic Risk**")
            st.write(f"GDP Growth: {gdp_risk}")
            st.write(f"Inflation: {inflation_risk}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # Geopolitical risk
            stability_risk = "High" if latest_data['political_stability_index'] < 50 else "Medium" if latest_data['political_stability_index'] < 70 else "Low"
            trade_risk = "High" if latest_data['trade_relations_index'] < 50 else "Medium" if latest_data['trade_relations_index'] < 70 else "Low"
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Geopolitical Risk**")
            st.write(f"Political Stability: {stability_risk}")
            st.write(f"Trade Relations: {trade_risk}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Date range selector
        if self.data is not None:
            min_date = self.data['date'].min()
            max_date = self.data['date'].max()
            
            st.sidebar.subheader("Date Range")
            start_date = st.sidebar.date_input("Start Date", min_date)
            end_date = st.sidebar.date_input("End Date", max_date)
        
        # Metric selector
        st.sidebar.subheader("Key Metrics")
        show_production = st.sidebar.checkbox("Production Metrics", value=True)
        show_climate = st.sidebar.checkbox("Climate Metrics", value=True)
        show_economic = st.sidebar.checkbox("Economic Metrics", value=True)
        show_geopolitical = st.sidebar.checkbox("Geopolitical Metrics", value=True)
        
        # Prediction settings
        st.sidebar.subheader("Predictions")
        forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        return {
            'start_date': start_date if self.data is not None else None,
            'end_date': end_date if self.data is not None else None,
            'show_production': show_production,
            'show_climate': show_climate,
            'show_economic': show_economic,
            'show_geopolitical': show_geopolitical,
            'forecast_days': forecast_days
        }
    
    def run(self):
        """Run the dashboard application"""
        # Load data and models
        data_loaded = self.load_data()
        models_loaded = self.load_models()
        
        if not data_loaded:
            st.error("Failed to load data. Please ensure the ETL pipeline has been run.")
            return
        
        # Render sidebar
        controls = self.render_sidebar()
        
        # Render main content
        self.render_header()
        
        # Overview metrics
        self.render_overview_metrics()
        
        # Time series charts
        self.render_time_series_charts()
        
        # Correlation analysis
        self.render_correlation_analysis()
        
        # Predictions
        self.render_predictions()
        
        # Risk assessment
        self.render_risk_assessment()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>🌾 Food Supply Chain Resilience Analyzer | Built with Streamlit, TensorFlow, and Statsmodels</p>
                <p>Last updated: {}</p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main() 