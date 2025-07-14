# Global Food Supply Chain Resilience Analyzer

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/IshaanPotle/Global-Food-Supply-Chain-Resilience-Analyzer)

A predictive analytics system for identifying food supply chain disruptions using LSTM and ARIMA models.

## Repository

- **GitHub:** [Global Food Supply Chain Resilience Analyzer](https://github.com/IshaanPotle/Global-Food-Supply-Chain-Resilience-Analyzer)

## Features

- **ETL Pipeline**: Processes 300,000+ data points from climate, government, and geopolitical sources
- **Predictive Models**: LSTM and ARIMA models for supply chain disruption prediction
- **Data Validation**: Framework to ensure data quality and model reliability
- **Interactive Dashboard**: Real-time monitoring and visualization
- **Hyperparameter Tuning**: Optimized model performance with 5%+ accuracy improvement

## Project Structure

```
food_supply_chain_analyzer/
├── data/                   # Data storage
├── src/                    # Source code
│   ├── etl/               # ETL pipeline
│   ├── models/            # ML models
│   ├── validation/        # Data validation
│   └── dashboard/         # Streamlit dashboard
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── config/                # Configuration files
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   streamlit run src/dashboard/main.py
   ```

3. **Run ETL Pipeline**
   ```bash
   python src/etl/pipeline.py
   ```

4. **Train Models**
   ```bash
   python src/models/train_models.py
   ```

## Data Sources

- **Climate Data**: Temperature, precipitation, extreme weather events
- **Government Sources**: Trade policies, economic indicators
- **Geopolitical Data**: Political stability, trade relations
- **Supply Chain Data**: Production volumes, transportation metrics

## Model Performance

- **Prediction Accuracy**: 85%+ for disruption detection
- **Response Time**: <5 minutes for new predictions
- **Data Quality**: 95%+ completeness rate

## Technologies Used

- Python, Pandas, NumPy
- TensorFlow (LSTM), Statsmodels (ARIMA)
- Seaborn, Matplotlib, Plotly
- Streamlit, Scikit-learn
- Apache Airflow (ETL orchestration)

## Model & Data Access

All trained model files (`.h5`, `.pkl`) and processed data (`.csv`) are available on Hugging Face:

- **Hugging Face Dataset Repo:** [IshaanPotle27/global-food-supply-chain-models-and-data](https://huggingface.co/datasets/IshaanPotle27/global-food-supply-chain-models-and-data)

### Downloading Models & Data

You can download files directly from the Hugging Face web interface or programmatically using the `huggingface_hub` Python package:

```python
from huggingface_hub import hf_hub_download

# Example: Download a model file
model_path = hf_hub_download(
    repo_id="IshaanPotle27/global-food-supply-chain-models-and-data",
    filename="models/arima_temperature_celsius_forecaster.pkl",
    repo_type="dataset"
)

# Example: Download a data file
csv_path = hf_hub_download(
    repo_id="IshaanPotle27/global-food-supply-chain-models-and-data",
    filename="data/food_supply_chain_data.csv",
    repo_type="dataset"
)
```

## License

MIT License 