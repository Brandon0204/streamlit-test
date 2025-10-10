# House Price Prediction System

A comprehensive machine learning system for analyzing and forecasting New Zealand house price growth using multiple data sources and modeling approaches.

**Live Demo**: [https://app-test-qxq5b9dukmh7yw6xuyufxc.streamlit.app/](https://app-test-qxq5b9dukmh7yw6xuyufxc.streamlit.app/)

## Project Overview

This system ingests housing market data, performs feature engineering, trains multiple forecasting models, and provides an interactive dashboard for data exploration and model experimentation. The entire data infrastructure is powered by Supabase, providing seamless database operations and file storage.

## Tech Stack

**Core**: Python, Pandas, NumPy  
**ML**: XGBoost, CatBoost, statsmodels, scikit-learn  
**Database & Storage**: Supabase (PostgreSQL + Object Storage)  
**Visualization**: Plotly, Streamlit  
**Data Processing**: openpyxl, papaparse

### About Supabase

[Supabase](https://supabase.com/) is an open-source Firebase alternative that provides:
- **PostgreSQL Database**: Robust relational database with real-time capabilities
- **Object Storage**: S3-compatible storage for files (Excel, CSV)
- **Auto-generated APIs**: RESTful APIs for database operations
- **Row Level Security**: Built-in authentication and authorization

This project leverages Supabase for:
- Storing raw data files in cloud storage buckets
- Managing structured data in PostgreSQL tables (`clean_house`, `feature_house`, `house_metrics`)
- Providing unified access through Python SDK for ETL and ML workflows

## Key Features

### 1. Data Pipeline
- **Automated ETL**: Extracts data from Supabase storage (Excel/CSV files)
- **Data Sources**: 
  - House sales, HPI (House Price Index), housing stock
  - Official Cash Rate (OCR)
  - Consumer Price Index (CPI)
  - GDP data
- **Storage**: Clean data stored in `clean_house` table, engineered features in `feature_house` table

### 2. Feature Engineering
- **Temporal Features**: Lags (1-4 quarters, 4 years), rolling means (1yr, 4yr, 10yr)
- **Derived Features**: Growth rates, differences, ratios
- **Policy Indicators**: COVID lockdown period, reopening/supply constraints flags
- **53 total features** created from 8 base variables

### 3. Machine Learning Models
- **ETS (Exponential Smoothing)**: Time series forecasting with trend/seasonal components
- **XGBoost**: Gradient boosting with custom feature selection
- **CatBoost**: Gradient boosting optimized for categorical features
- All models support train/test splitting and handle missing data (drop or impute)

### 4. Interactive Dashboard (Streamlit)
- **Data Viewer**: Browse and download clean/feature datasets
- **EDA Tools**: 
  - Feature importance analysis (F-statistic method)
  - Target distribution visualization
  - Correlation heatmaps
  - Missing value reports
  - Summary statistics
- **Model Training**: Configure and train models with custom parameters
- **Performance Comparison**: Side-by-side metrics and forecast charts
- **Pipeline Controls**: Trigger data refresh and feature engineering

## Project Structure

```
├── ingest_h.py              # ETL pipeline (raw data → clean_house)
├── feature_engineering.py   # Feature creation (clean_house → feature_house)
├── trainer.py               # Base class for model training
├── ets_trainer.py           # ETS model implementation
├── xgboost_trainer.py       # XGBoost model implementation
├── catboost_trainer.py      # CatBoost model implementation
├── test.py                  # Streamlit dashboard (main interface)
└── requirements.txt         # Python dependencies
```

## Database Schema

### Tables
- **clean_house**: Raw quarterly data (house_sales, hpi, house_stock, residential_investment, ocr, cpi, gdp)
- **feature_house**: Engineered features with 53 columns derived from clean data
- **house_metrics**: Model performance metrics and parameters

### Storage Bucket
- **project3bucket**: Contains source files in `house_row_files/` directory

## Usage

### Setup
```bash
pip install -r requirements.txt
```

Configure Supabase credentials in `.streamlit/secrets.toml`:
```toml
[supabase]
url = "your-supabase-url"
key = "your-anon-key"
service_role_key = "your-service-role-key"  # optional
```

### Run Dashboard
```bash
streamlit run test.py
```

### Programmatic Use
```python
from ingest_h import HousePriceETL
from feature_engineering import HouseFeatureEngineering
from xgboost_trainer import run_experiment

# Refresh data
etl = HousePriceETL()
raw = etl.extract()
clean = etl.transform(raw)
etl.load(clean)

# Generate features
fe = HouseFeatureEngineering()
df = fe.extract()
df_features = fe.transform(df)
fe.load(df_features)

# Train model
result = run_experiment(
    best_params={'n_estimators': 400, 'max_depth': 4},
    feature_list=['hpi_growth_lag1', 'house_sales_lag1'],
    test_size=0.2
)
print(result['metrics'])
```

## Model Outputs

All trained models automatically save metrics to `house_metrics` table:
- Train/test RMSE, MSE, MAE, R²
- Model parameters (JSON)
- Timestamp

## Data Requirements

Source files must be in Supabase storage bucket (`project3bucket` by default):
- `house_row_files/hm10.xlsx` - Housing data
- `house_row_files/OCRmodify.csv` - OCR data
- `house_row_files/consumers-price-index-*.csv` - CPI data
- `house_row_files/gross-domestic-product-*.csv` - GDP data

## Techniques Employed

- **Time Series Analysis**: Temporal feature engineering, train/test splitting
- **Missing Data Handling**: Forward fill, median imputation, row dropping
- **Feature Selection**: Statistical significance testing (F-statistic), correlation analysis
- **Model Evaluation**: Walk-forward validation, multiple metrics
- **Ensemble Methods**: Gradient boosting (XGBoost, CatBoost)
- **Classical Forecasting**: Exponential smoothing with seasonality

## Target Variable

**HPI Growth**: Quarterly percentage change in House Price Index  
Predicted using lagged values, economic indicators, and temporal patterns.

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Configure Supabase credentials
# Create .streamlit/secrets.toml with your credentials

# Run application
streamlit run test.py
```

## License

This project is for educational and research purposes.
