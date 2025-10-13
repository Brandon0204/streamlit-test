# Project 3: Team Contributions

## Team Members
- **Xuanhui**
- **Ming**
- **Zhengyang**
- **Yue** (Infrastructure & Deployment Lead)

---

## Individual Contributions

### Xuanhui - Project Lead & Feature Engineering
**Primary Responsibilities:**
- **Project Initiation**: Selected the New Zealand house price prediction project and defined the project scope
- **External Feature Research**: Identified and integrated external economic indicators (CPI, GDP, OCR) to enhance prediction accuracy
- **Exploratory Data Analysis (EDA)**: Conducted comprehensive analysis to justify the inclusion of macroeconomic features and their relationship with house price growth
- **Model Development**: Implemented and tested tree-based models with extensive hyperparameter tuning
- **Model Optimization**: Performed systematic parameter search and optimization across multiple model architectures

**Key Deliverables:**
- Project proposal and scope definition
- External data source integration (CPI, GDP, OCR datasets)
- EDA notebooks demonstrating feature relationships
- Tree-based model implementations with hyperparameter optimization

---

### Ming - Modeling & Explainability (Time-Series CV & XAI)
**Primary Responsibilities:**
- **End-to-end Modeling (leakage-safe)**: Built a reproducible pipeline to predict `hpi` with strict anti-leakage design (lag-based features only for target-related signals; conservative handling of rolling stats that directly depend on same-period `hpi`).
- **Time-Series Cross-Validation**: Implemented `TimeSeriesSplit` with unified evaluation (RMSE / MAE / R²), fixed random seeds, and consistent preprocessing.
- **Model Development & Comparison**: Implemented and compared baseline Ridge vs. tree models including Random Forest, LightGBM, and XGBoost under the same CV protocol.
- **Hyperparameter Optimization**: Used Optuna to tune LightGBM with early stopping and pruners; logged trials and selected the best configuration for artifact export.
- **Model Explainability (XAI)**: Produced SHAP global/dep/local analyses and PDP/ICE visuals to interpret feature effects and non-linear patterns.

**Key Deliverables:**
- Reproducible notebook `Ming_Zhang_model.ipynb` covering data processing, CV, model comparison, tuning, and interpretation.
- Optuna study artifacts: `artifacts/optuna_trials_lgbm.csv`, `artifacts/plots/optuna_history_lgbm.png`.
- Evaluation metrics: `artifacts/metrics.json` with RMSE/MAE/R² across models.
- Serialized example model: `artifacts/best_lgbm.pkl` (best LightGBM from Optuna) and feature manifest `artifacts/feature_list.csv`.
- XAI figures in `artifacts/plots/`: SHAP summary (dot/bar), dependence plots, local explanation bar, PDP/ICE.

**Note**: Under strict time-series CV and leakage control, the Ridge baseline proved more stable than tree models on RMSE/MAE. Recommendation was to use Ridge as default for demo while retaining tree models (include LightGBM) for exploration and explanation in the Streamlit app.

---

### Zhengyang - Linear Model Specialist
**Primary Responsibilities:**
- **Linear Model Implementation**: Developed and optimized linear regression approaches including Ridge Regression
- **Regularization Analysis**: Explored L1/L2 regularization techniques for linear models
- **Model Comparison**: Compared linear approaches against tree-based methods
- **Feature Selection**: Conducted feature selection specific to linear modeling assumptions

**Key Deliverables:**
- Ridge Regression implementation with cross-validation
- Linear model performance evaluation and comparison
- Analysis of feature linearity and multicollinearity
- Documentation of linear vs. non-linear model trade-offs

---

### Yue - Infrastructure & Deployment Engineer
**Primary Responsibilities:**
- **Infrastructure Development**: Designed and implemented the complete data engineering pipeline
- **ETL Pipeline**: Built automated Extract-Transform-Load system (`ingest_h.py`) for data ingestion from multiple sources
- **Database Architecture**: Designed and implemented Supabase database schema (tables: `clean_house`, `feature_house`, `house_metrics`)
- **Feature Engineering Pipeline**: Developed automated feature engineering module (`feature_engineering.py`) that generates 53 features from 8 base variables
- **Model Training Framework**: Created unified training framework (`trainer.py`) with base class architecture for consistent model development
- **Application Deployment**: Developed and deployed interactive Streamlit dashboard (`test.py`) for data exploration and model experimentation
- **Cloud Deployment**: Deployed production application to Streamlit Cloud at [https://app-test-qxq5b9dukmh7yw6xuyufxc.streamlit.app/](https://app-test-qxq5b9dukmh7yw6xuyufxc.streamlit.app/)
- **DevOps**: Managed dependencies, environment configuration, and deployment pipeline

**Key Deliverables:**
- Complete ETL pipeline with multi-source data integration
- Supabase database design and implementation
- Automated feature engineering system with temporal features, lags, and rolling statistics
- Unified model training framework supporting ETS, XGBoost, and CatBoost
- Production-ready Streamlit dashboard with EDA tools and model training interface
- Cloud deployment and production environment setup
- Technical documentation and README

**Technical Implementation:**
- Implemented temporal feature engineering (lags: 1-4 quarters, 4 years; rolling means: 1yr, 4yr, 10yr)
- Built modular trainer architecture enabling easy addition of new models
- Integrated Plotly visualizations for interactive data exploration
- Configured Supabase storage and database for team collaboration
- Developed missing data handling strategies (drop/impute)

---

## Model Distribution Strategy

The team adopted a complementary modeling approach:
- **Tree-Based Models** (Ming, Xuanhui): RandomForest, CatBoost, XGBoost - leveraging course content and natural fit for tabular time-series data
- **Linear Models** (Zhengyang): Ridge Regression and variants - exploring regularization and interpretability

This division ensured comprehensive model coverage while avoiding duplication of effort.

---

## Collaboration & Integration

All team members contributed to:
- Regular progress meetings and knowledge sharing
- Code reviews and feedback
- Documentation and final presentation preparation
- Integrated system testing and validation

The infrastructure developed by Yue served as the foundation enabling all team members to focus on their specialized modeling tasks with consistent data access and evaluation frameworks.
