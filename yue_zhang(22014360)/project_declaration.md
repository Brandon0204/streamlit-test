# Project Declaration and Architecture Justification

## Component Delivery Approach

### Jupyter Notebook Professional Report (35%)

**Team Approach:**
Each team member will submit their own Jupyter notebook demonstrating:
- Individual EDA specific to their selected features
- Model experimentation and hyperparameter tuning for their assigned model type (linear vs. tree-based)
- Analysis and interpretation of results
- Rich evaluation metrics and performance comparisons

**Integration with Production Application:**
The comprehensive EDA has been integrated into the production Streamlit application (`test.py`), providing interactive analysis including:
- **Target Distribution Analysis**: Histogram and time-series visualization of HPI growth
- **Feature Correlations**: Interactive correlation analysis with the target variable
- **Missing Values Analysis**: Comprehensive missingness reporting across all features
- **Summary Statistics**: Descriptive statistics for all engineered features
- **Feature Importance**: Model-neutral F-statistic based feature ranking with statistical significance testing

This approach ensures that EDA is not only documented in notebooks but also productionized as an interactive tool for ongoing analysis.

---

## Productionised Deliverable (25%)

### Model Experimentation and XAI Integration

The production application includes:

1. **Multiple Algorithm Support**: ETS (Exponential Smoothing), XGBoost, and CatBoost with configurable hyperparameters
2. **Interactive Hyperparameter Tuning**: UI-based parameter adjustment with JSON configuration
3. **Feature Selection Interface**: Dynamic feature selection for supervised models
4. **Missing Data Handling**: Configurable strategies (drop/impute)
5. **Performance Metrics**: Comprehensive evaluation (RMSE, MSE, MAE, R²) for train/test splits
6. **Model Comparison Dashboard**: Side-by-side performance visualization
7. **XAI (Explainability)**: SHAP integration available in the application for model interpretation

The application provides a complete MLOps workflow from data ingestion to model deployment and evaluation.

---

## Cloud Infrastructure: AWS vs. Supabase Decision

### Why Not AWS?

**Challenges Encountered:**
1. **Account Limitations**: The team did not have access to proper AWS accounts with sufficient permissions for the required services
2. **Learning Curve**: AWS services (S3, RDS, Lambda, SageMaker) have steep learning curves with complex IAM policies and networking configurations
3. **Collaboration Complexity**: Managing multi-user access, credentials, and development environments across a 4-member team would introduce significant overhead
4. **Cost Management**: AWS billing complexity and risk of unexpected charges posed concerns for a student project

### Supabase as the Solution

**Architecture Decision:**
Supabase was selected as a lightweight, developer-friendly alternative that provides equivalent functionality to AWS services:

| AWS Service | Supabase Equivalent | Underlying Technology |
|-------------|---------------------|----------------------|
| Amazon S3 | Supabase Storage | S3 (backend) |
| Amazon RDS | Supabase Database | PostgreSQL on AWS infrastructure |
| API Gateway | Auto-generated REST API | PostgREST |
| IAM | Row Level Security (RLS) | PostgreSQL policies |

**Key Advantages:**
- **Unified Platform**: Single SDK for storage and database operations
- **Zero DevOps**: Managed infrastructure with automatic scaling
- **Team Collaboration**: Simple credential sharing via environment variables or Streamlit secrets
- **PostgreSQL Native**: Direct SQL access with Python SDK abstraction
- **AWS-Backed**: Supabase infrastructure runs on AWS, providing enterprise-grade reliability
- **Cost-Effective**: Generous free tier suitable for development and deployment

**Technical Implementation:**
```python
# Single client for all operations
from supabase import create_client
sb = create_client(url, key)

# File storage (S3 equivalent)
blob = sb.storage.from_('bucket').download('file.xlsx')

# Database operations (RDS equivalent)
sb.table('clean_house').upsert(data).execute()
```

This abstraction allowed the team to focus on modeling and feature engineering rather than infrastructure management.

---

## Originality and Contribution (15%)

### Novel Aspects of Our Work

1. **End-to-End MLOps Pipeline**: Unlike typical notebook-only projects, we built a production-grade system with automated ETL, feature engineering, and model training
   
2. **Multi-Model Framework**: Unified trainer architecture (`trainer.py`) enabling consistent experimentation across diverse algorithm families (time-series, tree-based, linear)

3. **Dynamic Feature Engineering**: Automated generation of 53 temporal features with configurable windows, supporting rapid iteration

4. **Centralized Feature Store**: Implemented `feature_house` table as a common feature repository, ensuring consistency across all team members' modeling efforts

4. **Centralized Feature Store**: Implemented `feature_house` table as a common feature repository, ensuring consistency across all team members' modeling efforts

5. **Interactive Deployment**: Live application allowing non-technical users to explore data, train models, and compare results without code

5. **Interactive Deployment**: Live application allowing non-technical users to explore data, train models, and compare results without code

6. **Integrated XAI**: Explainability tools built directly into the production interface, not just notebook artifacts

### Where Difficulty Lay

**Infrastructure Challenges:**
- **AWS Migration Complexity**: Initial attempts to use AWS services proved too complex for team collaboration
- **Solution**: Pivoted to Supabase, drastically reducing infrastructure overhead while maintaining cloud-native architecture
- **Result**: Enabled seamless collaboration with shared data access and version-controlled codebase

**Technical Challenges:**
- **Multi-Source Data Integration**: Merging quarterly house data with economic indicators (CPI, GDP, OCR) requiring careful temporal alignment
- **Feature Engineering at Scale**: Creating 53 features from 8 base variables while handling missing data and maintaining temporal integrity
- **Model Framework Design**: Building an abstract base class that works for both univariate time-series (ETS) and multivariate supervised models (XGBoost, CatBoost)
- **Production Deployment**: Ensuring Streamlit Cloud compatibility with Supabase, managing secrets, and optimizing for cloud execution

**Data Engineering Challenges:**
- **Temporal Consistency**: Ensuring quarter-end date alignment across different data sources
- **Missing Data Strategy**: Implementing multiple strategies (drop, impute, forward-fill) with configurable options
- **Database Schema Design**: Structuring tables to support both raw data and engineered features efficiently
- **Centralized Feature Store Design**: Creating a unified feature repository (`feature_house` table) to prevent duplication and ensure consistency

### The Importance of a Common Feature Store

One of the critical architectural decisions was implementing a centralized feature store (`feature_house` table). This approach proved essential for several reasons:

**Without a Common Feature Store:**
- Each team member generates their own features independently
- **Naming Conflicts**: Different members might create the same feature with different names (e.g., "hpi_lag1" vs. "lagged_hpi_1q")
- **Duplicate Work**: Multiple team members unknowingly create identical features, wasting effort
- **Feature Leakage Risk**: Inconsistent feature generation can lead to data leakage if one member's approach inadvertently uses future information
- **Value Reconciliation**: Differences in implementation (e.g., forward-fill vs. median imputation) produce different values for conceptually similar features
- **Prediction Nightmare**: When deploying models, each model requires different feature preparation logic, making production deployment extremely complex
- **Communication Overhead**: Constant need to explain and document each member's feature engineering choices

**With Centralized Feature Store (`feature_house`):**
- **Single Source of Truth**: All 53 features generated once using consistent logic
- **Standardized Naming**: Clear, systematic naming conventions (e.g., `hpi_growth_lag1`, `house_sales_rolling_mean_1y`)
- **Reproducibility**: Every team member uses identical features, ensuring fair model comparisons
- **No Leakage**: Feature engineering logic reviewed and validated once, preventing temporal leakage
- **Simplified Deployment**: All models use the same feature set, requiring only one feature preparation pipeline for production
- **Easy Collaboration**: Team members can focus on model selection and tuning rather than feature engineering
- **Efficient Experimentation**: Feature selection becomes a configuration choice rather than a coding task

This design pattern follows industry best practices (as seen in tools like Feast, Tecton, and AWS SageMaker Feature Store) and was critical to enabling efficient team collaboration at scale.

### Contribution to the Field

This project demonstrates that production-grade ML systems can be built without enterprise AWS expertise by leveraging modern developer platforms (Supabase, Streamlit Cloud) while maintaining:
- Cloud-native architecture
- Scalable data pipelines
- Reproducible experiments
- User-friendly interfaces

The modular design allows easy extension with new models, features, or data sources—providing a template for future time-series prediction projects.

---

## Summary

Our project showcases a complete machine learning system that prioritizes:
- **Collaboration**: Simplified infrastructure enabling team productivity
- **Production Quality**: Deployed application with real-time model training
- **Explainability**: Integrated EDA and XAI tools
- **Pragmatism**: Choosing appropriate tools (Supabase) over complex alternatives (raw AWS) to deliver results efficiently

Individual notebooks will demonstrate deep dives into specific modeling approaches, while the integrated application represents our collective engineering effort to productionize the entire workflow.