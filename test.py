import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback
from supabase import create_client, Client
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from ingest_h import HousePriceETL
import io
from contextlib import redirect_stdout, redirect_stderr
from feature_engineering import HouseFeatureEngineering
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from io import BytesIO
from ets_trainer import run_experiment as run_ets
from xgboost_trainer import run_experiment as run_xgb
from catboost_trainer import run_experiment as run_cat
from ridge_trainer import run_experiment as run_ridge
from randomforest_trainer import run_experiment as run_rf
from elasticnet_trainer import run_experiment as run_en
from lgbm_trainer import run_experiment as run_lgbm
from ols_trainer import run_experiment as run_ols
from lasso_trainer import run_experiment as run_lasso
from predict import HPIPredictor

def check_multicollinearity(features: List[str], df: pd.DataFrame, threshold: float = 0.8) -> Tuple[bool, List[Tuple[str, str, float]]]:
    """
    Check for highly correlated features.
    
    Returns:
        has_issues: bool - True if correlations above threshold found
        correlated_pairs: List of (feature1, feature2, correlation) tuples
    """
    if len(features) < 2:
        return False, []
    
    # Get feature data
    X = df[features].copy()
    
    # Fill NaN for correlation calculation
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find pairs above threshold
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                correlated_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    return len(correlated_pairs) > 0, correlated_pairs

st.set_page_config(page_title="House Price Data", layout="wide")
st.title("House Price Data Viewer")

# Initialize session state for preserving results
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# -----------------------------
# Supabase Connection
# -----------------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("supabase", {}).get("url") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("supabase", {}).get("key") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Supabase credentials not found. Check `.streamlit/secrets.toml` or environment variables.")
        st.stop()
    return create_client(url, key)

sb = get_supabase_client()

# -----------------------------
# Fetch Data
# -----------------------------
@st.cache_data(ttl=300)
def fetch_table(table_name: str) -> pd.DataFrame:
    """Fetch entire table from Supabase."""
    try:
        response = sb.table(table_name).select("*").order("quarter").execute()
        if not response.data:
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        if "quarter" in df.columns:
            df["quarter"] = pd.to_datetime(df["quarter"])
        return df
    except Exception as e:
        st.error(f"Error fetching {table_name}: {e}")
        return pd.DataFrame()

# -----------------------------
# Main Interface
# -----------------------------
st.markdown("### Select Table")
table_name = st.selectbox(
    "Choose a table to view",
    options=["feature_house", "clean_house"],
    index=0,
    help="Select which table to view and download"
)

# Fetch button
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Load Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    st.markdown("&nbsp;")  # spacing

# Load data
with st.spinner(f"Loading {table_name}..."):
    df = fetch_table(table_name)

if df.empty:
    st.warning(f"No data found in {table_name}")
else:
    # Display info
    st.success(f"Loaded {len(df):,} rows")
    
    # Preview - Show data in expander for feature_house, always show for clean_house
    if table_name == "feature_house":
        st.markdown("### Data Preview")
        
        # Add view options
        view_cols = st.columns([2, 2, 3])
        with view_cols[0]:
            view_mode = st.radio(
                "View mode",
                ["First rows", "Last rows", "All rows"],
                horizontal=True,
                key="view_mode"
            )
        
        with view_cols[1]:
            if view_mode in ["First rows", "Last rows"]:
                n_rows = st.number_input(
                    "Number of rows",
                    min_value=5,
                    max_value=len(df),
                    value=min(20, len(df)),
                    step=5,
                    key="n_preview_rows"
                )
        
        # Display based on selection
        if view_mode == "First rows":
            st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True, height=600)
        elif view_mode == "Last rows":
            st.dataframe(df.tail(n_rows), use_container_width=True, hide_index=True, height=600)
        else:  # All rows
            st.dataframe(df, use_container_width=True, hide_index=True, height=600)
            st.caption(f"Showing all {len(df):,} rows")
    
    else:  # clean_house - show in collapsed expander
        with st.expander("📊 Data Preview", expanded=False):
            view_cols = st.columns([2, 2, 3])
            with view_cols[0]:
                view_mode = st.radio(
                    "View mode",
                    ["First rows", "Last rows", "All rows"],
                    horizontal=True,
                    key="view_mode_clean"
                )
            
            with view_cols[1]:
                if view_mode in ["First rows", "Last rows"]:
                    n_rows = st.number_input(
                        "Number of rows",
                        min_value=5,
                        max_value=len(df),
                        value=min(20, len(df)),
                        step=5,
                        key="n_preview_rows_clean"
                    )
            
            # Display based on selection
            if view_mode == "First rows":
                st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True, height=400)
            elif view_mode == "Last rows":
                st.dataframe(df.tail(n_rows), use_container_width=True, hide_index=True, height=400)
            else:  # All rows
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
                st.caption(f"Showing all {len(df):,} rows")
    
    # Column info
    with st.expander("📋 Column Information"):
        st.write(f"**Total columns:** {len(df.columns)}")
        st.write(f"**Columns:** {', '.join(df.columns)}")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write(f"**Numeric columns:** {len(numeric_cols)}")
        
        date_range = ""
        if "quarter" in df.columns:
            date_range = f"{df['quarter'].min().date()} to {df['quarter'].max().date()}"
            st.write(f"**Date range:** {date_range}")
    
    # -----------------------------
    # Download section
    # -----------------------------
    st.markdown("### Download Data")
    if table_name == "feature_house":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {table_name}.csv ({len(df):,} rows)",
            data=csv,
            file_name=f"{table_name}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
            key="download_feature_house"  # unique key just in case
        )
    else:
        st.caption("⬇️ Download is disabled for `clean_house`. Switch to `feature_house` to export.")


# -----------------------------
# EDA Section (Feature Importance)
# -----------------------------
if not df.empty and table_name == "feature_house" and "hpi_growth" in df.columns:
    st.divider()
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # Prepare data for feature importance
    target_col = "hpi_growth"
    exclude_cols = ["quarter", "year", "quarter_num", target_col, 
                   "house_sales", "hpi", "house_stock"]  # Exclude to prevent leakage
    
    # Get feature columns (numeric only)
    feature_cols = [col for col in df.select_dtypes(include='number').columns 
                   if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        st.warning("No feature columns found for analysis")
    else:
        # Remove rows with missing values in critical columns
        critical_cols = ["house_sales", "hpi", "house_stock", target_col]
        df_analysis = df.dropna(subset=critical_cols).copy()
        
        if len(df_analysis) < 10:
            st.warning(f"Not enough data for analysis (only {len(df_analysis)} rows with valid target)")
        else:
            st.markdown(f"### Feature Importance for `{target_col}`")
            st.caption(f"Analyzing {len(feature_cols)} features using {len(df_analysis):,} data points (excluding house_sales, hpi, house_stock to prevent leakage)")
            
            # Prepare features and target
            X = df_analysis[feature_cols].copy()
            y = df_analysis[target_col].copy()
            
            # Fill missing values with median
            for col in X.columns:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Use SelectKBest with f_regression for model-neutral feature importance
            with st.spinner("Computing feature importance (using F-statistic)..."):
                try:
                    # Calculate F-scores for all features
                    selector = SelectKBest(score_func=f_regression, k='all')
                    selector.fit(X, y)
                    
                    # Get feature scores
                    feature_importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'f_score': selector.scores_,
                        'p_value': selector.pvalues_
                    }).sort_values('f_score', ascending=False)
                    
                    # Add normalized importance (0-100 scale)
                    max_score = feature_importance_df['f_score'].max()
                    feature_importance_df['importance'] = (
                        feature_importance_df['f_score'] / max_score * 100
                    ).round(2)
                    
                    # Display top features
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Top 20 Most Important Features")
                        
                        # Create horizontal bar chart
                        top_n = min(20, len(feature_importance_df))
                        top_features = feature_importance_df.head(top_n)
                        
                        fig = px.bar(
                            top_features,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title=f'Top {top_n} Features by F-Score (Model-Neutral)',
                            labels={'importance': 'Normalized Importance (0-100)', 'feature': 'Feature'},
                            color='importance',
                            color_continuous_scale='Blues',
                            hover_data={'f_score': ':.2f', 'p_value': ':.2e'}
                        )
                        fig.update_layout(
                            height=600,
                            yaxis={'categoryorder': 'total ascending'},
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Feature Importance Table")
                        st.caption(f"Showing top 20 of {len(feature_importance_df)} features (F-statistic method)")
                        
                        # Format for display
                        display_df = top_features.copy()
                        display_df['rank'] = range(1, len(display_df) + 1)
                        display_df['significance'] = display_df['p_value'].apply(
                            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        )
                        
                        st.dataframe(
                            display_df[['rank', 'feature', 'importance', 'f_score', 'significance']].rename(
                                columns={
                                    'importance': 'importance (0-100)',
                                    'f_score': 'F-score'
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                            height=600
                        )
                        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05")
                    
                    # Download feature importance
                    st.markdown("#### Download Feature Importance")
                    csv_importance = feature_importance_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download All Feature Importance ({len(feature_importance_df)} features)",
                        data=csv_importance,
                        file_name="feature_importance.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Additional EDA sections
                    st.divider()
                    st.markdown("### Additional Analysis")
                    
                    eda_tabs = st.tabs([
                        "📈 Target Distribution", 
                        "🔗 Feature Correlations",
                        "📉 Missing Values",
                        "📊 Summary Statistics"
                    ])
                    
                    # Tab 1: Target Distribution
                    with eda_tabs[0]:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig_hist = px.histogram(
                                df_analysis,
                                x=target_col,
                                nbins=50,
                                title=f'Distribution of {target_col}',
                                labels={target_col: 'HPI Growth (%)'}
                            )
                            fig_hist.update_layout(showlegend=False)
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Time series
                            if "quarter" in df_analysis.columns:
                                fig_ts = px.line(
                                    df_analysis,
                                    x='quarter',
                                    y=target_col,
                                    title=f'{target_col} Over Time',
                                    labels={target_col: 'HPI Growth (%)', 'quarter': 'Quarter'}
                                )
                                st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Statistics
                        st.markdown("**Target Statistics:**")
                        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
                        with stats_col1:
                            st.metric("Mean", f"{y.mean():.2f}%")
                        with stats_col2:
                            st.metric("Median", f"{y.median():.2f}%")
                        with stats_col3:
                            st.metric("Std Dev", f"{y.std():.2f}%")
                        with stats_col4:
                            st.metric("Min", f"{y.min():.2f}%")
                        with stats_col5:
                            st.metric("Max", f"{y.max():.2f}%")
                    
                    # Tab 2: Correlations
                    with eda_tabs[1]:
                        st.markdown("#### Top 15 Features Correlated with Target")
                        
                        # Calculate correlations
                        correlations = X.corrwith(y).abs().sort_values(ascending=False).head(15)
                        corr_df = pd.DataFrame({
                            'feature': correlations.index,
                            'correlation': correlations.values
                        })
                        
                        fig_corr = px.bar(
                            corr_df,
                            x='correlation',
                            y='feature',
                            orientation='h',
                            title='Absolute Correlation with HPI Growth',
                            labels={'correlation': 'Absolute Correlation', 'feature': 'Feature'},
                            color='correlation',
                            color_continuous_scale='Reds'
                        )
                        fig_corr.update_layout(
                            height=500,
                            yaxis={'categoryorder': 'total ascending'},
                            showlegend=False
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Tab 3: Missing Values
                    with eda_tabs[2]:
                        missing_df = pd.DataFrame({
                            'feature': df[feature_cols].columns,
                            'missing_count': df[feature_cols].isnull().sum().values,
                            'missing_pct': (df[feature_cols].isnull().sum().values / len(df) * 100)
                        }).sort_values('missing_count', ascending=False)
                        
                        # Only show features with missing values
                        missing_df = missing_df[missing_df['missing_count'] > 0]
                        
                        if len(missing_df) == 0:
                            st.success("✅ No missing values in any features!")
                        else:
                            st.warning(f"⚠️ {len(missing_df)} features have missing values")
                            
                            fig_missing = px.bar(
                                missing_df.head(20),
                                x='missing_pct',
                                y='feature',
                                orientation='h',
                                title='Top 20 Features with Missing Values',
                                labels={'missing_pct': 'Missing (%)', 'feature': 'Feature'},
                                color='missing_pct',
                                color_continuous_scale='Oranges'
                            )
                            fig_missing.update_layout(
                                height=500,
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig_missing, use_container_width=True)
                            
                            st.dataframe(
                                missing_df,
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Tab 4: Summary Statistics
                    with eda_tabs[3]:
                        st.markdown("#### Summary Statistics for All Features")
                        
                        summary_stats = X.describe().T
                        summary_stats['missing'] = df[feature_cols].isnull().sum().values
                        summary_stats['missing_pct'] = (summary_stats['missing'] / len(df) * 100).round(2)
                        
                        st.dataframe(
                            summary_stats,
                            use_container_width=True,
                            height=500
                        )
                        
                        # Download summary stats
                        csv_stats = summary_stats.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Summary Statistics",
                            data=csv_stats,
                            file_name="feature_summary_stats.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error computing feature importance: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

# -----------------------------
# Model Experimentation Section
# -----------------------------
if not df.empty and table_name == "feature_house" and "hpi_growth" in df.columns:
    st.divider()
    st.header("🚀 Model Training & Evaluation")

    # Prepare available features once
    exclude_cols = ["quarter", "year", "quarter_num", "hpi_growth",
                    "house_sales", "hpi", "house_stock"]
    available_features = [c for c in df.select_dtypes(include="number").columns 
                        if c not in exclude_cols]
    
    # Categorize features for better UX
    lag_features = [f for f in available_features if '_lag' in f]
    rolling_features = [f for f in available_features if '_rolling_' in f]
    diff_features = [f for f in available_features if '_diff_' in f]
    ratio_features = [f for f in available_features if '_ratio_' in f]
    policy_features = [f for f in available_features if 'covid' in f or 'reopening' in f]
    scaled_features = [f for f in available_features if '_scaled' in f]

    with st.expander("⚙️ Model Configuration", expanded=True):
        # Model selection
        st.markdown("#### Select Models to Train")
        model_cols = st.columns(9)
        with model_cols[0]:
            use_ets = st.checkbox("📈 ETS (Univariate)", value=True)
        with model_cols[1]:
            use_xgb = st.checkbox("🌳 XGBoost", value=True)
        with model_cols[2]:
            use_cat = st.checkbox("🐱 CatBoost", value=True)
        with model_cols[3]:
            use_lgbm = st.checkbox("💡 LightGBM", value=True)
        with model_cols[4]:
            use_rf = st.checkbox("🌲 RandomForest", value=True)
        with model_cols[5]:
            use_en = st.checkbox("🎯 ElasticNet", value=True)
        with model_cols[6]:
            use_ridge = st.checkbox("📐 Ridge", value=True)
        with model_cols[7]:
            use_lasso = st.checkbox("🎯 Lasso", value=True)
        with model_cols[8]:
            use_ols = st.checkbox("📊 OLS", value=True) 

        st.divider()
        
        # Common training options
        st.markdown("#### Training Options")
        option_cols = st.columns(3)
        
        with option_cols[0]:
            test_size_pct = st.slider(
                "Test set size (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data for testing"
            )
            test_size = max(0.01, min(0.5, test_size_pct / 100.0))
        
        with option_cols[1]:
            missing_strategy = st.radio(
                "Missing value strategy",
                options=["drop", "impute"],
                help="How to handle missing values"
            )
        
        with option_cols[2]:
            show_shap = st.checkbox(
                "Show SHAP plots",
                value=True,
                help="Generate SHAP feature importance (adds ~10s per model)"
            )
        
        st.divider()
        
        # Feature info box
        with st.expander("ℹ️ Available Feature Categories"):
            info_cols = st.columns(6)
            with info_cols[0]:
                st.metric("Lag Features", len(lag_features))
            with info_cols[1]:
                st.metric("Rolling Features", len(rolling_features))
            with info_cols[2]:
                st.metric("Diff Features", len(diff_features))
            with info_cols[3]:
                st.metric("Ratio Features", len(ratio_features))
            with info_cols[4]:
                st.metric("Policy Features", len(policy_features))
            with info_cols[5]:
                st.metric("Scaled Features", len(scaled_features))

    # ========================================
    # Per-Model Configuration Tabs
    # ========================================
    st.markdown("#### Model-Specific Configuration")
    st.caption("Configure hyperparameters and features for each model independently")

    st.markdown("##### 🎛️ Feature Filter")
    st.caption("Filter available features to make selection easier")
    
    filter_cols = st.columns([1, 1, 1, 1, 1])
    
    with filter_cols[0]:
        show_lag = st.checkbox("Lag", value=True, key="filter_lag", 
                              help="Features with time lags (_lag1, _lag2, etc.)")
    with filter_cols[1]:
        show_rolling = st.checkbox("Rolling", value=True, key="filter_rolling",
                                   help="Rolling mean features (_rolling_mean_1y, etc.)")
    with filter_cols[2]:
        show_diff = st.checkbox("Diff/Ratio", value=True, key="filter_diff",
                               help="Difference and ratio features")
    with filter_cols[3]:
        show_policy = st.checkbox("Policy", value=True, key="filter_policy",
                                 help="COVID and reopening period flags")
    with filter_cols[4]:
        show_scaled = st.checkbox("Scaled", value=True, key="filter_scaled",
                                 help="Z-score normalized features (_scaled)")
    
    # Filter available_features based on checkboxes
    filtered_available = []
    for f in available_features:
        include = False
        
        if show_scaled and f.endswith('_scaled'):
            include = True
        elif show_policy and ('covid' in f or 'reopening' in f):
            include = True
        elif show_diff and ('_diff_' in f or '_ratio_' in f):
            include = True
        elif show_rolling and '_rolling_' in f:
            include = True
        elif show_lag and '_lag' in f and '_rolling_' not in f:  # lag but not rolling
            include = True
        
        if include:
            filtered_available.append(f)
    
    st.caption(f"📊 Showing **{len(filtered_available)}** of **{len(available_features)}** features")
    
    config_tabs = st.tabs([
        "📈 ETS", "🌳 XGBoost", "🐱 CatBoost", "💡 LightGBM", 
        "🌲 RandomForest", "🎯 ElasticNet", "📐 Ridge", "🎯 Lasso", "📊 OLS"
    ])
    # Default features for supervised models
    default_features = [f for f in [
        'hpi_growth_lag1', 'hpi_growth_lag2', 'hpi_growth_lag4',
        'hpi_growth_rolling_mean_1y',
        'house_sales_lag1', 'house_sales_rolling_mean_1y',
        'ocr_rolling_mean_1y', 'cpi_rolling_mean_1y'
    ] if f in filtered_available]
    
    # ETS Configuration
    with config_tabs[0]:
        st.markdown("##### Hyperparameters")
        ets_params = st.text_area(
            "ETS Parameters (JSON)",
            '{"trend": "add", "seasonal": "add", "seasonal_periods": 4}',
            height=100,
            help="Exponential Smoothing State Space Model parameters",
            key="ets_params"
        )
        st.info("ℹ️ ETS is a univariate model and doesn't use additional features")
    
    # XGBoost Configuration
    with config_tabs[1]:
        st.markdown("##### Hyperparameters")
        xgb_params = st.text_area(
            "XGBoost Parameters (JSON)",
            '{"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "n_jobs": 1, "nthread": 1, "random_state": 42}',
            height=120,
            help="Gradient boosting hyperparameters",
            key="xgb_params"
        )
        
        st.markdown("##### Feature Selection")
        xgb_features = st.multiselect(
            "Select features for XGBoost",
            options=filtered_available,
            default=default_features,
            help="Choose features for XGBoost model",
            key="xgb_features"
        )
        st.caption(f"Selected: **{len(xgb_features)}** features")
        
        if xgb_features:
            with st.expander("View selected features"):
                st.write(", ".join(xgb_features))
    
    # CatBoost Configuration
    with config_tabs[2]:
        st.markdown("##### Hyperparameters")
        cat_params = st.text_area(
            "CatBoost Parameters (JSON)",
            '{"depth": 6, "learning_rate": 0.05, "n_estimators": 600, "random_seed": 42}',
            height=120,
            help="CatBoost gradient boosting parameters",
            key="cat_params"
        )
        
        st.markdown("##### Feature Selection")
        cat_features = st.multiselect(
            "Select features for CatBoost",
            options=filtered_available,
            default=default_features,
            help="Choose features for CatBoost model",
            key="cat_features"
        )
        st.caption(f"Selected: **{len(cat_features)}** features")
        
        if cat_features:
            with st.expander("View selected features"):
                st.write(", ".join(cat_features))

    # LightGBM Configuration
    with config_tabs[3]:
        st.markdown("##### Hyperparameters")
        lgbm_params = st.text_area(
            "LightGBM Parameters (JSON)",
            '{"n_estimators": 1036, "learning_rate": 0.0699, "num_leaves": 39, "max_depth": 14, "subsample": 0.834, "colsample_bytree": 0.755, "reg_alpha": 0.00005, "reg_lambda": 0.0027, "n_jobs": 1}',
            height=120,
            help="LightGBM gradient boosting parameters",
            key="lgbm_params"
        )
        
        st.markdown("##### Feature Selection")
        lgbm_features = st.multiselect(
            "Select features for LightGBM",
            options=filtered_available,
            default=default_features,
            help="Choose features for LightGBM model",
            key="lgbm_features"
        )
        st.caption(f"Selected: **{len(lgbm_features)}** features")
        
        if lgbm_features:
            with st.expander("View selected features"):
                st.write(", ".join(lgbm_features))

    # RandomForest Configuration
    with config_tabs[4]:
        st.markdown("##### Hyperparameters")
        rf_params = st.text_area(
            "RandomForest Parameters (JSON)",
            '{"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}',
            height=120,
            help="Random Forest ensemble parameters",
            key="rf_params"
        )
        
        st.markdown("##### Feature Selection")
        rf_features = st.multiselect(
            "Select features for RandomForest",
            options=filtered_available,
            default=default_features,
            help="Choose features for RandomForest model",
            key="rf_features"
        )
        st.caption(f"Selected: **{len(rf_features)}** features")
        
        if rf_features:
            with st.expander("View selected features"):
                st.write(", ".join(rf_features))

    # ElasticNet Configuration
    with config_tabs[5]:
        st.markdown("##### Hyperparameters")
        en_params = st.text_area(
            "ElasticNet Parameters (JSON)",
            '{"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000, "random_state": 42}',
            height=120,
            help="ElasticNet linear regression parameters. alpha=regularization strength, l1_ratio=0.5 means equal L1/L2",
            key="en_params"
        )
        
        st.markdown("##### Feature Selection")
        en_features = st.multiselect(
            "Select features for ElasticNet",
            options=filtered_available,
            default=default_features,
            help="Choose features for ElasticNet model",
            key="en_features"
        )
        st.caption(f"Selected: **{len(en_features)}** features")
        
        if en_features and len(en_features) > 1:
            has_issues, corr_pairs = check_multicollinearity(en_features, df, threshold=0.85)
            if has_issues:
                st.warning(f"⚠️ **Multicollinearity detected!** {len(corr_pairs)} feature pairs have correlation > 0.85")
                with st.expander("View correlated features"):
                    for f1, f2, corr in corr_pairs[:10]:  # Show first 10
                        st.write(f"• `{f1}` ↔ `{f2}`: {corr:.3f}")
                    if len(corr_pairs) > 10:
                        st.caption(f"... and {len(corr_pairs) - 10} more pairs")
                    st.info("💡 **Tip:** Linear models work best when features are not highly correlated. Consider removing one from each pair.")

        if en_features:
            with st.expander("View selected features"):
                st.write(", ".join(en_features))

    # Ridge Configuration
    with config_tabs[6]:
        st.markdown("##### Hyperparameters")
        ridge_params = st.text_area(
            "Ridge Parameters (JSON)",
            '{"alpha": 1.0, "max_iter": 1000, "random_state": 42}',
            height=100,
            help="Ridge regression parameters. alpha=regularization strength (higher = more regularization)",
            key="ridge_params"
        )
        
        st.markdown("##### Feature Selection")
        ridge_features = st.multiselect(
            "Select features for Ridge",
            options=filtered_available,
            default=default_features,
            help="Choose features for Ridge regression model",
            key="ridge_features"
        )
        st.caption(f"Selected: **{len(ridge_features)}** features")
        if ridge_features and len(ridge_features) > 1:
            has_issues, corr_pairs = check_multicollinearity(ridge_features, df, threshold=0.85)
            if has_issues:
                st.warning(f"⚠️ **Multicollinearity detected!** {len(corr_pairs)} feature pairs have correlation > 0.85")
                with st.expander("View correlated features"):
                    for f1, f2, corr in corr_pairs[:10]:
                        st.write(f"• `{f1}` ↔ `{f2}`: {corr:.3f}")
                    if len(corr_pairs) > 10:
                        st.caption(f"... and {len(corr_pairs) - 10} more pairs")
                    st.info("💡 **Tip:** While Ridge handles multicollinearity better than OLS, reducing correlation still improves stability.")
        if ridge_features:
            with st.expander("View selected features"):
                st.write(", ".join(ridge_features))

    # Lasso Configuration
    with config_tabs[7]:
        st.markdown("##### Hyperparameters")
        lasso_params = st.text_area(
            "Lasso Parameters (JSON)",
            '{"alpha": 1.0, "max_iter": 1000, "random_state": 42}',
            height=100,
            help="Lasso regression parameters. alpha=regularization strength (higher = more sparsity/feature selection)",
            key="lasso_params"
        )
        
        st.markdown("##### Feature Selection")
        lasso_features = st.multiselect(
            "Select features for Lasso",
            options=filtered_available,
            default=default_features,
            help="Choose features for Lasso regression model",
            key="lasso_features"
        )
        st.caption(f"Selected: **{len(lasso_features)}** features")
        if lasso_features and len(lasso_features) > 1:
            has_issues, corr_pairs = check_multicollinearity(lasso_features, df, threshold=0.85)
            if has_issues:
                st.warning(f"⚠️ **Multicollinearity detected!** {len(corr_pairs)} feature pairs have correlation > 0.85")
                with st.expander("View correlated features"):
                    for f1, f2, corr in corr_pairs[:10]:
                        st.write(f"• `{f1}` ↔ `{f2}`: {corr:.3f}")
                    if len(corr_pairs) > 10:
                        st.caption(f"... and {len(corr_pairs) - 10} more pairs")
                    st.info("💡 **Tip:** Lasso naturally handles multicollinearity by zeroing out redundant features.")
        if lasso_features:
            with st.expander("View selected features"):
                st.write(", ".join(lasso_features))

    # OLS Configuration
    with config_tabs[8]:
        st.markdown("##### Hyperparameters")
        ols_params = st.text_area(
            "OLS Parameters (JSON)",
            '{}',
            height=80,
            help="Ordinary Least Squares - no hyperparameters needed. Uses statsmodels.OLS",
            key="ols_params"
        )
        
        st.markdown("##### Feature Selection")
        ols_features = st.multiselect(
            "Select features for OLS",
            options=filtered_available,
            default=default_features,
            help="Choose features for OLS regression model",
            key="ols_features"
        )
        st.caption(f"Selected: **{len(ols_features)}** features")
        if ols_features and len(ols_features) > 1:
            has_issues, corr_pairs = check_multicollinearity(ols_features, df, threshold=0.8)
            if has_issues:
                st.error(f"🚨 **High Multicollinearity Detected!** {len(corr_pairs)} feature pairs have correlation > 0.80")
                with st.expander("View correlated features", expanded=True):
                    for f1, f2, corr in corr_pairs[:15]:
                        st.write(f"• `{f1}` ↔ `{f2}`: **{corr:.3f}**")
                    if len(corr_pairs) > 15:
                        st.caption(f"... and {len(corr_pairs) - 15} more pairs")
                    st.warning("⚠️ **OLS is very sensitive to multicollinearity!** This can cause unstable coefficients and extreme predictions. Please remove highly correlated features.")
                    st.info("💡 **Common culprits:** Don't mix scaled features (e.g., `cpi_scaled`) with their rolling means (e.g., `cpi_rolling_mean_1y`)")
        if ols_features:
            with st.expander("View selected features"):
                st.write(", ".join(ols_features))

    # Training button
    st.markdown("---")
    go_train = st.button(
        "🚀 Train Selected Models", 
        type="primary", 
        use_container_width=True,
        help="This will train selected models and save results to database"
    )

    if go_train:
        # Validation
        selected_models = [use_ets, use_xgb, use_cat, use_rf, use_en, use_lgbm, use_ridge, use_lasso, use_ols]
        if not any(selected_models):
            st.warning("⚠️ Please select at least one model to train")
        elif use_xgb and not xgb_features:
            st.warning("⚠️ XGBoost selected but no features chosen")
        elif use_cat and not cat_features:
            st.warning("⚠️ CatBoost selected but no features chosen")
        elif use_rf and not rf_features:
            st.warning("⚠️ RandomForest selected but no features chosen")
        elif use_en and not en_features:
            st.warning("⚠️ ElasticNet selected but no features chosen")
        elif use_lgbm and not lgbm_features:
            st.warning("⚠️ LightGBM selected but no features chosen")
        elif use_ridge and not ridge_features:
            st.warning("⚠️ Ridge selected but no features chosen")
        elif use_lasso and not lasso_features:
            st.warning("⚠️ Lasso selected but no features chosen")
        elif use_ols and not ols_features:
            st.warning("⚠️ OLS selected but no features chosen")
        else:
            results = []
            charts = []
            shap_plots = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_train = sum(selected_models)
            current_model = 0

            # Train ETS
            if use_ets:
                current_model += 1
                status_text.text(f"Training ETS ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(ets_params)
                    r = run_ets(
                        best_params=p, 
                        test_size=test_size, 
                        missing_strategy=missing_strategy,
                        feature_list=None
                    )
                    results.append({"model_name": "ETS", **r["metrics"]})
                    charts.append(("ETS", r["figure"]))
                    st.success("✅ ETS: Training completed")
                except Exception as e:
                    st.error(f"❌ ETS failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Train XGBoost
            if use_xgb and xgb_features:
                current_model += 1
                status_text.text(f"Training XGBoost ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(xgb_params)
                    r = run_xgb(
                        best_params=p, 
                        feature_list=xgb_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "XGBoost", **r["metrics"]})
                    charts.append(("XGBoost", r["figure"]))
                    
                    # Generate SHAP plot
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for XGBoost...")
                        try:
                            trainer = r["trainer"]
                            explainer = shap.TreeExplainer(trainer.model)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(xgb_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=xgb_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(xgb_features)
                            )
                            plt.title("XGBoost - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("XGBoost", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ XGBoost: Training completed")
                except Exception as e:
                    st.error(f"❌ XGBoost failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Train CatBoost
            if use_cat and cat_features:
                current_model += 1
                status_text.text(f"Training CatBoost ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(cat_params)
                    r = run_cat(
                        best_params=p, 
                        feature_list=cat_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "CatBoost", **r["metrics"]})
                    charts.append(("CatBoost", r["figure"]))
                    
                    # Generate SHAP plot
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for CatBoost...")
                        try:
                            trainer = r["trainer"]
                            explainer = shap.TreeExplainer(trainer.model)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(cat_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=cat_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(cat_features)
                            )
                            plt.title("CatBoost - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("CatBoost", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ CatBoost: Training completed")
                except Exception as e:
                    st.error(f"❌ CatBoost failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            # Train LightGBM
            if use_lgbm and lgbm_features:
                current_model += 1
                status_text.text(f"Training LightGBM ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(lgbm_params)
                    r = run_lgbm(
                        best_params=p, 
                        feature_list=lgbm_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "LightGBM", **r["metrics"]})
                    charts.append(("LightGBM", r["figure"]))
                    
                    # Generate SHAP plot
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for LightGBM...")
                        try:
                            trainer = r["trainer"]
                            explainer = shap.TreeExplainer(trainer.model)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(lgbm_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=lgbm_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(lgbm_features)
                            )
                            plt.title("LightGBM - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("LightGBM", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ LightGBM: Training completed")
                except Exception as e:
                    st.error(f"❌ LightGBM failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Train RandomForest
            if use_rf and rf_features:
                current_model += 1
                status_text.text(f"Training RandomForest ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(rf_params)
                    r = run_rf(
                        best_params=p, 
                        feature_list=rf_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "RandomForest", **r["metrics"]})
                    charts.append(("RandomForest", r["figure"]))
                    
                    # Generate SHAP plot
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for RandomForest...")
                        try:                            
                            trainer = r["trainer"]
                            explainer = shap.TreeExplainer(trainer.model)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(rf_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=rf_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(rf_features)
                            )
                            plt.title("RandomForest - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("RandomForest", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ RandomForest: Training completed")
                except Exception as e:
                    st.error(f"❌ RandomForest failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            # Train ElasticNet
            if use_en and en_features:
                current_model += 1
                status_text.text(f"Training ElasticNet ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(en_params)
                    r = run_en(
                        best_params=p, 
                        feature_list=en_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "ElasticNet", **r["metrics"]})
                    charts.append(("ElasticNet", r["figure"]))
                    
                    # SHAP for ElasticNet using LinearExplainer
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for ElasticNet...")
                        try:                            
                            trainer = r["trainer"]
                            
                            # Use LinearExplainer for ElasticNet
                            explainer = shap.LinearExplainer(trainer.model, trainer.X_train)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(en_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=en_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(en_features)
                            )
                            plt.title("ElasticNet - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("ElasticNet", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ ElasticNet: Training completed")
                except Exception as e:
                    st.error(f"❌ ElasticNet failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())            

            # Train Ridge
            if use_ridge and ridge_features:
                current_model += 1
                status_text.text(f"Training Ridge ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(ridge_params)
                    r = run_ridge(
                        best_params=p, 
                        feature_list=ridge_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "Ridge", **r["metrics"]})
                    charts.append(("Ridge", r["figure"]))
                    
                    # SHAP for Ridge using LinearExplainer
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for Ridge...")
                        try:                            
                            trainer = r["trainer"]
                            
                            # Use LinearExplainer for Ridge
                            explainer = shap.LinearExplainer(trainer.model, trainer.X_train)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(ridge_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=ridge_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(ridge_features)
                            )
                            plt.title("Ridge - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("Ridge", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ Ridge: Training completed")
                except Exception as e:
                    st.error(f"❌ Ridge failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Train Lasso
            if use_lasso and lasso_features:
                current_model += 1
                status_text.text(f"Training Lasso ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(lasso_params)
                    r = run_lasso(
                        best_params=p, 
                        feature_list=lasso_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "Lasso", **r["metrics"]})
                    charts.append(("Lasso", r["figure"]))
                    
                    # SHAP for Lasso using LinearExplainer
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for Lasso...")
                        try:                            
                            trainer = r["trainer"]
                            
                            explainer = shap.LinearExplainer(trainer.model, trainer.X_train)
                            shap_values = explainer.shap_values(trainer.X_test)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(lasso_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                trainer.X_test, 
                                feature_names=lasso_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(lasso_features)
                            )
                            plt.title("Lasso - SHAP Summary Plot", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("Lasso", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot: {shap_err}")
                    
                    st.success("✅ Lasso: Training completed")
                except Exception as e:
                    st.error(f"❌ Lasso failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Train OLS
            if use_ols and ols_features:
                current_model += 1
                status_text.text(f"Training OLS ({current_model}/{models_to_train})...")
                progress_bar.progress(current_model / models_to_train)
                
                try:
                    p = json.loads(ols_params)
                    r = run_ols(
                        best_params=p, 
                        feature_list=ols_features,
                        test_size=test_size, 
                        missing_strategy=missing_strategy
                    )
                    results.append({"model_name": "OLS", **r["metrics"]})
                    charts.append(("OLS", r["figure"]))
                    
                    # SHAP for OLS using KernelExplainer (statsmodels not directly supported)
                    if show_shap and "trainer" in r:
                        status_text.text(f"Generating SHAP summary plot for OLS...")
                        try:                            
                            trainer = r["trainer"]
                            
                            # Create a wrapper function for statsmodels prediction
                            # This must add the constant in the same way as training
                            def predict_fn(X):
                                # X is a numpy array, convert to DataFrame with feature names
                                X_df = pd.DataFrame(X, columns=ols_features)
                                X_with_const = sm.add_constant(X_df, has_constant='add')
                                
                                # Ensure column order matches model expectations
                                if hasattr(trainer.model.model, 'exog_names'):
                                    expected_cols = trainer.model.model.exog_names
                                    X_with_const = X_with_const[expected_cols]
                                
                                return trainer.model.predict(X_with_const)
                            
                            # Use KernelExplainer with a sample of training data as background
                            # Use fewer samples for speed (KernelExplainer is slow)
                            background_size = min(50, len(trainer.X_train))
                            background = shap.sample(trainer.X_train, background_size)
                            
                            # Use fewer test samples for SHAP (it's very slow for OLS)
                            test_sample_size = min(20, len(trainer.X_test))
                            test_sample = shap.sample(trainer.X_test, test_sample_size)
                            
                            explainer = shap.KernelExplainer(predict_fn, background)
                            shap_values = explainer.shap_values(test_sample)
                            
                            fig_matplotlib, ax = plt.subplots(figsize=(10, max(6, len(ols_features) * 0.3)))
                            shap.summary_plot(
                                shap_values, 
                                test_sample, 
                                feature_names=ols_features,
                                show=False,
                                plot_type="dot",
                                color_bar=True,
                                max_display=len(ols_features)
                            )
                            plt.title("OLS - SHAP Summary Plot (sampled)", fontsize=14, pad=20)
                            plt.xlabel("SHAP value (impact on model output)", fontsize=11)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig_matplotlib)
                            
                            shap_plots.append(("OLS", buf))
                        except Exception as shap_err:
                            st.warning(f"Could not generate SHAP plot for OLS: {shap_err}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    st.success("✅ OLS: Training completed")
                except Exception as e:
                    st.error(f"❌ OLS failed: {e}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Store results in session state
            if results:
                st.session_state.training_results = {
                    'results': results,
                    'charts': charts,
                    'shap_plots': shap_plots
                }

    # Display training results (from session state or just generated)
    if st.session_state.training_results:
        display_data = st.session_state.training_results
        results = display_data['results']
        charts = display_data['charts']
        shap_plots = display_data['shap_plots']
        
        if results:
            st.divider()
            st.subheader("📊 Results")
            
            # Metrics comparison
            tab1, tab2, tab3 = st.tabs(["📋 Metrics Table", "📈 Forecast Charts", "🔍 SHAP Analysis"])
            
            with tab1:
                st.markdown("### Performance Metrics Comparison")
                cmp_df = pd.DataFrame(results).set_index("model_name")
                
                # Highlight best scores
                def highlight_best(s):
                    if s.name in ['test_rmse', 'test_mae', 'test_mse', 'train_rmse', 'train_mae', 'train_mse']:
                        is_best = s == s.min()
                    else:  # R² - higher is better
                        is_best = s == s.max()
                    return ['background-color: lightgreen' if v else '' for v in is_best]
                
                st.dataframe(
                    cmp_df.style.apply(highlight_best).format("{:.4f}"),
                    use_container_width=True
                )
                
                # Show warning if any R² is negative
                if (cmp_df['test_r2'] < 0).any():
                    st.warning("⚠️ Some models have negative test R². This means they perform worse than predicting the mean. Consider retraining with different features or hyperparameters.")
                
                # Visual comparison
                st.markdown("### Test Set Performance")
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    fig_rmse = px.bar(
                        cmp_df.reset_index(),
                        x="model_name", 
                        y="test_rmse",
                        text="test_rmse",
                        title="Test RMSE (lower is better)",
                        color="model_name"
                    )
                    fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_rmse.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with metric_cols[1]:
                    fig_r2 = px.bar(
                        cmp_df.reset_index(),
                        x="model_name", 
                        y="test_r2",
                        text="test_r2",
                        title="Test R² (higher is better)",
                        color="model_name"
                    )
                    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_r2.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with metric_cols[2]:
                    fig_mae = px.bar(
                        cmp_df.reset_index(),
                        x="model_name", 
                        y="test_mae",
                        text="test_mae",
                        title="Test MAE (lower is better)",
                        color="model_name"
                    )
                    fig_mae.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_mae.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_mae, use_container_width=True)
            
            with tab2:
                st.markdown("### Forecast Visualizations")
                if charts:
                    for name, fig in charts:
                        st.markdown(f"#### {name}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No forecast charts available.")
            
            with tab3:
                st.markdown("### SHAP Summary Plot Analysis")
                if shap_plots:
                    st.caption("📊 SHAP summary plots show how each feature impacts predictions. Each dot is a test sample. Red = high feature value, Blue = low feature value. Position shows impact on prediction.")
                    
                    # Show interpretation guide once at the top
                    with st.expander("💡 How to Read SHAP Summary Plots"):
                        st.markdown("""
                        **Reading the plot:**
                        - **Y-axis**: Features sorted by importance (most important at top)
                        - **X-axis**: SHAP value (impact on prediction)
                            - Positive (right) = increases predicted HPI growth
                            - Negative (left) = decreases predicted HPI growth
                        - **Color**: Feature value
                            - Red = high feature value
                            - Blue = low feature value
                        - **Each dot**: One test prediction
                        
                        **Example insights:**
                        - If red dots are mostly on the right for a feature, it means high values of that feature increase predictions
                        - If dots are spread across both sides, the feature has complex/non-linear effects
                        - Features at the top matter most for predictions
                        """)
                    
                    # Display all SHAP plots
                    for name, img_buffer in shap_plots:
                        st.markdown(f"#### {name}")
                        st.image(img_buffer, use_container_width=True)
                else:
                    st.info("ℹ️ No SHAP plots available. Enable 'Show SHAP plots' in the training options to generate feature importance visualizations.")
        else:
            st.warning("⚠️ No models were successfully trained. Please check the error messages above.")

# -----------------------------
# Production Prediction Section
# -----------------------------
st.divider()
st.markdown("### 🔮 Forecast Future HPI Growth")
st.caption("Predict HPI growth for the latest quarter(s)")

# Center the content
col_left, col_center, col_right = st.columns([1, 6, 1])

with col_center:
    st.markdown("""
    This will:
    1. 🏆 Select the best performing model from training history
    2. 📊 Train it on all available historical data
    3. 🎯 Predict HPI growth for the latest quarter(s)
    4. 📈 Provide 95% confidence intervals for predictions
    """)
    
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            try:
                predictor = HPIPredictor()
                result = predictor.predict()
                
                # Store in session state
                st.session_state.prediction_results = result
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())
    
    # Display prediction results (from session state or just generated)
    if st.session_state.prediction_results:
        result = st.session_state.prediction_results
        
        # Display results
        st.success("✅ Predictions completed!")
                
        # Model info
        st.markdown("#### 📊 Model Information")
        model_info = result['model_info']
        
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Best Model", model_info['model_name'])
        with info_cols[1]:
            st.metric("Test RMSE", f"{model_info['test_rmse']:.4f}")
        with info_cols[2]:
            st.metric("Test R²", f"{model_info['test_r2']:.4f}")
        with info_cols[3]:
            st.metric("Test MAE", f"{model_info['test_mae']:.4f}")
        
        # Show features used (if supervised model)
        if model_info['features']:
            with st.expander(f"📋 Features used ({len(model_info['features'])} features)"):
                st.write(", ".join(model_info['features']))
        
        # Predictions
        predictions_df = result['predictions']
        
        if not predictions_df.empty:
            st.markdown("#### 🎯 Predictions")
            st.caption(f"Trained on {result['train_rows']} historical quarters")
            
            # Format predictions nicely
            display_df = predictions_df.copy()
            
            # Fix quarter formatting - use proper quarter format
            display_df['quarter_str'] = display_df['quarter'].dt.to_period('Q').astype(str).str.replace('Q', ' Q')
            
            display_df['prediction'] = display_df['prediction'].round(2)
            display_df['lower_95'] = display_df['lower_95'].round(2)
            display_df['upper_95'] = display_df['upper_95'].round(2)
            display_df['confidence_interval'] = display_df.apply(
                lambda row: f"[{row['lower_95']:.2f}%, {row['upper_95']:.2f}%]", 
                axis=1
            )
            
            # Display table
            st.dataframe(
                display_df[['quarter_str', 'prediction', 'confidence_interval']].rename(columns={
                    'quarter_str': 'Quarter',
                    'prediction': 'Predicted HPI Growth (%)',
                    'confidence_interval': '95% Confidence Interval'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization - Show historical data + predictions
            st.markdown("#### 📈 Prediction Visualization")
            
            temp_predictor = HPIPredictor()
            train_data, _ = temp_predictor.load_data()
            
            fig_pred = go.Figure()
            
            # Add historical data (last 20 quarters for context)
            recent_history = train_data.tail(20)
            fig_pred.add_trace(go.Scatter(
                x=recent_history['quarter'],
                y=recent_history['hpi_growth'],
                mode='lines+markers',
                name='Historical',
                marker=dict(size=6, color='blue'),
                line=dict(color='blue', width=2)
            ))
            
            # Add prediction points
            fig_pred.add_trace(go.Scatter(
                x=predictions_df['quarter'],
                y=predictions_df['prediction'],
                mode='markers+lines',
                name='Prediction',
                marker=dict(size=12, color='red'),
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add confidence interval
            fig_pred.add_trace(go.Scatter(
                x=predictions_df['quarter'].tolist() + predictions_df['quarter'].tolist()[::-1],
                y=predictions_df['upper_95'].tolist() + predictions_df['lower_95'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                name='95% Confidence',
                showlegend=True
            ))
            
            fig_pred.update_layout(
                title=f"HPI Growth Forecast - {model_info['model_name']}",
                xaxis_title="Quarter",
                yaxis_title="HPI Growth (%)",
                hovermode='x unified',
                height=500
            )
            
            # st.plotly_chart(fig_pred, use_container_width=True)
            
            # === Tabs under Prediction Visualization: Chart + SHAP Waterfalls ===
            viz_tabs = st.tabs(["📈 Forecast Chart", "🧮 SHAP Waterfalls"])

            with viz_tabs[0]:
                # Re-show the forecast chart here so everything related stays together
                st.plotly_chart(fig_pred, use_container_width=True)

            with viz_tabs[1]:
                st.caption("Per-quarter feature contribution breakdown using SHAP waterfalls.")

                model = result.get("model")
                X_pred = result.get("X_pred")          # features for the rows being predicted (None for ETS)
                feat_names = result.get("features")    # list of feature names (None for ETS)
                model_name = result.get("model_info", {}).get("model_name", "")

                # SHAP waterfalls only apply to supervised models with features
                if model is None or X_pred is None or not isinstance(X_pred, pd.DataFrame) or not feat_names:
                    st.info("Waterfall plots are only available for supervised models with features "
                            "(e.g., XGBoost, CatBoost, RandomForest, ElasticNet). ETS is univariate, so no per-feature waterfall.")
                else:
                    # How many features (bars) to show per plot (rest will be grouped as 'other')
                    max_display = st.slider(
                        "Max features to display per plot",
                        min_value=5,
                        max_value=min(20, len(feat_names)),
                        value=min(10, len(feat_names)),
                        help="Shows the top contributors and groups the rest as 'other'."
                    )

                    # Build a SHAP explainer appropriate to the model
                    # (TreeExplainer for tree models, LinearExplainer for ElasticNet)
                    explainer = None
                    shap_values = None
                    expected_value = None

                    try:
                        lower_name = (model_name or "").lower()
                        if any(k in lower_name for k in ["xgboost", "catboost", "randomforest", "lightgbm"]):
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_pred)  # (n_samples, n_features)
                            expected_value = explainer.expected_value
                            if not np.isscalar(expected_value):
                                # Some versions return an array; take scalar for regression
                                expected_value = float(np.asarray(expected_value).reshape(-1)[0])
                        elif any(k in lower_name for k in ["elasticnet", "ridge"]):  # CHANGE THIS LINE
                            explainer = shap.LinearExplainer(model, X_pred)
                            shap_values = explainer.shap_values(X_pred)
                            expected_value = explainer.expected_value
                            if not np.isscalar(expected_value):
                                expected_value = float(np.asarray(expected_value).reshape(-1)[0])
                        else:
                            st.info(f"Waterfall not implemented for model type: {model_name}")
                    except Exception as e:
                        st.warning(f"Could not prepare SHAP explainer for {model_name}: {e}")

                    if shap_values is not None and expected_value is not None:
                        # Normalize to a 2D array (n_samples, n_features)
                        if isinstance(shap_values, list):  # defensive (multiclass-style)
                            shap_values = shap_values[0]
                        shap_values = np.asarray(shap_values)

                        # Ensure columns are aligned
                        feature_names = X_pred.columns.tolist()
                        feature_data = X_pred.to_numpy()

                        # Render a waterfall per prediction row
                        for i, (_, p_row) in enumerate(predictions_df.iterrows()):
                            if i >= shap_values.shape[0]:
                                break

                            # Build a SHAP Explanation object (works nicely with shap.plots.waterfall)
                            exp = shap.Explanation(
                                values=shap_values[i],                 # SHAP contributions per feature
                                base_values=expected_value,            # model baseline
                                data=feature_data[i],                  # actual feature values
                                feature_names=feature_names
                            )

                            # Title with quarter label
                            qlabel = pd.to_datetime(p_row["quarter"]).to_period("Q").strftime("%Y Q%q")
                            st.markdown(f"**{model_name} — {qlabel}**")

                            # Draw the matplotlib-based waterfall and show it in Streamlit
                            # Note: shap.plots.waterfall returns a matplotlib figure/axes
                            try:
                                fig_wf = shap.plots.waterfall(exp, max_display=max_display, show=False)
                                st.pyplot(fig_wf, clear_figure=True)
                            except Exception as plot_err:
                                # Older/newer SHAP versions sometimes return None; fallback: force draw via plt.gcf()
                                import matplotlib.pyplot as plt
                                shap.plots.waterfall(exp, max_display=max_display, show=False)
                                st.pyplot(plt.gcf(), clear_figure=True)

                    else:
                        st.info("Could not compute SHAP values for this model to build waterfalls.")


            # Interpretation
            st.markdown("#### 💡 Interpretation")
            
            latest_pred = predictions_df.iloc[-1]
            pred_val = latest_pred['prediction']
            lower_val = latest_pred['lower_95']
            upper_val = latest_pred['upper_95']
            
            # Fix quarter string formatting
            quarter_str = latest_pred['quarter'].to_period('Q').strftime('%Y Q%q')
            
            # Determine sentiment
            if pred_val > 2:
                sentiment = "🟢 **Strong Growth**"
                desc = "robust increase"
            elif pred_val > 0:
                sentiment = "🟡 **Moderate Growth**"
                desc = "modest increase"
            elif pred_val > -2:
                sentiment = "🟠 **Slight Decline**"
                desc = "small decrease"
            else:
                sentiment = "🔴 **Significant Decline**"
                desc = "notable decrease"
            
            st.markdown(f"""
            **{quarter_str} Forecast: {sentiment}**
            
            The {model_info['model_name']} model predicts a **{abs(pred_val):.2f}% {desc}** in house price index growth 
            for {quarter_str}, with 95% confidence that the actual value will fall between 
            **{lower_val:.2f}%** and **{upper_val:.2f}%**.
            
            This prediction is based on {result['train_rows']} quarters of historical data and considers 
            {len(model_info['features']) if model_info['features'] else 'time series'} 
            {'features' if model_info['features'] else 'patterns'} in the model.
            """)
            
            # Download predictions
            st.markdown("#### 💾 Export Predictions")
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions (CSV)",
                data=csv,
                file_name=f"hpi_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("ℹ️ No predictions needed - all quarters in the dataset have HPI values.")
                
       
# -----------------------------
# Data Refresh Section
# -----------------------------
st.divider()
st.markdown("### Refresh Data Pipeline")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Ingest Raw Data")
    st.caption("Refresh clean_house table from source files")
    if st.button("Run Ingest Pipeline", use_container_width=True):
        with st.spinner("Running ingest pipeline..."):
            try:                
                output_buffer = io.StringIO()
                error_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    etl = HousePriceETL()
                    raw = etl.extract()
                    combined = etl.transform(raw)
                    etl.load(combined)
                
                output = output_buffer.getvalue()
                errors = error_buffer.getvalue()
                
                st.success("Ingest completed successfully!")
                with st.expander("View Output"):
                    st.code(output if output else "No output")
                if errors:
                    with st.expander("View Warnings"):
                        st.code(errors)
                
                st.cache_data.clear()
                
            except Exception as e:
                st.error(f"Ingest failed: {str(e)}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

with col2:
    st.markdown("#### Generate Features")
    st.caption("Refresh feature_house table from clean_house")
    if st.button("Run Feature Engineering", use_container_width=True):
        with st.spinner("Running feature engineering..."):
            try:                
                output_buffer = io.StringIO()
                error_buffer = io.StringIO()
                
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    fe = HouseFeatureEngineering()
                    df = fe.extract()
                    if not df.empty:
                        df_features = fe.transform(df)
                        fe.load(df_features)
                    else:
                        print("No data found in clean_house table")
                
                output = output_buffer.getvalue()
                errors = error_buffer.getvalue()
                
                st.success("Feature engineering completed successfully!")
                with st.expander("View Output"):
                    st.code(output if output else "No output")
                if errors:
                    with st.expander("View Warnings"):
                        st.code(errors)
                
                st.cache_data.clear()
                
            except Exception as e:
                st.error(f"Feature engineering failed: {str(e)}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

st.markdown("---")
st.caption("Use the buttons above to refresh data from source files")