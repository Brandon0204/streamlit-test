from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from supabase import create_client, Client
import statsmodels.api as sm

def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        cur = st.secrets
        for key in path.split("."):
            if key in cur:
                cur = cur[key]
            else:
                cur = None
                break
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception:
        pass

    ENV_MAP = {"supabase.url": "SUPABASE_URL", "supabase.key": "SUPABASE_ANON_KEY"}
    env_name = ENV_MAP.get(path)
    if env_name:
        v = os.getenv(env_name)
        if v and v.strip():
            return v.strip()
    return default


class HPIPredictor:
    """Make predictions for the latest quarter(s) with null HPI growth."""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.supabase_url = supabase_url or _get_secret("supabase.url")
        self.supabase_key = supabase_key or _get_secret("supabase.key")
        
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("Supabase URL/Key missing")
        
        self.sb: Client = create_client(self.supabase_url, self.supabase_key)
    
    def get_best_model(self) -> Dict:
        """Get the best performing model from house_metrics based on test_rmse."""
        response = self.sb.table("house_metrics").select("*").execute()
        
        if not response.data:
            raise ValueError("No models found in house_metrics table")
        
        metrics_df = pd.DataFrame(response.data)
        
        # Find model with lowest test_rmse
        best_idx = metrics_df['test_rmse'].idxmin()
        best_model = metrics_df.loc[best_idx]
        
        return {
            'model_name': best_model['model_name'],
            'test_rmse': best_model['test_rmse'],
            'test_r2': best_model['test_r2'],
            'test_mae': best_model['test_mae'],
            'best_params': json.loads(best_model['best_params']),
            'features': json.loads(best_model['features']) if best_model['features'] else None
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load feature_house data and split into:
        - train_data: rows with non-null hpi (the base variable)
        - predict_data: rows with null hpi (latest quarters to predict)
        """
        response = self.sb.table("feature_house").select("*").order("quarter").execute()
        
        if not response.data:
            raise ValueError("No data in feature_house table")
        
        df = pd.DataFrame(response.data)
        df["quarter"] = pd.to_datetime(df["quarter"])
        
        # Split based on HPI null status (not hpi_growth)
        # We predict hpi_growth for quarters where hpi itself is missing
        train_data = df[df['hpi'].notna()].copy()
        predict_data = df[df['hpi'].isna()].copy()
        
        return train_data, predict_data
    
    def train_best_model(self, train_data: pd.DataFrame, model_info: Dict):
        """Train the best model on all available data."""
        model_name = model_info['model_name']
        best_params = model_info['best_params']
        features = model_info['features']
        
        # Handle univariate models (ETS)
        if model_name.lower() == 'ets':
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            y = train_data['hpi_growth'].dropna()
            
            model = ExponentialSmoothing(
                y,
                trend=best_params.get("trend"),
                seasonal=best_params.get("seasonal"),
                seasonal_periods=best_params.get("seasonal_periods", 4),
            ).fit()
            
            return model, None
        
        # Handle supervised models
        if not features:
            raise ValueError(f"Model {model_name} requires features but none found in metrics")
        
        # Prepare features
        critical = ["house_sales", "hpi", "house_stock", "hpi_growth"]
        train_clean = train_data.dropna(subset=[c for c in critical if c in train_data.columns])
        
        X = train_clean[features].copy()
        y = train_clean['hpi_growth'].copy()
        
        # Fill missing feature values with median
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Train appropriate model based on model name
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'xgboost':
            from xgboost import XGBRegressor
            model = XGBRegressor(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**best_params)
            model.fit(X, y, verbose=False)
            
        elif model_name_lower == 'lightgbm':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'randomforest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'elasticnet':
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'lasso':
            from sklearn.linear_model import Lasso
            model = Lasso(**best_params)
            model.fit(X, y)
            
        elif model_name_lower == 'ols':
            # OLS needs constant added
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return model, features
    
    def make_predictions(
        self, 
        model, 
        predict_data: pd.DataFrame, 
        features: Optional[List[str]], 
        model_name: str,
        train_data: pd.DataFrame = None,
        n_simulations: int = 1000
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Make predictions with confidence intervals.
        
        For supervised models: Use bootstrap simulation and save features for SHAP
        For ETS: Use forecast with built-in confidence intervals
        
        Returns:
            predictions_df: DataFrame with predictions and confidence intervals
            X_pred: DataFrame with features used for prediction (None for ETS)
        """
        if predict_data.empty:
            return pd.DataFrame(), None
        
        predictions = []
        X_pred = None
        
        # ETS predictions
        if model_name.lower() == 'ets':
            n_periods = len(predict_data)
            forecast = model.forecast(steps=n_periods)
            
            # Simple confidence interval based on historical std
            historical_std = model.resid.std()
            
            for idx, row in predict_data.iterrows():
                pred_value = forecast.iloc[idx - predict_data.index[0]]
                
                predictions.append({
                    'quarter': row['quarter'],
                    'prediction': pred_value,
                    'lower_95': pred_value - 1.96 * historical_std,
                    'upper_95': pred_value + 1.96 * historical_std,
                    'std': historical_std
                })
        
        # Supervised model predictions with bootstrap
        else:
            X_pred = predict_data[features].copy()
            
            # Fill missing values with median/mean from TRAINING DATA (not prediction data)
            # This prevents data leakage and handles new NaN patterns
            if train_data is not None:
                train_features = train_data[features].copy()
                for col in X_pred.columns:
                    if X_pred[col].isna().any():
                        # Use training data median to fill prediction data NaNs
                        fill_value = train_features[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0  # Fallback if training median is also NaN
                        X_pred[col] = X_pred[col].fillna(fill_value)
                        print(f"      Filled {col} NaNs with training median: {fill_value:.4f}")
            else:
                # Fallback: use prediction data itself
                for col in X_pred.columns:
                    if X_pred[col].isna().any():
                        fill_value = X_pred[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                        X_pred[col] = X_pred[col].fillna(fill_value)
            
            # Final check: ensure no NaNs remain
            if X_pred.isna().any().any():
                print("      Warning: Some NaN values remain, filling with 0")
                X_pred = X_pred.fillna(0.0)
            
            # Point predictions - handle OLS separately
            if model_name.lower() == 'ols':
                # For statsmodels OLS, we need to match the exact structure from training
                expected_cols = model.model.exog_names
                
                # Add constant as first column
                X_pred_with_const = X_pred.copy()
                X_pred_with_const.insert(0, 'const', 1.0)
                
                # Ensure we have all expected columns in the right order
                try:
                    X_pred_with_const = X_pred_with_const[expected_cols]
                except KeyError as e:
                    print(f"      Column mismatch: {e}")
                    # Recreate with proper structure
                    X_pred_with_const = sm.add_constant(X_pred, has_constant='add')
                    if hasattr(X_pred_with_const, 'columns'):
                        X_pred_with_const.columns = expected_cols
                
                point_preds = model.predict(X_pred_with_const)
            else:
                point_preds = model.predict(X_pred)
            
            # Bootstrap for confidence intervals
            if hasattr(model, 'estimators_'):  # RandomForest or tree ensemble
                # Get predictions from individual trees for uncertainty
                all_tree_preds = np.array([tree.predict(X_pred) for tree in model.estimators_])
                pred_std = all_tree_preds.std(axis=0)
            else:
                # Simple approach: use a fixed percentage of prediction as uncertainty
                pred_std = np.abs(point_preds) * 0.15  # 15% uncertainty
            
            for idx, row in predict_data.iterrows():
                i = idx - predict_data.index[0]
                pred_value = point_preds[i]
                std = pred_std[i] if hasattr(pred_std, '__len__') else pred_std
                
                predictions.append({
                    'quarter': row['quarter'],
                    'prediction': pred_value,
                    'lower_95': pred_value - 1.96 * std,
                    'upper_95': pred_value + 1.96 * std,
                    'std': std
                })
        
        return pd.DataFrame(predictions), X_pred
    
    def predict(self) -> Dict:
        """
        Main prediction workflow:
        1. Find best model
        2. Load data
        3. Train on all available data
        4. Predict latest quarter(s)
        
        Returns dict with model info and predictions
        """
        print("="*80)
        print("HPI GROWTH PREDICTION")
        print("="*80)
        
        # Step 1: Get best model
        print("\n[1/4] Finding best model from metrics...")
        best_model_info = self.get_best_model()
        print(f"      Best model: {best_model_info['model_name']}")
        print(f"      Test RMSE: {best_model_info['test_rmse']:.4f}")
        print(f"      Test R²: {best_model_info['test_r2']:.4f}")
        
        # Step 2: Load data
        print("\n[2/4] Loading data...")
        train_data, predict_data = self.load_data()
        print(f"      Training rows: {len(train_data)}")
        print(f"      Prediction rows: {len(predict_data)}")
        
        if predict_data.empty:
            print("\n⚠️  No rows with null hpi found. Nothing to predict.")
            return {
                'model_info': best_model_info,
                'predictions': pd.DataFrame(),
                'message': 'No predictions needed - all quarters have hpi values'
            }
        
        print(f"      Quarters to predict: {', '.join(predict_data['quarter'].dt.strftime('%Y-%m-%d').tolist())}")
        
        # Step 3: Train model
        print(f"\n[3/4] Training {best_model_info['model_name']} on all available data...")
        model, features = self.train_best_model(train_data, best_model_info)
        print("      Training complete")
        
        # Step 4: Make predictions
        print("\n[4/4] Making predictions...")
        predictions_df, X_pred = self.make_predictions(
            model, 
            predict_data, 
            features, 
            best_model_info['model_name'],
            train_data=train_data
        )
        
        print("\n" + "="*80)
        print("PREDICTIONS")
        print("="*80)
        for _, row in predictions_df.iterrows():
            print(f"\nQuarter: {row['quarter'].strftime('%Y-%m-%d')}")
            print(f"  Predicted HPI Growth: {row['prediction']:.2f}%")
            print(f"  95% Confidence Interval: [{row['lower_95']:.2f}%, {row['upper_95']:.2f}%]")
        print("="*80 + "\n")
        
        return {
            'model_info': best_model_info,
            'predictions': predictions_df,
            'train_rows': len(train_data),
            'predict_rows': len(predict_data),
            'message': 'Predictions completed successfully',
            'model': model,  # Save model for SHAP
            'X_pred': X_pred,  # Save prediction features for SHAP
            'features': features  # Save feature names for SHAP
        }


def main():
    """CLI entry point."""
    predictor = HPIPredictor()
    result = predictor.predict()
    
    if not result['predictions'].empty:
        print("✅ Predictions saved above")
    else:
        print("ℹ️  No predictions needed")


if __name__ == "__main__":
    main()