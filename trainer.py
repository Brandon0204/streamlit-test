from __future__ import annotations
import os, json
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from supabase import create_client, Client
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go

def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    # Try Streamlit secrets if available
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

    # Fallback: env vars
    ENV_MAP = {"supabase.url": "SUPABASE_URL", "supabase.key": "SUPABASE_ANON_KEY"}
    env_name = ENV_MAP.get(path)
    if env_name:
        v = os.getenv(env_name)
        if v and v.strip():
            return v.strip()

    return default


class BaseTrainer(ABC):
    """
    A tiny, readable base class that:
      - connects to Supabase
      - loads feature_house
      - prepares temporal train/test split
      - computes metrics
      - builds a Plotly forecast chart
      - upserts results to house_metrics
    Subclasses only need to implement `fit()` and `predict_split()`.
    """
    model_name: str = "base"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.supabase_url = supabase_url or _get_secret("supabase.url")
        self.supabase_key = supabase_key or _get_secret("supabase.key")
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("Supabase URL/Key missing. Provide via kwargs, st.secrets, or environment.")
        self.sb: Client = create_client(self.supabase_url, self.supabase_key)

        # Prepared split artifacts
        self.train_idx = None
        self.test_idx = None

        # Univariate target series
        self.y_train = None
        self.y_test = None
        self.train_dates = None
        self.test_dates = None

        # For supervised learners
        self.X_train = None
        self.X_test = None

        # Track features used
        self.feature_list = None

        # model holder
        self.model = None

    # ---------- data helpers ----------
    def load_feature_house(self) -> pd.DataFrame:
        """Load data from feature_house table."""
        res = self.sb.table("feature_house").select("*").order("quarter").execute()
        if not res.data:
            raise ValueError("No data found in feature_house")
        df = pd.DataFrame(res.data)
        if "quarter" in df.columns:
            df["quarter"] = pd.to_datetime(df["quarter"])
        return df

    def prepare_univariate(
        self,
        df: pd.DataFrame,
        target_col: str = "hpi_growth",
        test_size: float = 0.2,
        missing_strategy: str = "drop",
    ) -> None:
        """Prepare univariate time series data for training."""
        df = df.sort_values("quarter").reset_index(drop=True)
        y = df[target_col].copy()
        dates = df["quarter"].copy() if "quarter" in df.columns else None

        if y.isna().any():
            if missing_strategy == "drop":
                valid = y.notna()
                y = y[valid]
                dates = dates[valid] if dates is not None else None
            elif missing_strategy == "impute":
                y = y.fillna(method="ffill").fillna(y.median())
            else:
                raise ValueError("missing_strategy must be 'drop' or 'impute'")

        # Reset index after filtering
        y = y.reset_index(drop=True)
        if dates is not None:
            dates = dates.reset_index(drop=True)

        split_idx = int(len(y) * (1 - test_size))
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        if dates is not None:
            self.train_dates, self.test_dates = dates.iloc[:split_idx], dates.iloc[split_idx:]

    def prepare_supervised(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "hpi_growth",
        test_size: float = 0.2,
        missing_strategy: str = "drop",
    ) -> None:
        """Prepare supervised learning data with features."""
        # Store feature list
        self.feature_list = feature_cols.copy() if feature_cols else None
        
        df = df.sort_values("quarter").reset_index(drop=True)
        critical = ["house_sales", "hpi", "house_stock", target_col]
        df = df.dropna(subset=[c for c in critical if c in df.columns])

        X, y = df[feature_cols].copy(), df[target_col].copy()
        dates = df["quarter"].copy() if "quarter" in df.columns else None

        if missing_strategy == "drop":
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            y = y.loc[valid]
            if dates is not None:
                dates = dates.loc[valid]
        elif missing_strategy == "impute":
            for c in X.columns:
                if X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())
        else:
            raise ValueError("missing_strategy must be 'drop' or 'impute'")

        # Reset index after filtering to ensure alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if dates is not None:
            dates = dates.reset_index(drop=True)

        n = len(X)
        split_idx = int(n * (1 - test_size))
        split_idx = max(1, min(n - 1, split_idx))

        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        if dates is not None:
            self.train_dates = dates.iloc[:split_idx]
            self.test_dates = dates.iloc[split_idx:]

    # ---------- training API (override) ----------
    @abstractmethod
    def fit(self, best_params: Dict) -> None:
        """Train the model with given parameters."""
        ...

    @abstractmethod
    def predict_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (y_train_pred, y_test_pred) aligned with prepared splits.
        """
        ...

    # ---------- metrics, plot, upsert ----------
    def compute_metrics(self, y_train_pred: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics for train and test sets."""
        train_r2 = float(r2_score(self.y_train, y_train_pred))
        test_r2 = float(r2_score(self.y_test, y_test_pred))
        
        # Warn about negative R2
        if test_r2 < 0:
            print(f"WARNING: Test R² is negative ({test_r2:.4f}). Model performs worse than baseline!")
            print("         Consider: more features, different hyperparameters, or longer training data")
        if train_r2 < 0:
            print(f"WARNING: Train R² is negative ({train_r2:.4f}). Serious model issues!")
        
        m = {
            "train_mse": float(mean_squared_error(self.y_train, y_train_pred)),
            "test_mse": float(mean_squared_error(self.y_test, y_test_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(self.y_train, y_train_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": float(mean_absolute_error(self.y_train, y_train_pred)),
            "test_mae": float(mean_absolute_error(self.y_test, y_test_pred)),
        }
        return m

    def plot_forecast_figure(
        self,
        y_train_pred: np.ndarray,
        y_test_pred: np.ndarray,
        title: Optional[str] = None
    ):
        """Create a Plotly figure showing train/test forecasts."""
        title = title or f"{self.model_name} — train/test forecast"
        fig = go.Figure()

        # Prepare x-axis data
        x_train = self.train_dates if self.train_dates is not None else list(range(len(self.y_train)))
        x_test = self.test_dates if self.test_dates is not None else list(range(len(self.y_train), len(self.y_train) + len(self.y_test)))

        # Train actual + fitted
        fig.add_trace(go.Scatter(
            x=x_train, 
            y=self.y_train, 
            mode="lines", 
            name="Train Actual",
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_train, 
            y=y_train_pred, 
            mode="lines", 
            name="Train Fitted",
            line=dict(color='#aec7e8', width=2)
        ))

        # Test actual + forecast
        if len(x_train) > 0 and len(x_test) > 0:
            # Get last train values
            if hasattr(x_train, 'iloc'):
                last_x_train = x_train.iloc[-1]
            else:
                last_x_train = x_train[-1]
            
            last_y_train = self.y_train.iloc[-1] if hasattr(self.y_train, 'iloc') else self.y_train[-1]
            last_y_train_pred = y_train_pred[-1]
            
            x_test_list = list(x_test) if not isinstance(x_test, list) else x_test
            x_test_connected = [last_x_train] + x_test_list
            y_test_actual_connected = [last_y_train] + list(self.y_test)
            y_test_pred_connected = [last_y_train_pred] + list(y_test_pred)
        else:
            x_test_connected = x_test
            y_test_actual_connected = self.y_test
            y_test_pred_connected = y_test_pred

        fig.add_trace(go.Scatter(
            x=x_test_connected, 
            y=y_test_actual_connected, 
            mode="lines", 
            name="Test Actual",
            line=dict(color='#d62728', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_test_connected, 
            y=y_test_pred_connected, 
            mode="lines", 
            name="Test Forecast",
            line=dict(color='#ff9896', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Target",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode='x unified'
        )
        return fig

    def upsert_metrics(self, best_params: Dict, metrics: Dict[str, float]) -> None:
        """Upsert metrics to house_metrics table with feature tracking."""
        payload = {
            "model_name": self.model_name,
            "best_params": json.dumps(best_params),
            "features": json.dumps(self.feature_list) if self.feature_list else None,
            **metrics
        }
        # upsert using model_name as key
        self.sb.table("house_metrics").upsert(payload, on_conflict="model_name").execute()