from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import statsmodels.api as sm
from trainer import BaseTrainer

DEFAULT_FEATURES = [
    "hpi_growth_lag1", "hpi_growth_lag2", "hpi_growth_lag4",
    "house_sales_lag1", "house_sales_rolling_mean_1y",
]

class OLSTrainer(BaseTrainer):
    model_name = "ols"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        super().__init__(supabase_url, supabase_key)

    def fit(self, best_params: Dict) -> None:
        if self.X_train is None:
            raise ValueError("Call prepare_supervised() before fit()")
        
        # Add constant for intercept
        X_train_with_const = sm.add_constant(self.X_train)
        
        # Fit OLS model
        self.model = sm.OLS(self.y_train, X_train_with_const).fit()
        
        # Store the constant flag for prediction
        self._has_constant = True

    def predict_split(self):
        if self.model is None:
            raise ValueError("Model not fitted")
        
        # Add constant to both train and test sets
        X_train_with_const = sm.add_constant(self.X_train)
        X_test_with_const = sm.add_constant(self.X_test)
        
        return (
            np.asarray(self.model.predict(X_train_with_const)),
            np.asarray(self.model.predict(X_test_with_const)),
        )

def run_experiment(
    best_params: Dict,
    feature_list: Optional[List[str]] = None,
    test_size: float = 0.2,
    missing_strategy: str = "drop",
):
    t = OLSTrainer()
    df = t.load_feature_house()
    features = feature_list or DEFAULT_FEATURES
    t.prepare_supervised(df, feature_cols=features, target_col="hpi_growth",
                         test_size=test_size, missing_strategy=missing_strategy)
    t.fit(best_params)
    ytr_pred, yte_pred = t.predict_split()
    metrics = t.compute_metrics(ytr_pred, yte_pred)
    fig = t.plot_forecast_figure(ytr_pred, yte_pred, title="OLS – train/test forecast")
    t.upsert_metrics(best_params, metrics)
    return {"metrics": metrics, "figure": fig, "trainer": t}