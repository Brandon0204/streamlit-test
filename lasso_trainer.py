from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import Lasso
from trainer import BaseTrainer

DEFAULT_FEATURES = [
    "hpi_growth_lag1", "hpi_growth_lag2", "hpi_growth_lag4",
    "house_sales_lag1", "house_sales_rolling_mean_1y",
]

class LassoTrainer(BaseTrainer):
    model_name = "lasso"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        super().__init__(supabase_url, supabase_key)

    def fit(self, best_params: Dict) -> None:
        if self.X_train is None:
            raise ValueError("Call prepare_supervised() before fit()")
        
        # Create and fit Lasso model
        self.model = Lasso(**best_params)
        self.model.fit(self.X_train, self.y_train)

    def predict_split(self):
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return (
            np.asarray(self.model.predict(self.X_train)),
            np.asarray(self.model.predict(self.X_test)),
        )

def run_experiment(
    best_params: Dict,
    feature_list: Optional[List[str]] = None,
    test_size: float = 0.2,
    missing_strategy: str = "drop",
):
    t = LassoTrainer()
    df = t.load_feature_house()
    features = feature_list or DEFAULT_FEATURES
    t.prepare_supervised(df, feature_cols=features, target_col="hpi_growth",
                         test_size=test_size, missing_strategy=missing_strategy)
    t.fit(best_params)
    ytr_pred, yte_pred = t.predict_split()
    metrics = t.compute_metrics(ytr_pred, yte_pred)
    fig = t.plot_forecast_figure(ytr_pred, yte_pred, title="Lasso – train/test forecast")
    t.upsert_metrics(best_params, metrics)
    return {"metrics": metrics, "figure": fig, "trainer": t}