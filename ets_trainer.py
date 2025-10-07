# ets_trainer.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from trainer import BaseTrainer, _get_secret  # NEW

class ETSTrainer(BaseTrainer):
    model_name = "ets"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        super().__init__(supabase_url, supabase_key)
        self.fitted_model = None

    def fit(self, best_params: Dict) -> None:
        if self.y_train is None:
            raise ValueError("Call prepare_univariate() before fit()")
        self.fitted_model = ExponentialSmoothing(
            self.y_train,
            trend=best_params.get("trend"),
            seasonal=best_params.get("seasonal"),
            seasonal_periods=best_params.get("seasonal_periods", 4),
        ).fit()

    def predict_split(self):
        if self.fitted_model is None:
            raise ValueError("Model not fitted")
        y_train_pred = self.fitted_model.fittedvalues
        y_test_pred = self.fitted_model.forecast(steps=len(self.y_test))
        return np.asarray(y_train_pred), np.asarray(y_test_pred)

def run_experiment(
    best_params: Dict, 
    test_size: float = 0.2, 
    missing_strategy: str = "drop",
    feature_list: Optional[List[str]] = None  # NEW - for consistency
):
    t = ETSTrainer()
    df = t.load_feature_house()
    t.prepare_univariate(df, target_col="hpi_growth", test_size=test_size, missing_strategy=missing_strategy)
    t.fit(best_params)
    ytr_pred, yte_pred = t.predict_split()
    metrics = t.compute_metrics(ytr_pred, yte_pred)
    fig = t.plot_forecast_figure(ytr_pred, yte_pred, title="ETS — train/test forecast")
    # metrics -> house_metrics (feature_list will be None for ETS)
    t.upsert_metrics(best_params, metrics)
    return {"metrics": metrics, "figure": fig, "trainer": t}

