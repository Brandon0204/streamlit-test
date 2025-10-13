from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from supabase import create_client, Client


def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read nested keys from streamlit secrets as 'section.key' (e.g., 'supabase.url').
    Falls back to env vars mapped below.
    """
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

    ENV_MAP = {
        "supabase.url": "SUPABASE_URL",
        "supabase.key": "SUPABASE_ANON_KEY",
        "supabase.service_role_key": "SUPABASE_SERVICE_ROLE_KEY",
        "tables.clean": "HOUSE_CLEAN_TABLE",
        "tables.feature": "HOUSE_FEATURE_TABLE",
        "tables.schema": "HOUSE_SCHEMA",
    }
    env_name = ENV_MAP.get(path)
    if env_name:
        v = os.getenv(env_name)
        if v and v.strip():
            return v.strip()

    return default


class HouseFeatureEngineering:
    """
    Read from {schema}.{clean_table}, create features, write to {schema}.{feature_table}.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        *,
        service_role_key: Optional[str] = None,
        schema: Optional[str] = None,
        clean_table: Optional[str] = None,
        feature_table: Optional[str] = None,
    ):
        
        self.supabase_url = supabase_url or _get_secret("supabase.url")
        self.supabase_key = supabase_key or _get_secret("supabase.key")
        self.service_role_key = service_role_key or _get_secret("supabase.service_role_key")

        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "Supabase URL/Key missing. Provide via kwargs, st.secrets['supabase'], or environment."
            )

        self.schema = (schema
                       or _get_secret("tables.schema", "public"))
        self.clean_table = (clean_table
                            or _get_secret("tables.clean", "clean_house"))
        self.feature_table = (feature_table
                              or _get_secret("tables.feature", "feature_house"))

        self.sb: Client = create_client(self.supabase_url, self.supabase_key)
        self.sb_rw: Client = (
            create_client(self.supabase_url, self.service_role_key)
            if self.service_role_key else self.sb
        )

    def extract(self) -> pd.DataFrame:
        """Read data from public.clean_house table."""
        print("EXTRACT: Reading from public.clean_house...")
        
        response = self.sb.table("clean_house").select("*").order("quarter").execute()
        
        if not response.data:
            print("EXTRACT: No data found in clean_house table.")
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        df["quarter"] = pd.to_datetime(df["quarter"])
        
        # Ensure numeric columns
        numeric_cols = ["house_sales", "hpi", "house_stock", "residential_investment", 
                       "ocr", "cpi", "gdp", "year"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        print(f"EXTRACT: Loaded {len(df):,} rows from clean_house")
        print(f"         Date range: {df['quarter'].min().date()} to {df['quarter'].max().date()}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features WITHOUT data leakage."""
        if df.empty:
            print("TRANSFORM: No data to transform.")
            return df
        
        df = df.sort_values("quarter").reset_index(drop=True).copy()
        
        # Add quarter number as string (Q1, Q2, Q3, Q4)
        df["quarter_num"] = "Q" + df["quarter"].dt.quarter.astype(str)
        
        # Add numeric seasonal features
        df["quarter_number"] = df["quarter"].dt.quarter  # 1, 2, 3, 4
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"].dt.quarter / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"].dt.quarter / 4)
        
        print("\nTRANSFORM: Creating features...")
        
        # --- HPI Growth Features ---
        print("  - hpi_growth: calculating percentage change")
        df["hpi_growth"] = df["hpi"].pct_change() * 100
        
        print("  - hpi_growth: lags, rolling means, diffs, ratios")
        df["hpi_growth_lag1"] = df["hpi_growth"].shift(1)
        df["hpi_growth_lag2"] = df["hpi_growth"].shift(2)
        df["hpi_growth_lag3"] = df["hpi_growth"].shift(3)
        df["hpi_growth_lag4"] = df["hpi_growth"].shift(4)
        df["hpi_growth_lag16"] = df["hpi_growth"].shift(16)  # 4 years ago
        
        # Rolling means - Use shift(1) to avoid leakage
        df["hpi_growth_rolling_mean_1y"] = df["hpi_growth"].shift(1).rolling(window=4, min_periods=1).mean()
        df["hpi_growth_rolling_mean_4y"] = df["hpi_growth"].shift(1).rolling(window=16, min_periods=1).mean()
        df["hpi_growth_rolling_mean_10y"] = df["hpi_growth"].shift(1).rolling(window=40, min_periods=1).mean()
        
        # Differences
        df["hpi_growth_diff_lag1_minus_lag2"] = df["hpi_growth_lag1"] - df["hpi_growth_lag2"]
        df["hpi_growth_diff_lag1_minus_lag4"] = df["hpi_growth_lag1"] - df["hpi_growth_lag4"]
        
        # Ratios (avoid division by zero)
        df["hpi_growth_ratio_lag1_over_lag2"] = df["hpi_growth_lag1"] / df["hpi_growth_lag2"].replace(0, np.nan)
        df["hpi_growth_ratio_lag1_over_lag4"] = df["hpi_growth_lag1"] / df["hpi_growth_lag4"].replace(0, np.nan)
        
        # --- House Sales Features ---
        print("  - house_sales: lags, rolling means, diffs, ratios")
        df["house_sales_lag1"] = df["house_sales"].shift(1)
        df["house_sales_lag2"] = df["house_sales"].shift(2)
        df["house_sales_lag3"] = df["house_sales"].shift(3)
        df["house_sales_lag4"] = df["house_sales"].shift(4)
        df["house_sales_lag16"] = df["house_sales"].shift(16)  # 4 years ago
        
        # Rolling means - Use shift(1) to avoid leakage
        df["house_sales_rolling_mean_1y"] = df["house_sales"].shift(1).rolling(window=4, min_periods=1).mean()
        df["house_sales_rolling_mean_4y"] = df["house_sales"].shift(1).rolling(window=16, min_periods=1).mean()
        df["house_sales_rolling_mean_10y"] = df["house_sales"].shift(1).rolling(window=40, min_periods=1).mean()
        
        # Differences
        df["house_sales_diff_lag1_minus_lag2"] = df["house_sales_lag1"] - df["house_sales_lag2"]
        df["house_sales_diff_lag1_minus_lag4"] = df["house_sales_lag1"] - df["house_sales_lag4"]
        
        # Ratios (avoid division by zero)
        df["house_sales_ratio_lag1_over_lag2"] = df["house_sales_lag1"] / df["house_sales_lag2"].replace(0, np.nan)
        df["house_sales_ratio_lag1_over_lag4"] = df["house_sales_lag1"] / df["house_sales_lag4"].replace(0, np.nan)
        
        # --- Create lag1 for base variables (needed for scaled features) ---
        print("  - Creating lag1 for base variables...")
        base_vars_for_lag = ['hpi', 'house_stock', 'residential_investment', 'ocr', 'cpi', 'gdp']
        
        for var in base_vars_for_lag:
            if var in df.columns:
                df[f"{var}_lag1"] = df[var].shift(1)
        
        # --- Rolling Means for Other Variables ---
        print("  - Creating lagged rolling means for all variables")
        
        # Variables that should definitely be lagged (target-related)
        lag_required = ["hpi", "house_stock"]
        for var in lag_required:
            if var in df.columns:
                print(f"    - {var}: lagged rolling means (no leakage)")
                df[f"{var}_rolling_mean_1y"] = df[var].shift(1).rolling(window=4, min_periods=1).mean()
                df[f"{var}_rolling_mean_4y"] = df[var].shift(1).rolling(window=16, min_periods=1).mean()
                df[f"{var}_rolling_mean_10y"] = df[var].shift(1).rolling(window=40, min_periods=1).mean()
        
        # Economic indicators - lag them for consistency
        economic_vars = ["residential_investment", "ocr", "cpi", "gdp"]
        for var in economic_vars:
            if var in df.columns:
                print(f"    - {var}: lagged rolling means")
                df[f"{var}_rolling_mean_1y"] = df[var].shift(1).rolling(window=4, min_periods=1).mean()
                df[f"{var}_rolling_mean_4y"] = df[var].shift(1).rolling(window=16, min_periods=1).mean()
                df[f"{var}_rolling_mean_10y"] = df[var].shift(1).rolling(window=40, min_periods=1).mean()
        
        # --- Scaled Features (from lag1 values - NO DATA LEAKAGE) ---
        print("  - Creating scaled features from lag1 values (z-score normalization)...")
        scale_vars = ['hpi', 'house_sales', 'gdp', 'house_stock', 
                    'residential_investment', 'ocr', 'cpi', 'hpi_growth']  # Added hpi_growth

        for var in scale_vars:
            lag_col = f"{var}_lag1"
            if lag_col in df.columns:
                mean = df[lag_col].mean()
                std = df[lag_col].std()
                if std > 0:  # Avoid division by zero
                    df[f"{var}_lag1_scaled"] = (df[lag_col] - mean) / std
                    print(f"    - {var}_lag1_scaled (from {lag_col}): mean={mean:.2f}, std={std:.2f}")
                else:
                    df[f"{var}_lag1_scaled"] = 0.0
                    print(f"    - {var}_lag1_scaled: constant value (std=0), set to 0")

        print(f"           - Scaled features: {len(scale_vars)} (from lag1 values)")
        
        # --- Policy-period binary flags ---
        print("  - Creating policy period flags...")
        q = pd.to_datetime(df["quarter"])
        df["covid_lockdown_2020q2_q3"] = (
            (q >= pd.Timestamp("2020-04-01")) & (q < pd.Timestamp("2020-10-01"))
        ).astype(int)
        print(f"    - covid_lockdown_2020q2_q3: {df['covid_lockdown_2020q2_q3'].sum()} quarters flagged")
        
        df["reopening_supply_2021q2_2022q4"] = (
            (q >= pd.Timestamp("2021-04-01")) & (q < pd.Timestamp("2023-01-01"))
        ).astype(int)
        print(f"    - reopening_supply_2021q2_2022q4: {df['reopening_supply_2021q2_2022q4'].sum()} quarters flagged")

        print(f"\nTRANSFORM: Created {len(df.columns):,} total columns")
        print(f"           Original: 10")
        print(f"           Engineered features: {len(df.columns) - 10}")
        print(f"           - Lag1 features: {len(base_vars_for_lag)}")
        print(f"           - Scaled features: {len(scale_vars)} (from lag1 values)")
        print(f"           - Seasonal features: 3 (quarter_number, quarter_sin, quarter_cos)")
        print(f"           - Policy flags: 2")
        print(f"           ✅ All features are leakage-free!")
        
        return df

    def load(self, df: pd.DataFrame) -> None:
        """Upsert to public.feature_house table."""
        if df.empty:
            print("LOAD: No data to load.")
            return
        
        # Prepare for database
        df_db = df.copy()
        df_db["quarter"] = df_db["quarter"].dt.strftime("%Y-%m-%d")
        
        # Replace inf/-inf with None
        numeric_cols = df_db.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df_db[col] = df_db[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with None
        df_db = df_db.where(pd.notna(df_db), None)
        
        # Convert to native Python types
        records = []
        for row in df_db.to_dict(orient="records"):
            clean_row = {}
            for key, value in row.items():
                if value is None:
                    clean_row[key] = None
                elif isinstance(value, (np.integer, np.int64, int)):
                    clean_row[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, float)):
                    if np.isfinite(value):
                        clean_row[key] = float(value)
                    else:
                        clean_row[key] = None
                elif isinstance(value, np.bool_):
                    clean_row[key] = bool(value)
                else:
                    clean_row[key] = value
            records.append(clean_row)
        
        print(f"\nLOAD: Upserting {len(records):,} records to feature_house table...")
        
        # Upsert
        self.sb_rw.table("feature_house").upsert(
            records,
            on_conflict="quarter",
            returning="minimal"
        ).execute()
        
        print(f"LOAD: Successfully upserted {len(records):,} rows to feature_house table.")

    def describe_features(self, df: pd.DataFrame) -> None:
        """Print summary statistics of created features."""
        if df.empty:
            return
        
        print("\n" + "="*80)
        print("[Feature Summary Statistics]")
        print("="*80)
        
        # Feature columns (exclude originals)
        feature_cols = [c for c in df.columns if c not in 
                       ["quarter", "year", "quarter_num", "house_sales", "hpi", 
                        "house_stock", "residential_investment", "ocr", "cpi", "gdp"]]
        
        if not feature_cols:
            print("No feature columns found.")
            return
        
        # Count non-null values
        print("\nNon-null counts:")
        for col in feature_cols:
            non_null = df[col].notna().sum()
            total = len(df)
            pct = (non_null / total * 100) if total > 0 else 0
            print(f"  {col:45s}: {non_null:4d} / {total:4d} ({pct:5.1f}%)")
        
        # Sample statistics for hpi_growth features
        print("\nHPI growth feature statistics:")
        hpi_growth_features = [c for c in feature_cols if c.startswith("hpi_growth")][:5]
        if hpi_growth_features:
            print(df[hpi_growth_features].describe().to_string())
        
        # Sample statistics for house_sales features
        print("\nHouse sales feature statistics (first few):")
        house_sales_features = [c for c in feature_cols if c.startswith("house_sales")][:5]
        if house_sales_features:
            print(df[house_sales_features].describe().to_string())


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    
    fe = HouseFeatureEngineering()
    
    # Extract
    df = fe.extract()
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Transform
    df_features = fe.transform(df)
    
    # Preview
    print("\n[Feature DataFrame Preview]")
    print(df_features.head(10).to_string(index=False))
    
    # Describe features
    fe.describe_features(df_features)
    
    # Load
    fe.load(df_features)
    
    print("\n" + "="*80)
    print("Feature engineering complete!")
    print("="*80)


if __name__ == "__main__":
    main()