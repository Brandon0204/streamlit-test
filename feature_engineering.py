# feature_engineering.py — Feature engineering for house price data
from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client


class HouseFeatureEngineering:
    """
    Read from public.clean_house, create features, write to public.feature_house.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        load_dotenv()
        self.supabase_url = supabase_url or os.environ["SUPABASE_URL"]
        self.supabase_key = supabase_key or os.environ["SUPABASE_ANON_KEY"]
        
        # Anon client for reading
        self.sb: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Service role client for writing
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.sb_rw: Client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.sb

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
        """Create all features."""
        if df.empty:
            print("TRANSFORM: No data to transform.")
            return df
        
        df = df.sort_values("quarter").reset_index(drop=True).copy()
        
        # Add quarter number (Q1, Q2, Q3, Q4)
        df["quarter_num"] = df["quarter"].dt.quarter
        
        print("\nTRANSFORM: Creating features...")
        
        # --- House Sales Features (comprehensive) ---
        print("  - house_sales: lags, rolling means, diffs, ratios")
        df["house_sales_lag1"] = df["house_sales"].shift(1)
        df["house_sales_lag2"] = df["house_sales"].shift(2)
        df["house_sales_lag3"] = df["house_sales"].shift(3)
        df["house_sales_lag4"] = df["house_sales"].shift(4)
        df["house_sales_lag16"] = df["house_sales"].shift(16)  # 4 years ago (4 quarters × 4 years)
        
        # Rolling means (windows in quarters: 4=1yr, 16=4yr, 40=10yr)
        df["house_sales_rolling_mean_1y"] = df["house_sales"].rolling(window=4, min_periods=1).mean()
        df["house_sales_rolling_mean_4y"] = df["house_sales"].rolling(window=16, min_periods=1).mean()
        df["house_sales_rolling_mean_10y"] = df["house_sales"].rolling(window=40, min_periods=1).mean()
        
        # Differences
        df["house_sales_diff_lag1_minus_lag2"] = df["house_sales_lag1"] - df["house_sales_lag2"]
        df["house_sales_diff_lag1_minus_lag4"] = df["house_sales_lag1"] - df["house_sales_lag4"]
        
        # Ratios (avoid division by zero)
        df["house_sales_ratio_lag1_over_lag2"] = df["house_sales_lag1"] / df["house_sales_lag2"].replace(0, np.nan)
        df["house_sales_ratio_lag1_over_lag4"] = df["house_sales_lag1"] / df["house_sales_lag4"].replace(0, np.nan)
        
        # --- Rolling Means for Other Variables ---
        variables = ["hpi", "house_stock", "residential_investment", "ocr", "cpi", "gdp"]
        
        for var in variables:
            if var in df.columns:
                print(f"  - {var}: rolling means")
                df[f"{var}_rolling_mean_1y"] = df[var].rolling(window=4, min_periods=1).mean()
                df[f"{var}_rolling_mean_4y"] = df[var].rolling(window=16, min_periods=1).mean()
                df[f"{var}_rolling_mean_10y"] = df[var].rolling(window=40, min_periods=1).mean()
        
        print(f"\nTRANSFORM: Created {len(df.columns):,} total columns")
        print(f"           Original: 8, Features: {len(df.columns) - 8}")
        
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