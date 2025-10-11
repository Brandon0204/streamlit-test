from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Dict
import numpy as np
import pandas as pd
from supabase import create_client, Client

from typing import Optional
import os
from supabase import create_client, Client
import streamlit as st

def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read nested keys from streamlit secrets as 'section.key' (e.g., 'supabase.url').
    Falls back to env var using LAST segment uppercased with SUPABASE_ prefix when sensible.
    """
    try:
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

    # Try environment variables
    ENV_MAP = {
        "supabase.url": "SUPABASE_URL",
        "supabase.key": "SUPABASE_ANON_KEY",
        "supabase.service_role_key": "SUPABASE_SERVICE_ROLE_KEY",
        "supabase.bucket": "SUPABASE_BUCKET",
        "paths.excel": "HOUSE_EXCEL_PATH",
        "paths.ocr_csv": "HOUSE_OCR_CSV_PATH",
        "paths.cpi_csv": "HOUSE_CPI_CSV_PATH",
        "paths.gdp_csv": "HOUSE_GDP_CSV_PATH",
    }
    env_name = ENV_MAP.get(path)
    if env_name:
        v = os.getenv(env_name)
        if v and v.strip():
            return v.strip()

    return default

class HousePriceETL:
    """
    Minimal API:
      etl = HousePriceETL()
      raw = etl.extract()   # {"house": df, "ocr": df, "cpi": df, "gdp": df}
      tidy = etl.transform(raw)  # (stub)
      etl.load(tidy)             # (stub)
    """

    DEFAULT_BUCKET = "project3bucket"
    EXCEL_PATH = "house_row_files/hm10.xlsx"
    OCR_CSV_PATH = "house_row_files/OCRmodify.csv"
    CPI_CSV_PATH = "house_row_files/consumers-price-index-june-2025-quarter-seasonally-adjusted.csv"
    GDP_CSV_PATH = "house_row_files/gross-domestic-product-june-2025-quarter.csv"

    SHEET_NAME = "Data"
    SKIP_ROWS = 5

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket: Optional[str] = None,
        excel_path: Optional[str] = None,
        ocr_csv_path: Optional[str] = None,
        cpi_csv_path: Optional[str] = None,
        gdp_csv_path: Optional[str] = None,
        service_role_key: Optional[str] = None,
    ):
        
        self.supabase_url = (supabase_url
                             or _get_secret("supabase.url"))
        self.supabase_key = (supabase_key
                             or _get_secret("supabase.key"))
        self.service_role_key = (service_role_key
                                 or _get_secret("supabase.service_role_key"))

        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError("Supabase URL/Key missing. Provide via kwargs, st.secrets['supabase'], or env.")

        # Paths & bucket can also come from secrets/env
        self.bucket = (bucket
                       or _get_secret("supabase.bucket", self.DEFAULT_BUCKET))
        self.excel_path   = (excel_path   or _get_secret("paths.excel",   self.EXCEL_PATH))
        self.ocr_csv_path = (ocr_csv_path or _get_secret("paths.ocr_csv", self.OCR_CSV_PATH))
        self.cpi_csv_path = (cpi_csv_path or _get_secret("paths.cpi_csv", self.CPI_CSV_PATH))
        self.gdp_csv_path = (gdp_csv_path or _get_secret("paths.gdp_csv", self.GDP_CSV_PATH))

        # Anon client 
        self.sb: Client = create_client(self.supabase_url, self.supabase_key)
        self.sb_rw: Client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.sb


    # ----------------- Helpers -----------------

    @staticmethod
    def _to_quarter_end(s: pd.Series) -> pd.Series:
        """Parse any reasonable date string to end-of-quarter date."""
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.to_period("Q").dt.end_time.dt.normalize()

    @staticmethod
    def _parse_period_to_quarter(s: pd.Series) -> pd.Series:
        """
        Convert Period format like '1926.03', '1926.06', '1926.09', '1926.12'
        to quarter-end dates.
        
        Format: YYYY.MM where MM indicates quarter:
          .03 -> Q1 (Mar 31)
          .06 -> Q2 (Jun 30)
          .09 -> Q3 (Sep 30)
          .12 -> Q4 (Dec 31)
        """
        def convert_period(val):
            try:
                if pd.isna(val):
                    return pd.NaT
                
                str_val = str(val)
                parts = str_val.split('.')
                if len(parts) != 2:
                    return pd.NaT
                
                year = int(parts[0])
                month = int(parts[1])
                
                quarter_map = {
                    3: f"{year}-03-31",
                    6: f"{year}-06-30",
                    9: f"{year}-09-30",
                    12: f"{year}-12-31"
                }
                
                if month not in quarter_map:
                    return pd.NaT
                
                return pd.to_datetime(quarter_map[month])
            except:
                return pd.NaT
        
        return s.apply(convert_period)

    @staticmethod
    def _coerce_numeric(col: pd.Series) -> pd.Series:
        """Coerce to float; harmless if already numeric."""
        return pd.to_numeric(col, errors="coerce")

    # ----------------- Extract Methods -----------------

    def _extract_house(self) -> pd.DataFrame:
        """Extract house metrics from Excel file."""
        blob_xlsx = self.sb.storage.from_(self.bucket).download(self.excel_path)
        df = pd.read_excel(
            BytesIO(blob_xlsx),
            sheet_name=self.SHEET_NAME,
            header=None,
            engine="openpyxl",
            skiprows=self.SKIP_ROWS,
            usecols="A:E",
            names=["quarter_raw", "house_sales", "hpi", "house_stock", "residential_investment"],
        )

        df["quarter"] = self._to_quarter_end(df["quarter_raw"])
        for col in ["house_sales", "hpi", "house_stock", "residential_investment"]:
            df[col] = self._coerce_numeric(df[col])

        return (
            df.drop(columns=["quarter_raw"])
              .dropna(subset=["quarter"])
              .sort_values("quarter")
              .reset_index(drop=True)
        )

    def _extract_ocr(self) -> pd.DataFrame:
        """Extract OCR data from CSV file."""
        try:
            blob_csv = self.sb.storage.from_(self.bucket).download(self.ocr_csv_path)
            df = pd.read_csv(BytesIO(blob_csv))

            # Standardize column names
            df.columns = [c.strip().lower() for c in df.columns]

            # Allow 'date' or 'quarter' for date column
            if "quarter" not in df.columns and "date" in df.columns:
                df = df.rename(columns={"date": "quarter"})

            if "quarter" not in df.columns or "ocr" not in df.columns:
                raise ValueError("OCR CSV must contain date column ('Date'/'quarter') and 'Ocr'/'ocr' value column.")

            df["quarter"] = self._to_quarter_end(df["quarter"])
            df["ocr"] = self._coerce_numeric(df["ocr"])

            return (
                df.dropna(subset=["quarter"])
                  .loc[:, ["quarter", "ocr"]]
                  .sort_values("quarter")
                  .reset_index(drop=True)
            )
        except Exception as e:
            print(f"[warn] OCR CSV not loaded ({e}).")
            return pd.DataFrame(columns=["quarter", "ocr"])

    def _extract_cpi(self) -> pd.DataFrame:
        """Extract CPI data from CSV file (filtered for CPIQ.SE9SA series)."""
        try:
            blob_cpi = self.sb.storage.from_(self.bucket).download(self.cpi_csv_path)
            df = pd.read_csv(BytesIO(blob_cpi))

            # Filter for specific series
            df = df[df["Series_reference"] == "CPIQ.SE9SA"].copy()

            # Select and rename columns
            df = df[["Period", "Data_value"]].copy()
            df.columns = ["period_raw", "cpi"]

            # Convert Period format to quarter-end dates
            df["quarter"] = self._parse_period_to_quarter(df["period_raw"])
            df["cpi"] = self._coerce_numeric(df["cpi"])

            return (
                df.drop(columns=["period_raw"])
                  .dropna(subset=["quarter"])
                  .loc[:, ["quarter", "cpi"]]
                  .sort_values("quarter")
                  .reset_index(drop=True)
            )
        except Exception as e:
            print(f"[warn] CPI CSV not loaded ({e}).")
            return pd.DataFrame(columns=["quarter", "cpi"])

    def _extract_gdp(self) -> pd.DataFrame:
        """Extract GDP data from CSV file (filtered for Gross National Expenditure series)."""
        try:
            blob_gdp = self.sb.storage.from_(self.bucket).download(self.gdp_csv_path)
            df = pd.read_csv(BytesIO(blob_gdp))

            # Filter for specific series reference, group, and series title
            df = df[
                (df["Series_reference"] == "SNEQ.SG02NAC00B21Z") &
                (df["Group"] == "Series, GDP(E), Nominal, Actual, Total") &
                (df["Series_title_1"] == "Gross National Expenditure")
            ].copy()

            # Select and rename columns
            df = df[["Period", "Data_value"]].copy()
            df.columns = ["period_raw", "gdp"]

            # Convert Period format to quarter-end dates
            df["quarter"] = self._parse_period_to_quarter(df["period_raw"])
            df["gdp"] = self._coerce_numeric(df["gdp"])

            return (
                df.drop(columns=["period_raw"])
                  .dropna(subset=["quarter"])
                  .loc[:, ["quarter", "gdp"]]
                  .sort_values("quarter")
                  .reset_index(drop=True)
            )
        except Exception as e:
            print(f"[warn] GDP CSV not loaded ({e}).")
            return pd.DataFrame(columns=["quarter", "gdp"])

    # ----------------- Public API -----------------

    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Load four separate DataFrames:
          - 'house': quarter, house_sales, hpi, house_stock, residential_investment
          - 'ocr'  : quarter, ocr
          - 'cpi'  : quarter, cpi
          - 'gdp'  : quarter, gdp
        """
        df_house = self._extract_house()
        df_ocr = self._extract_ocr()
        df_cpi = self._extract_cpi()
        df_gdp = self._extract_gdp()

        print(f"EXTRACT[house]: rows={len(df_house):,}  cols={list(df_house.columns)}")
        print(f"EXTRACT[ocr]  : rows={len(df_ocr):,}  cols={list(df_ocr.columns)}")
        print(f"EXTRACT[cpi]  : rows={len(df_cpi):,}  cols={list(df_cpi.columns)}")
        print(f"EXTRACT[gdp]  : rows={len(df_gdp):,}  cols={list(df_gdp.columns)}")

        return {
            "house": df_house,
            "ocr": df_ocr,
            "cpi": df_cpi,
            "gdp": df_gdp
        }

    def transform(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Join all dataframes on quarter (left join from house base).
        Returns a single consolidated DataFrame.
        """
        df_house = raw["house"].copy()
        df_ocr = raw["ocr"].copy()
        df_cpi = raw["cpi"].copy()
        df_gdp = raw["gdp"].copy()

        # Left join all dataframes on quarter
        df = df_house
        df = df.merge(df_ocr, on="quarter", how="left")
        df = df.merge(df_cpi, on="quarter", how="left")
        df = df.merge(df_gdp, on="quarter", how="left")

        print(f"\nTRANSFORM: Combined {len(df):,} rows × {len(df.columns)} columns")
        return df

    def load(self, data: pd.DataFrame) -> None:
        """
        Upsert combined house data to clean_house table.
        Simple overwrite approach using quarter as the unique key.
        """
        if data.empty:
            print("LOAD: No data to load.")
            return
        
        # Prepare data for database
        df = data.copy()
        
        # Convert date to string format for JSON serialization
        df["quarter"] = df["quarter"].dt.strftime("%Y-%m-%d")
        
        # Clean numeric columns: replace inf/-inf/NaN with None
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            # Replace inf/-inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace all NaN with None for JSON compatibility
        df = df.where(pd.notna(df), None)
        
        # Convert to records and ensure native Python types
        import json
        records = []
        for idx, row in enumerate(df.to_dict(orient="records")):
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
            
            # Test if this record is JSON serializable
            try:
                json.dumps(clean_row)
            except (ValueError, TypeError) as e:
                print(f"ERROR: Row {idx} cannot be serialized to JSON:")
                print(f"  Row data: {clean_row}")
                for k, v in clean_row.items():
                    print(f"  {k}: {v} (type: {type(v)})")
                raise
            
            records.append(clean_row)
        
        print(f"LOAD: Upserting {len(records):,} records to clean_house table...")
        
        # Upsert using quarter as conflict key
        self.sb_rw.table("clean_house").upsert(
            records,
            on_conflict="quarter",
            returning="minimal"
        ).execute()
        
        print(f"LOAD: Successfully upserted {len(records):,} rows to clean_house table.")


# ----------------------------- CLI -----------------------------

def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    etl = HousePriceETL()
    
    # Extract
    raw = etl.extract()
    
    datasets = ["house", "ocr", "cpi", "gdp"]
    for dataset in datasets:
        print(f"\n[{dataset} preview]")
        df = raw[dataset]
        if not df.empty:
            print(df.head(8).to_string(index=False))
            print(f"\nRange: {df['quarter'].min().date()} → {df['quarter'].max().date()}")
        else:
            print("(empty)")
    
    # Transform
    print("\n" + "="*80)
    combined = etl.transform(raw)
    
    # Preview combined data
    print("\n[Combined Data Preview]")
    print(combined.head(10).to_string(index=False))
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Date range: {combined['quarter'].min().date()} → {combined['quarter'].max().date()}")
    
    # Missingness report by year
    print("\n" + "="*80)
    print("[Missingness Report by Year]\n")
    
    # Extract year from quarter
    combined["year"] = combined["quarter"].dt.year
    
    # Columns to check for missingness (exclude quarter and year)
    data_columns = [col for col in combined.columns if col not in ["quarter", "year"]]
    
    # Calculate missingness by year
    missingness_by_year = []
    for year in sorted(combined["year"].unique()):
        year_data = combined[combined["year"] == year]
        total_quarters = len(year_data)
        
        year_stats = {"year": year, "quarters": total_quarters}
        for col in data_columns:
            missing_count = year_data[col].isna().sum()
            missing_pct = (missing_count / total_quarters) * 100 if total_quarters > 0 else 0
            year_stats[f"{col}_missing"] = missing_count
            year_stats[f"{col}_pct"] = missing_pct
        
        missingness_by_year.append(year_stats)
    
    df_missingness = pd.DataFrame(missingness_by_year)
    
    # Display report
    print("Missing Count per Year:")
    display_cols_count = ["year", "quarters"] + [f"{col}_missing" for col in data_columns]
    print(df_missingness[display_cols_count].to_string(index=False))
    
    print("\n\nMissing Percentage per Year:")
    display_cols_pct = ["year", "quarters"] + [f"{col}_pct" for col in data_columns]
    print(df_missingness[display_cols_pct].to_string(index=False, float_format=lambda x: f"{x:.1f}%"))
    
    # Overall missingness summary
    print("\n" + "="*80)
    print("[Overall Missingness Summary]\n")
    for col in data_columns:
        total_missing = combined[col].isna().sum()
        total_rows = len(combined)
        pct_missing = (total_missing / total_rows) * 100
        print(f"{col:25s}: {total_missing:4d} / {total_rows:4d} missing ({pct_missing:5.1f}%)")
    
    # Load
    print("\n" + "="*80)
    etl.load(combined)

if __name__ == "__main__":
    main()