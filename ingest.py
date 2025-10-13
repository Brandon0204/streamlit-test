# ingest.py — Auckland daily air-quality ETL
# Order: load → monitor⟷trend → pivot → grid (town×full date range) → merge lagged features → missingness
from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple, List, Dict

import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client


class AirQualityETL:
    """
    Minimal API:
      etl = AirQualityETL()
      raw = etl.extract()             # {'monitoring': df, 'trends': df, 'features': df}
      final_df = etl.transform(raw)   # single DataFrame with pm + trend + lagged features
      etl.load(final_df)              # upsert to tables
    """

    # CSV feature positions (0-based) and constants
    CSV_RAIN_COL_IDX = 3       # column D in rain CSV
    CSV_RADIATION_COL_IDX = 4  # column E in radiation CSV
    CSV_WIND_COL_IDX = 5       # column F in wind CSV
    CSV_DATE_COL_NAME = "Observation time UTC"
    LOCAL_TZ = "Pacific/Auckland"

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket: Optional[str] = None,
        folder: Optional[str] = None,
    ):
        load_dotenv()
        self.supabase_url = supabase_url or os.environ["SUPABASE_URL"]
        self.supabase_key = supabase_key or os.environ["SUPABASE_ANON_KEY"]
        self.bucket = (bucket or os.getenv("BUCKET", "project3bucket")).strip()
        self.folder = (folder or os.getenv("FOLDER", "raw_files")).strip()

        # Anon (reads) and service-role (writes)
        self.sb: Client = create_client(self.supabase_url, self.supabase_key)
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.sb_rw: Client = create_client(self.supabase_url, self.service_role_key) if self.service_role_key else self.sb

    # ----------------- helpers -----------------

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out.columns = (
            out.columns.str.strip().str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        return out

    @staticmethod
    def _normalize_indicator(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
             .str.strip()
             .str.lower()
             .str.replace(".", "_", regex=False)
             .replace({"pm2.5": "pm2_5"})
        )

    @staticmethod
    def _normalize_key(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.lower()
        return s.str.replace(r"\s+", " ", regex=True)

    @staticmethod
    def _is_auckland(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().eq("auckland")

    @staticmethod
    def _find_sheet(xl: pd.ExcelFile, name: str) -> Optional[str]:
        want = name.strip().lower()
        look = {s.strip().lower(): s for s in xl.sheet_names}
        return look.get(want)

    def _list_paths(self, exts: tuple) -> List[str]:
        """List storage paths under folder for given extensions."""
        paths: List[str] = []
        for cand in {self.folder, self.folder.rstrip('/') + '/'}:
            items = self.sb.storage.from_(self.bucket).list(path=cand) or []
            for it in items:
                nm = (it.get("name") or "").strip()
                if nm.lower().endswith(exts):
                    paths.append(f"{cand.rstrip('/')}/{nm}")
        return sorted(set(paths))

    def _read_workbook(self, file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bio = BytesIO(file_bytes)
        xl = pd.ExcelFile(bio, engine="openpyxl")

        mon_sheet = self._find_sheet(xl, "Monitoring dataset") or self._find_sheet(xl, "Monitoring dataset ")
        trd_sheet = self._find_sheet(xl, "Ten-year trends")

        mon = pd.read_excel(xl, sheet_name=mon_sheet) if mon_sheet else pd.DataFrame()
        trd = pd.read_excel(xl, sheet_name=trd_sheet) if trd_sheet else pd.DataFrame()

        if not mon.empty:
            mon = self._standardize_columns(mon)
            if "sample_date" in mon:
                mon["sample_date"] = pd.to_datetime(mon["sample_date"], errors="coerce").dt.normalize()
            if "concentration_ug_m3" in mon:
                mon["concentration_ug_m3"] = pd.to_numeric(mon["concentration_ug_m3"], errors="coerce")

        if not trd.empty:
            trd = self._standardize_columns(trd)
            ren = {}
            for c in trd.columns:
                if c.startswith("10_year_trend_date"):
                    ren[c] = "ten_year_trend_date"
                if c.startswith("10_year_trend_result"):
                    ren[c] = "ten_year_trend_result"
            if ren:
                trd = trd.rename(columns=ren)
            if "ten_year_trend_date" in trd:
                trd["ten_year_trend_date"] = pd.to_datetime(trd["ten_year_trend_date"], errors="coerce", format="%Y")
        return mon, trd

    # ---------- CSV features ----------

    def _extract_feature_from_csv(self, file_bytes: bytes, value_col_index: int, feature_name: str) -> pd.DataFrame:
        """
        Read a CSV, use exact 'Observation time UTC' (UTC), convert to Pacific/Auckland DATE,
        and take the numeric column at value_col_index.
        """
        df = pd.read_csv(BytesIO(file_bytes))
        if df.empty or self.CSV_DATE_COL_NAME not in df.columns:
            return pd.DataFrame(columns=["date", feature_name])

        dt_utc = pd.to_datetime(df[self.CSV_DATE_COL_NAME], utc=True, errors="coerce")
        dt_local = dt_utc.dt.tz_convert(self.LOCAL_TZ).dt.normalize()
        dt_local = dt_local.dt.tz_localize(None) 

        if value_col_index >= len(df.columns):
            return pd.DataFrame(columns=["date", feature_name])

        values = pd.to_numeric(df.iloc[:, value_col_index], errors="coerce")

        out = pd.DataFrame({"date": dt_local, feature_name: values})
        out = (
            out.dropna(subset=["date"])
               .drop_duplicates(subset=["date"])
               .sort_values("date")
               .reset_index(drop=True)
        )
        return out

    def _read_csv_features(self) -> pd.DataFrame:
        """
        Build per-date features:
          rainfall (Rain CSV col D), radiation (Radiation CSV col E), speed (Wind CSV col F)
        Merge on local NZ date.
        """
        paths_csv = self._list_paths((".csv",))
        if not paths_csv:
            return pd.DataFrame(columns=["date", "rainfall", "radiation", "speed"])

        parts = []
        for p in paths_csv:
            nm = p.lower()
            try:
                b = self.sb.storage.from_(self.bucket).download(p)
            except Exception:
                continue

            if "rain" in nm and "daily" in nm:
                parts.append(self._extract_feature_from_csv(b, self.CSV_RAIN_COL_IDX, "rainfall"))
            elif "radiation" in nm and "daily" in nm:
                parts.append(self._extract_feature_from_csv(b, self.CSV_RADIATION_COL_IDX, "radiation"))
            elif "wind" in nm and "daily" in nm:
                parts.append(self._extract_feature_from_csv(b, self.CSV_WIND_COL_IDX, "speed"))

        if not parts:
            return pd.DataFrame(columns=["date", "rainfall", "radiation", "speed"])

        features = parts[0]
        for i in range(1, len(parts)):
            features = features.merge(parts[i], on="date", how="outer")

        return features.sort_values("date").reset_index(drop=True)

    # ----------------- public: extract + transform -----------------

    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Returns Auckland-only raw tables + CSV features:
          {'monitoring': df, 'trends': df, 'features': df}
        """
        # Excel
        paths_xls = self._list_paths((".xlsx", ".xls"))
        mon_frames, trd_frames = [], []
        for p in paths_xls:
            b = self.sb.storage.from_(self.bucket).download(p)
            mon, trd = self._read_workbook(b)
            if not mon.empty:
                mon_frames.append(mon)
            if not trd.empty:
                trd_frames.append(trd)

        monitoring = pd.concat(mon_frames, ignore_index=True) if mon_frames else pd.DataFrame()
        trends = pd.concat(trd_frames, ignore_index=True) if trd_frames else pd.DataFrame()

        if "region" in monitoring:
            monitoring = monitoring.loc[self._is_auckland(monitoring["region"])]
        if "region" in trends:
            trends = trends.loc[self._is_auckland(trends["region"])]

        # CSV features
        features = self._read_csv_features()

        print(f"EXTRACT: monitoring={len(monitoring):,} | trends={len(trends):,} | features_dates={len(features):,} (Auckland)")
        return {"monitoring": monitoring.reset_index(drop=True),
                "trends": trends.reset_index(drop=True),
                "features": features.reset_index(drop=True)}

    def transform(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Output columns:
        town, lat, lon, date, year, trend_pm10, trend_pm2_5, pm10, pm2_5,
        rainfall_lag1, radiation_lag1, speed_lag1

        Process (town-level):
        1) monitoring ⟷ trends (canonical town)
        2) aggregate monitoring to town+indicator+date (avg across sites) + town lat/lon (mean)
        3) pivot pm wide
        4) build town×date grid from union of pm dates and (truncated) feature dates
        5) left-join pm/trend; left-join lag-1 features
        6) print missingness per year per town
        """
        monitoring = raw.get("monitoring", pd.DataFrame()).copy()
        trends     = raw.get("trends", pd.DataFrame()).copy()
        features   = raw.get("features", pd.DataFrame()).copy()

        # ---------- 1) monitoring ⟷ trends (make town canonical on BOTH)
        if not monitoring.empty:
            keep = ["town", "latitude", "longitude", "indicator", "sample_date", "concentration_ug_m3"]
            m = monitoring[[c for c in keep if c in monitoring.columns]].copy()

            m["indicator"]   = self._normalize_indicator(m.get("indicator", pd.Series(dtype="object")))
            m["town_orig"]   = m.get("town", pd.Series(dtype="object"))
            m["town_canon"]  = self._canon_series(m["town_orig"])
            m["date"]        = pd.to_datetime(m.get("sample_date", pd.NaT), errors="coerce").dt.normalize()  # tz-naive
            m["conc_ug_m3"]  = pd.to_numeric(m.get("concentration_ug_m3", pd.NA), errors="coerce")
            m["lat"]         = pd.to_numeric(m.get("latitude"), errors="coerce")
            m["lon"]         = pd.to_numeric(m.get("longitude"), errors="coerce")
            m = m.dropna(subset=["date", "indicator"])  # keep if we have a date+indicator
        else:
            m = pd.DataFrame(columns=["town_canon","lat","lon","indicator","date","conc_ug_m3"])

        if not trends.empty:
            t = trends.copy()
            t["indicator"]  = self._normalize_indicator(t.get("indicator", pd.Series(dtype="object")))
            t["town_canon"] = self._canon_series(t.get("town", pd.Series(dtype="object")))
            # derive year
            if "ten_year_trend_date" in t:
                year = pd.to_datetime(t["ten_year_trend_date"], errors="coerce").dt.year.astype("Int64")
            elif "year" in t:
                year = pd.to_numeric(t["year"], errors="coerce").astype("Int64")
            else:
                year = pd.to_numeric(
                    t.astype(str).agg(" ".join, axis=1).str.extract(r"(\d{4})", expand=False),
                    errors="coerce"
                ).astype("Int64")
            trend_text = (
                t["ten_year_trend_result"] if "ten_year_trend_result" in t
                else t.get("trend", pd.Series(pd.NA, index=t.index))
            )
            trends_yearly = (
                pd.DataFrame({
                    "town_canon": t["town_canon"],
                    "indicator":  t["indicator"],
                    "year":       year,
                    "trend":      trend_text,
                })
                .dropna(subset=["town_canon","indicator","year"])
                .drop_duplicates(subset=["town_canon","indicator","year"])
                .reset_index(drop=True)
            )
        else:
            trends_yearly = pd.DataFrame(columns=["town_canon","indicator","year","trend"])

        # Attach YEAR to monitoring (for later join after aggregation)
        if not m.empty:
            m["year"] = m["date"].dt.year.astype("Int64")

        # ---------- 2) aggregate monitoring from sites → town level
        # Strategy: for each (town_canon, indicator, date), take the AVERAGE of conc_ug_m3.
        # If you prefer 'max', change .mean() to .max() below.
        if not m.empty:
            # site→town lat/lon: mean per town across sites (stable representative)
            town_coords = (
                m.groupby("town_canon", dropna=False)[["lat","lon"]]
                .mean(numeric_only=True)
                .reset_index()
            )

            agg = (
                m.groupby(["town_canon","indicator","date"], dropna=False)["conc_ug_m3"]
                .mean()  # <-- change to .max() if you want 'max across sites'
                .reset_index()
                .rename(columns={"conc_ug_m3":"conc"})
            )
            agg["year"] = agg["date"].dt.year.astype("Int64")

            # join yearly trend (now by canonical town)
            if not trends_yearly.empty:
                agg = agg.merge(trends_yearly, on=["town_canon","indicator","year"], how="left")
            else:
                agg["trend"] = pd.NA

            # attach representative coords
            agg = agg.merge(town_coords, on="town_canon", how="left")
        else:
            agg = pd.DataFrame(columns=["town_canon","indicator","date","year","conc","trend","lat","lon"])

        # ---------- 3) pivot pm wide (town-level)
        if not agg.empty:
            conc_wide = (
                agg.pivot_table(
                    index=["town_canon","lat","lon","date","year"],
                    columns="indicator",
                    values="conc",
                    aggfunc="first",
                )
                .rename_axis(None, axis=1)
                .reset_index()
            )
            trend_wide = (
                agg.pivot_table(
                    index=["town_canon","lat","lon","date","year"],
                    columns="indicator",
                    values="trend",
                    aggfunc=lambda x: x.dropna().iloc[0] if len(x.dropna()) else pd.NA,  # first non-null
                )
                .rename_axis(None, axis=1)
                .add_prefix("trend_")
                .reset_index()
            )
        else:
            conc_wide  = pd.DataFrame(columns=["town_canon","lat","lon","date","year","pm10","pm2_5"])
            trend_wide = pd.DataFrame(columns=["town_canon","lat","lon","date","year","trend_pm10","trend_pm2_5"])

        for c in ["pm10","pm2_5"]:
            if c not in conc_wide.columns:
                conc_wide[c] = pd.NA
        for c in ["trend_pm10","trend_pm2_5"]:
            if c not in trend_wide.columns:
                trend_wide[c] = pd.NA

        pm_df = conc_wide.merge(
            trend_wide[["town_canon","lat","lon","date","year","trend_pm10","trend_pm2_5"]],
            on=["town_canon","lat","lon","date","year"],
            how="left"
        ) if not conc_wide.empty else trend_wide.copy()

        # ---------- 4) normalize & TRUNCATE features; build union grid
        # features['date'] is already local-naive from extract(); ensure and truncate
        if not features.empty:
            features = features.copy()
            # force to datetime; handle tz-aware by converting to LOCAL_TZ then strip tz
            s_dt = pd.to_datetime(features["date"], errors="coerce")
            if getattr(getattr(s_dt, "dt", None), "tz", None) is not None:
                # tz-aware: convert to local day, drop tz
                s_dt = s_dt.dt.tz_convert(self.LOCAL_TZ).dt.normalize().dt.tz_localize(None)
            else:
                # tz-naive: just normalize
                s_dt = s_dt.dt.normalize()
            features["date"] = s_dt
            features = features.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")

            # Truncate features to >= monitoring min date (monitoring is tz-naive)
            min_monitor_date = pd.to_datetime(pm_df["date"], errors="coerce").min() if not pm_df.empty else None
            if pd.notna(min_monitor_date):
                features = features.loc[features["date"] >= min_monitor_date]
        else:
            features = pd.DataFrame(columns=["date","rainfall","radiation","speed"])

        # towns ref (canonical)
        towns = (
            pm_df[["town_canon","lat","lon"]].drop_duplicates()
            if not pm_df.empty else pd.DataFrame(columns=["town_canon","lat","lon"])
        )

        # union of dates (both tz-naive)
        pm_dates_idx = (
            pd.DatetimeIndex(pd.to_datetime(pm_df["date"], errors="coerce").dropna().unique())
            if ("date" in pm_df.columns and not pm_df.empty) else pd.DatetimeIndex([])
        )
        feat_dates_idx = (
            pd.DatetimeIndex(pd.to_datetime(features["date"], errors="coerce").dropna().unique())
            if ("date" in features.columns and not features.empty) else pd.DatetimeIndex([])
        )
        all_dates_idx = pm_dates_idx.union(feat_dates_idx)

        if len(all_dates_idx) > 0 and not towns.empty:
            date_range = pd.date_range(all_dates_idx.min(), all_dates_idx.max(), freq="D")
            grid = (
                towns.assign(_k=1)
                .merge(pd.DataFrame({"date": date_range, "_k": 1}), on="_k", how="outer")
                .drop(columns="_k")
            )
            grid["year"] = grid["date"].dt.year.astype("Int64")
        else:
            grid = (pm_df[["town_canon","lat","lon","date","year"]].copy()
                    if not pm_df.empty else
                    pd.DataFrame(columns=["town_canon","lat","lon","date","year"]))

        # join town-level pm/trend
        merged = grid.merge(
            pm_df[["town_canon","lat","lon","date","year","pm10","pm2_5","trend_pm10","trend_pm2_5"]],
            on=["town_canon","lat","lon","date","year"],
            how="left"
        )

        # Lag-1 features (value on D = original D-1)
        if not features.empty:
            feat_lag1 = features.rename(columns={
                "rainfall":  "rainfall_lag1",
                "radiation": "radiation_lag1",
                "speed":     "speed_lag1"
            }).assign(date=features["date"] + pd.Timedelta(days=1))
        else:
            feat_lag1 = pd.DataFrame(columns=["date","rainfall_lag1","radiation_lag1","speed_lag1"])

        merged = merged.merge(feat_lag1, on="date", how="left")

        # Final tidy: expose canonical town as 'town'
        final_df = (
            merged.rename(columns={"town_canon":"town"})[[
                "town","lat","lon","date","year",
                "trend_pm10","trend_pm2_5","pm10","pm2_5",
                "rainfall_lag1","radiation_lag1","speed_lag1"
            ]]
            .sort_values(["town","date"])
            .reset_index(drop=True)
        )

        print(f"TRANSFORM: final_df rows = {len(final_df):,}")

        # ---------- 6) Missingness per year per town
        self._print_missingness_by_year_town(final_df)

        return final_df

    @staticmethod
    def _canonicalize_town_name(name: str) -> str:
        """
        Map variant station/town labels to a single canonical town name.
        Current rule: any 'Queen Street' variant -> 'Auckland City Centre'.
        """
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        s = str(name).strip().lower()
        s = pd.Series([s]).str.replace(r"\s+", " ", regex=True).iloc[0]

        # aliases → canonical
        aliases = {
            "queen street": "auckland city centre",
            "queen st": "auckland city centre",
            "queen st.": "auckland city centre",
            "queen-street": "auckland city centre",
            "aucklan city centra": "auckland city centre",  # common typo reported
        }
        return aliases.get(s, s)

    def _canon_series(self, s: pd.Series) -> pd.Series:
        """Vectorized canonicalization wrapper."""
        return s.astype("object").map(self._canonicalize_town_name)



    # ----------------- reporting -----------------

    def _print_missingness_by_year_town(self, df: pd.DataFrame) -> None:
        """
        Print missingness per YEAR per TOWN for key columns.
        Produces a compact table for each year: one row per town with % missing.
        """
        if df.empty or "year" not in df or "town" not in df:
            print("\n[missingness per year per town] (no data)")
            return

        cols = [c for c in ["pm10","pm2_5","trend_pm10","trend_pm2_5",
                            "rainfall_lag1","radiation_lag1","speed_lag1"]
                if c in df.columns]

        if not cols:
            print("\n[missingness per year per town] (no target columns)")
            return

        # Group by year, town
        grp = df.groupby(["year", "town"], dropna=False)
        rows = []
        for (y, t), sub in grp:
            total = len(sub)
            rec = {
                "year": int(y) if pd.notna(y) else None,
                "town": t,
                "rows": int(total)
            }
            for c in cols:
                miss = sub[c].isna().sum()
                rec[f"{c}_pct_missing"] = round((miss / total * 100.0), 2) if total else None
            rows.append(rec)

        miss_df = pd.DataFrame(rows).sort_values(["year", "town"]).reset_index(drop=True)

        # Pretty print per year
        print("\n[missingness per year per town]  (percent missing per column)")
        for year_val, sub in miss_df.groupby("year", dropna=False):
            hdr = f"Year: {year_val if year_val is not None else 'NULL'}"
            print(f"\n{hdr}  (towns={sub['town'].nunique()}, rows={int(sub['rows'].sum())})")

            # Build a compact table per year
            display_cols = ["town", "rows"] + [f"{c}_pct_missing" for c in cols]
            # For stable, readable order, rename columns to shorter labels
            rename_map = {f"{c}_pct_missing": c for c in cols}
            table = sub[display_cols].rename(columns=rename_map)
            # Align columns
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(table.to_string(index=False))

    # ----------------- write to DB -----------------

    @staticmethod
    def _enforce_types_for_db(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize types for DB + JSON.
        """
        out = df.copy()

        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        if "year" in out.columns:
            y = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
            out["year"] = y.where(~y.isna(), None)

        float_cols = [c for c in ["lat","lon","pm10","pm2_5","rainfall_lag1","radiation_lag1","speed_lag1"] if c in out.columns]
        for c in float_cols:
            s = pd.to_numeric(out[c], errors="coerce").astype(float)
            s[~np.isfinite(s)] = np.nan
            out[c] = s.where(~s.isna(), None)

        for c in [col for col in ["town","trend_pm10","trend_pm2_5"] if col in out.columns]:
            out[c] = out[c].astype(str)
            out.loc[out[c].isin(["nan","NaN","None"]), c] = None

        out = out.where(pd.notna(out), None)
        return out

    @staticmethod
    def _to_postgrest_records(df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame rows to plain Python scalars for PostgREST."""
        def py_scalar(v):
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                fv = float(v)
                return fv if math.isfinite(fv) else None
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v

        cols = df.columns.tolist()
        records: List[Dict] = []
        for row in df.itertuples(index=False, name=None):
            records.append({c: py_scalar(v) for c, v in zip(cols, row)})
        return records

    def _postgrest_upsert(self, table: str, df: pd.DataFrame) -> None:
        """Deterministic upsert via service-role client."""
        if df.empty:
            return
        df = self._enforce_types_for_db(df)
        records = self._to_postgrest_records(df)

        def chunks(lst, n=1000):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        for batch in chunks(records, 1000):
            self.sb_rw.table(table).upsert(
                batch,
                on_conflict="town,date",
                returning="minimal"
            ).execute()

    def load(self, final_df: pd.DataFrame) -> None:
        """
        Upsert to two existing tables; include lagged features.
        Requires nullable float columns rainfall_lag1/radiation_lag1/speed_lag1 in both tables.
        """
        cols10 = ["town","lat","lon","date","year","trend_pm10","pm10","rainfall_lag1","radiation_lag1","speed_lag1"]
        cols25 = ["town","lat","lon","date","year","trend_pm2_5","pm2_5","rainfall_lag1","radiation_lag1","speed_lag1"]

        df10 = final_df[cols10].copy()
        df25 = final_df[cols25].copy()
        self._postgrest_upsert("clean_auckland_pm10", df10)
        self._postgrest_upsert("clean_auckland_pm2_5", df25)
        print(f"WRITE: upserted {len(df10):,} rows -> public.clean_auckland_pm10")
        print(f"WRITE: upserted {len(df25):,} rows -> public.clean_auckland_pm2_5")

    def reset_tables(self) -> None:
        """Delete all rows from both tables via PostgREST (TRUNCATE-like)."""
        self.sb_rw.table("clean_auckland_pm10").delete().neq("town", "__never__").execute()
        self.sb_rw.table("clean_auckland_pm2_5").delete().neq("town", "__never__").execute()
        print("RESET: both tables cleared.")


# ----------------------------- CLI -----------------------------

def main():
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)

    etl = AirQualityETL()
    raw = etl.extract()
    final_df = etl.transform(raw)

    # print("\n[final_df] columns:", list(final_df.columns))
    # etl.load(final_df)


if __name__ == "__main__":
    main()










# test.py - House Data Viewer, Downloader & EDA
import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

from ets_trainer import run_experiment as run_ets
from xgboost_trainer import run_experiment as run_xgb
from catboost_trainer import run_experiment as run_cat

st.set_page_config(page_title="House Price Data", layout="wide")
st.title("House Price Data Viewer")

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
    options=["clean_house", "feature_house"],
    index=0
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
    
    # Preview
    st.markdown("### Data Preview")
    preview_rows = st.slider("Preview rows", 5, 100, 20, 5)
    st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)
    
    # Column info
    with st.expander("Column Information"):
        st.write(f"**Total columns:** {len(df.columns)}")
        st.write(f"**Columns:** {', '.join(df.columns)}")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write(f"**Numeric columns:** {len(numeric_cols)}")
        
        date_range = ""
        if "quarter" in df.columns:
            date_range = f"{df['quarter'].min().date()} to {df['quarter'].max().date()}"
            st.write(f"**Date range:** {date_range}")
    
    # Download section
    st.markdown("### Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {table_name}.csv ({len(df):,} rows)",
        data=csv,
        file_name=f"{table_name}.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

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
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

# -----------------------------
# Model Experimentation Section
# -----------------------------
if not df.empty and table_name == "feature_house" and "hpi_growth" in df.columns:
    st.divider()
    st.markdown("## 🤖 Model Experimentation")
    st.caption("Train different models with custom parameters and feature selection")
    
    # Model selector
    model_type = st.selectbox(
        "Select Model Type",
        options=["CatBoost", "XGBoost", "Exponential Smoothing (ETS)"],
        help="Choose which model to train"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"#### {model_type} Parameters")
        st.caption("Enter parameters as JSON")
        
        # Default parameters for each model
        if model_type == "CatBoost":
            default_params = """{
    "depth": 3,
    "iterations": 100,
    "learning_rate": 0.05
}"""
        elif model_type == "XGBoost":
            default_params = """{
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.05
}"""
        else:  # ETS
            default_params = """{
    "trend": "add",
    "seasonal": "add",
    "seasonal_periods": 4
}"""
        
        params_json = st.text_area(
            "Parameters (JSON format)",
            value=default_params,
            height=150,
            help="Enter model parameters in JSON format"
        )
    
    with col2:
        if model_type != "Exponential Smoothing (ETS)":
            st.markdown("#### Feature Selection")
            st.caption("Select features for training")
            
            # Get available features (exclude non-feature columns and leakage columns)
            exclude_cols = ["quarter", "year", "quarter_num", "hpi_growth", 
                           "house_sales", "hpi", "house_stock"]
            available_features = [col for col in df.select_dtypes(include='number').columns 
                                 if col not in exclude_cols]
            
            # Default selected features (top performers)
            default_features = [
                'hpi_growth_lag1',
                'hpi_growth_lag2', 
                'hpi_growth_lag4',
                'hpi_growth_rolling_mean_1y',
                'house_sales_lag1',
                'house_sales_rolling_mean_1y',
                'ocr_rolling_mean_1y',
                'cpi_rolling_mean_1y'
            ]
            # Only use defaults that exist
            default_features = [f for f in default_features if f in available_features]
            
            selected_features = st.multiselect(
                "Select Features",
                options=available_features,
                default=default_features,
                help=f"Choose from {len(available_features)} available features"
            )
            
            st.info(f"✅ {len(selected_features)} features selected")
        else:
            st.markdown("#### Model Info")
            st.info("📊 ETS is a univariate time series model - no feature selection needed")
            selected_features = []
    
    # Missing value strategy
    st.markdown("---")
    missing_strategy = st.radio(
        "Missing Value Strategy",
        options=["drop", "impute"],
        horizontal=True,
        help="How to handle missing values in selected features: drop rows or impute with median"
    )
    
    # Train button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        if model_type != "Exponential Smoothing (ETS)" and not selected_features:
            st.error("Please select at least one feature!")
        else:
            # Validate JSON
            try:
                params_dict = json.loads(params_json)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
                st.stop()
            
            # Run experiment
            with st.spinner(f"Training {model_type} model..."):
                try:
                    import io
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    output_buffer = io.StringIO()
                    error_buffer = io.StringIO()
                    
                    # Import and run appropriate model
                    if model_type == "CatBoost":
                        from catboost_trainer import run_experiment
                        
                        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                            results = run_experiment(
                                best_params=params_dict,
                                feature_list=selected_features,
                                missing_strategy=missing_strategy
                            )
                    
                    elif model_type == "XGBoost":
                        from xgboost_trainer import run_experiment
                        
                        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                            results = run_experiment(
                                best_params=params_dict,
                                feature_list=selected_features,
                                missing_strategy=missing_strategy
                            )
                    
                    else:  # ETS
                        from ets_trainer import run_experiment
                        
                        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                            results = run_experiment(
                                best_params=params_dict,
                                missing_strategy=missing_strategy
                            )
                    
                    output = output_buffer.getvalue()
                    errors = error_buffer.getvalue()
                    
                    # Display results
                    st.success(f"✅ {model_type} training completed successfully!")
                    
                    # Metrics
                    st.markdown("### 📊 Model Performance Metrics")
                    
                    metrics = results['metrics']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                        st.metric("Train R²", f"{metrics['train_r2']:.4f}")
                    with col2:
                        st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
                        st.metric("Train RMSE", f"{metrics['train_rmse']:.4f}")
                    with col3:
                        st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
                        st.metric("Train MAE", f"{metrics['train_mae']:.4f}")
                    
                    # Metrics table
                    st.markdown("#### Detailed Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
                        'Train': [
                            f"{metrics['train_mse']:.4f}",
                            f"{metrics['train_rmse']:.4f}",
                            f"{metrics['train_mae']:.4f}",
                            f"{metrics['train_r2']:.4f}"
                        ],
                        'Test': [
                            f"{metrics['test_mse']:.4f}",
                            f"{metrics['test_rmse']:.4f}",
                            f"{metrics['test_mae']:.4f}",
                            f"{metrics['test_r2']:.4f}"
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Model-specific visualizations
                    if model_type in ["CatBoost", "XGBoost"]:
                        # SHAP Summary Plot
                        st.markdown("### 🎯 SHAP Feature Importance")
                        st.caption("SHAP values show how each feature impacts model predictions")
                        
                        shap_fig = results['shap_figure']
                        st.pyplot(shap_fig)
                    
                    else:  # ETS
                        # Forecast Plot
                        st.markdown("### 📈 Forecast Visualization")
                        st.caption("Time series forecast vs actual values")
                        
                        forecast_fig = results['forecast_figure']
                        st.pyplot(forecast_fig)
                    
                    # Training output
                    with st.expander("📝 View Training Output"):
                        st.code(output if output else "No output")
                    
                    if errors:
                        with st.expander("⚠️ View Warnings"):
                            st.code(errors)
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

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
                from ingest_h import HousePriceETL
                
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
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
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

with col2:
    st.markdown("#### Generate Features")
    st.caption("Refresh feature_house table from clean_house")
    if st.button("Run Feature Engineering", use_container_width=True):
        with st.spinner("Running feature engineering..."):
            try:
                from feature_engineering import HouseFeatureEngineering
                
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
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
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

st.markdown("---")
st.caption("Use the buttons above to refresh data from source files")