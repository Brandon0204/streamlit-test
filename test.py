# test.py
import os
import io
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from supabase import create_client, Client

st.set_page_config(page_title="📊 Hi Project 3", layout="wide")
st.title("📊 Hi Project 3")

# -----------------------------
# Supabase connection helpers
# -----------------------------
def _get_supabase_client() -> Client:
    # Prefer Streamlit secrets, fallback to env vars
    url = st.secrets.get("supabase", {}).get("url") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("supabase", {}).get("key") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Supabase URL/Key not found. Add to `.streamlit/secrets.toml` or environment variables.")
        st.stop()
    return create_client(url, key)

SUPABASE = _get_supabase_client()

# Config
TABLE_CHOICES = ["clean_auckland_pm10", "clean_auckland_pm2_5"]
DEFAULT_PAGE_SIZE = 1000  # per API call

@st.cache_data(show_spinner=False)
def fetch_table_page(table: str, offset: int, limit: int) -> pd.DataFrame:
    """
    Fetch a single page from Supabase using range (0-indexed, inclusive).
    """
    start = offset
    end = offset + limit - 1
    # PostgREST range is inclusive
    res = SUPABASE.table(table).select("*").range(start, end).execute()
    data = res.data or []
    return pd.DataFrame(data)

@st.cache_data(show_spinner=True)
def fetch_table_all(table: str, page_size: int = DEFAULT_PAGE_SIZE, max_rows: int | None = None) -> pd.DataFrame:
    """
    Fetch the entire table (or up to max_rows) in pages. Cached.
    """
    frames = []
    offset = 0
    total_fetched = 0
    while True:
        df_page = fetch_table_page(table, offset=offset, limit=page_size)
        if df_page.empty:
            break
        frames.append(df_page)
        fetched = len(df_page)
        total_fetched += fetched
        offset += fetched
        if max_rows is not None and total_fetched >= max_rows:
            break
        if fetched < page_size:
            break  # last page
    if not frames:
        return pd.DataFrame()
    # Normalize columns/types a bit
    df = pd.concat(frames, ignore_index=True)
    # Try parse dates if a 'date' or 'ds' column exists
    for col in ("date", "ds"):
        if col in df.columns:
            with pd.option_context("mode.chained_assignment", None):
                df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Download full table as CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------
# Section 1: Supabase table viewer
# ---------------------------------
with st.expander("👀 View ingested tables (clean_auckland_pm10 / clean_auckland_pm2_5)", expanded=True):
    col_sel, col_actions = st.columns([2, 1])
    with col_sel:
        table = st.selectbox("Choose a table to view", TABLE_CHOICES, index=0)
        preview_rows = st.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)
    with col_actions:
        page_size = st.number_input("Fetch page size", min_value=100, max_value=5000, value=DEFAULT_PAGE_SIZE, step=100)

    # Fetch a preview (first page only)
    preview_df = fetch_table_page(table, offset=0, limit=preview_rows)
    if preview_df.empty:
        st.warning("No rows found.")
    else:
        st.write("### Data Preview")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        # Summary
        with st.expander("ℹ️ Quick summary"):
            st.write(f"Columns: `{', '.join(preview_df.columns)}`")
            num_cols = preview_df.select_dtypes(include="number").columns.tolist()
            st.write(f"Numeric columns: `{', '.join(num_cols) if num_cols else '—'}`")

        # Histogram for numeric column
        numeric_cols = preview_df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Choose a numeric column to plot (preview slice)", numeric_cols, key="tbl_num_col")
            fig, ax = plt.subplots()
            ax.hist(preview_df[col].dropna(), bins=20)
            ax.set_title(f"Histogram of {col} (preview)")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns found in preview for plotting.")

        # Download full table (will fetch all pages, cached)
        st.divider()
        st.write("#### Export")
        max_rows_opt = st.radio(
            "Download size",
            options=["All rows", "Limit rows…"],
            index=0,
            horizontal=True,
        )
        max_rows = None
        if max_rows_opt == "Limit rows…":
            max_rows = st.number_input("Max rows to download", min_value=1000, max_value=2_000_000, value=100_000, step=10_000)

        if st.button("Prepare CSV", type="primary", use_container_width=True):
            with st.spinner("Fetching data…"):
                full_df = fetch_table_all(table, page_size=page_size, max_rows=max_rows)
            if full_df.empty:
                st.warning("No data fetched.")
            else:
                st.success(f"Fetched {len(full_df):,} rows.")
                download_csv_button(full_df, filename=f"{table}.csv")
