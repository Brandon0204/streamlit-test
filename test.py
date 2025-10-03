# test.py - Simple House Data Viewer & Downloader
import os
import sys
import streamlit as st
import pandas as pd
from supabase import create_client, Client

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
                # Import locally to avoid loading heavy dependencies at startup
                from ingest_h import HousePriceETL
                
                # Capture output
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
                # Import locally
                from feature_engineering import HouseFeatureEngineering
                
                # Capture output
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