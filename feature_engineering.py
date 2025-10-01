# feature_engineering.py
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()
# -----------------------------
# Supabase connection
# -----------------------------
def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in env")
    return create_client(url, key)

supabase = get_supabase_client()

# -----------------------------
# Data loader with pagination
# -----------------------------
def load_table_all(table: str, page_size: int = 1000) -> pd.DataFrame:
    """Fetch the entire table by paging through Supabase."""
    frames = []
    offset = 0
    while True:
        res = supabase.table(table).select("*").range(offset, offset + page_size - 1).execute()
        data = res.data or []
        if not data:
            break
        frames.append(pd.DataFrame(data))
        offset += len(data)
        if len(data) < page_size:
            break  # last page
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def summarize_df(df: pd.DataFrame, table: str):
    print(f"\n===== {table} =====")
    if df.empty:
        print("No rows found")
        return
    print(f"Total rows fetched: {len(df):,}")
    print("\nPreview:")
    print(df.head())
    print("\nDescribe (numeric):")
    print(df.describe(include="number"))
    print("\nInfo:")
    df.info()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tables = ["clean_auckland_pm10", "clean_auckland_pm2_5"]
    for t in tables:
        df = load_table_all(t)
        summarize_df(df, t)
