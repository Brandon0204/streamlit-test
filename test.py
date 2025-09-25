import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Hi Project 3")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    
    # Show dataframe
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Select column for histogram
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Choose a column to plot", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for plotting.")