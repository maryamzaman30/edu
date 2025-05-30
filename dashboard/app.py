# File: app.py

import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
st.set_page_config(page_title="TOEIC Recommendation Dashboard", layout="wide")
st.title("📚 Personalized TOEIC Recommendations Dashboard")

# --- Load user IDs ---
@st.cache_data
def get_user_ids():
    df = pd.read_csv("data/cleaned/merged_cleaned_data.csv")
    return sorted(df["user_id"].unique().tolist())

# --- Load tuning results ---
@st.cache_data
def load_tuning_results():
    tfidf = pd.read_csv("data/cleaned/tfidf_tuning_results.csv")
    hybrid = pd.read_csv("data/cleaned/hybrid_tuning_results.csv")
    return tfidf, hybrid

# --- Sidebar User Selector ---
user_ids = get_user_ids()
user_id = st.sidebar.selectbox("Select User ID", user_ids)
top_k = st.sidebar.slider("Number of recommendations (Top-K)", 5, 20, 10)

# --- Fetch Recommendations ---
def fetch_recommendations(user_id, type="hybrid", k=10):
    endpoint = f"http://127.0.0.1:8000/recommendations/{'' if type=='hybrid' else 'content/'}{user_id}?n={k}"
    try:
        res = requests.get(endpoint)
        if res.status_code == 200:
            return res.json()
        return []
    except:
        return []

col1, col2 = st.columns(2)

# --- Hybrid Recs ---
hybrid_recs = fetch_recommendations(user_id, "hybrid", top_k)
with col1:
    st.subheader("Hybrid Recommendations")
    if hybrid_recs:
        st.dataframe(pd.DataFrame(hybrid_recs))
    else:
        st.warning("No hybrid recommendations available.")

# --- Content Recs ---
content_recs = fetch_recommendations(user_id, "content", top_k)
with col2:
    st.subheader("Content-Based Recommendations")
    if content_recs:
        st.dataframe(pd.DataFrame(content_recs))
    else:
        st.warning("No content-based recommendations available.")

# --- Charts Section ---
st.markdown("---")
st.header("📊 Evaluation and Tuning Results")
tfidf_df, hybrid_df = load_tuning_results()

# TF-IDF plot
st.subheader("TF-IDF Precision Comparison")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=tfidf_df.sort_values("precision", ascending=False), x="params", y="precision", ax=ax1)
ax1.set_title("TF-IDF Parameter Tuning")
ax1.set_xlabel("Parameters")
ax1.set_ylabel("Precision@10")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

# Hybrid plot
st.subheader("Hybrid Weight vs Metrics")
fig2, ax2 = plt.subplots(figsize=(10, 4))
hybrid_df = hybrid_df.sort_values("weight")
ax2.plot(hybrid_df["weight"], hybrid_df["precision@10"], label="Precision@10", marker="o")
ax2.plot(hybrid_df["weight"], hybrid_df["recall@10"], label="Recall@10", marker="o")
ax2.plot(hybrid_df["weight"], hybrid_df["rmse"], label="RMSE", marker="o")
ax2.set_title("Hybrid Model Tuning")
ax2.set_xlabel("Blend Weight α")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.markdown("---")
st.caption("Streamlit UI")