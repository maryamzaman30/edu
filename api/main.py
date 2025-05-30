# File: api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

# --- Load Hybrid SVD Model ---
from src.recommender.hybrid import SVDHybridRecommender # Needed for unpickling

with open("models/svd_hybrid_model.pkl", "rb") as f:
    hybrid_package = pickle.load(f)

hybrid_model = hybrid_package['recommender']
user_map = hybrid_package['user_map']
item_map = hybrid_package['item_map']
reverse_user_map = hybrid_package['reverse_user_map']
reverse_item_map = hybrid_package['reverse_item_map']

# --- Load Content-Based Model ---
with open("models/content_based_model.pkl", "rb") as f:
    cb_model = pickle.load(f)

cb_vectorizer = cb_model['tfidf_vectorizer']
cb_matrix = cb_model['tfidf_matrix']
cb_cosine_sim = cb_model['cosine_sim']
bundle_features = cb_model['bundle_features']
bundle_indices = cb_model['bundle_indices']

# --- Load additional metadata ---
merged_df = pd.read_csv("data/cleaned/merged_cleaned_data.csv")
lectures_df = pd.read_csv("data/cleaned/cleaned_lectures.csv")

# Rebuild train_matrix for hybrid_model
user_item_counts = merged_df.groupby(["user_id", "bundle_id"]).size().reset_index(name="interaction_count")
unique_users = merged_df["user_id"].unique()
unique_bundles = merged_df["bundle_id"].unique()

uid_map = {uid: i for i, uid in enumerate(unique_users)}
bid_map = {bid: i for i, bid in enumerate(unique_bundles)}

train_matrix = np.zeros((len(uid_map), len(bid_map)))
for _, row in user_item_counts.iterrows():
    train_matrix[uid_map[row["user_id"]], bid_map[row["bundle_id"]]] = row["interaction_count"]

hybrid_model.train_matrix = train_matrix

# --- FastAPI Setup ---
app = FastAPI(title="Educational Recommender API")

class Recommendation(BaseModel):
    bundle_id: str
    score: float
    title: str
    subject: str
    part: str


def enrich_bundle(bundle_id):
    row = lectures_df[lectures_df['lecture_id'] == bundle_id]
    if row.empty:
        return ("N/A", "N/A", "N/A")
    r = row.iloc[0]
    return r.get("title", "N/A"), r.get("subject_category", "N/A"), r.get("part_name", "N/A")

@app.get("/")
def root():
    return {"message": "Use /recommendations/{user_id}, /recommendations/content/{user_id}, or /recommendations/collab/{user_id}"}

@app.get("/recommendations/{user_id}", response_model=list[Recommendation])
def recommend_hybrid(user_id: str, n: int = 10):
    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="User ID not found")

    user_idx = user_map[user_id]
    known_items = np.where(hybrid_model.train_matrix[user_idx] > 0)[0].tolist()
    recs = hybrid_model.get_recommendations(user_idx, n=n, exclude_seen=True, known_items=known_items)

    if not recs:
        raise HTTPException(status_code=204, detail="No recommendations available")

    return [
        {
            "bundle_id": reverse_item_map[item],
            "score": round(float(score), 4),
            "title": enrich_bundle(reverse_item_map[item])[0],
            "subject": enrich_bundle(reverse_item_map[item])[1],
            "part": enrich_bundle(reverse_item_map[item])[2]
        }
        for item, score in recs
    ]

@app.get("/recommendations/content/{user_id}", response_model=list[Recommendation])
def recommend_content(user_id: str, n: int = 10):
    user_data = merged_df[merged_df['user_id'] == user_id]
    if user_data.empty:
        raise HTTPException(status_code=404, detail="User ID not found or has no history")

    seen = set(user_data['bundle_id'].unique())
    scores = {}

    for bundle_id in seen:
        if bundle_id not in bundle_indices:
            continue
        idx = bundle_indices[bundle_id]
        sim_scores = cb_cosine_sim[idx]
        correctness = (user_data[user_data['bundle_id'] == bundle_id]['user_answer'] == user_data[user_data['bundle_id'] == bundle_id]['correct_answer']).mean()
        weight = correctness * (0.5 + len(user_data[user_data['bundle_id'] == bundle_id]) / 10)

        for i, sim in enumerate(sim_scores):
            candidate_id = bundle_features.iloc[i]['bundle_id']
            if candidate_id in seen:
                continue
            scores[candidate_id] = scores.get(candidate_id, 0) + sim * weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [
        {
            "bundle_id": bid,
            "score": round(score, 4),
            "title": enrich_bundle(bid)[0],
            "subject": enrich_bundle(bid)[1],
            "part": enrich_bundle(bid)[2]
        }
        for bid, score in ranked
    ]

@app.get("/recommendations/collab/{user_id}", response_model=list[Recommendation])
def recommend_collab(user_id: str, n: int = 10):
    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="User ID not found")

    user_idx = user_map[user_id]
    known_items = np.where(hybrid_model.train_matrix[user_idx] > 0)[0].tolist()

    # Simulate pure CF by setting combine_weight to 1.0
    collab_model = SVDHybridRecommender(n_factors=20, combine_weight=1.0)
    collab_model.user_factors = hybrid_model.user_factors
    collab_model.item_factors = hybrid_model.item_factors

    recs = collab_model.get_recommendations(user_idx, n=n, exclude_seen=True, known_items=known_items)

    if not recs:
        raise HTTPException(status_code=204, detail="No recommendations available")

    return [
        {
            "bundle_id": reverse_item_map[item],
            "score": round(float(score), 4),
            "title": enrich_bundle(reverse_item_map[item])[0],
            "subject": enrich_bundle(reverse_item_map[item])[1],
            "part": enrich_bundle(reverse_item_map[item])[2]
        }
        for item, score in recs
    ]
