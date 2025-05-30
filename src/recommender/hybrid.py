# File: src/recommender/hybrid.py

import numpy as np
import time
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class SVDHybridRecommender:
    """SVD-based hybrid recommender system"""

    def __init__(self, n_factors=20, combine_weight=0.7):
        self.n_factors = n_factors
        self.combine_weight = combine_weight
        self.user_factors = None
        self.item_factors = None
        self.content_similarity = None
        self.mean_rating = None

    def fit(self, ratings_matrix, content_similarity=None):
        self.content_similarity = content_similarity
        self.mean_rating = np.mean(ratings_matrix[ratings_matrix > 0])
        ratings_filled = ratings_matrix.copy()
        ratings_filled[ratings_filled == 0] = self.mean_rating
        start_time = time.time()
        u, sigma, vt = svds(ratings_filled, k=min(self.n_factors, min(ratings_filled.shape) - 1))
        end_time = time.time()
        self.user_factors = u
        self.item_factors = vt.T
        print(f"SVD model trained in {end_time - start_time:.2f} seconds")
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")

    def predict_rating(self, user_idx, item_idx):
        if self.user_factors is None or self.item_factors is None:
            return self.mean_rating
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

    def _get_content_score(self, known_items, item_idx):
        if self.content_similarity is None or len(known_items) == 0:
            return 0.0
        return np.mean([self.content_similarity[item_idx, idx] for idx in known_items])

    def get_recommendations(self, user_idx, n=10, exclude_seen=True, known_items=None):
        if self.user_factors is None or self.item_factors is None:
            return []

        cf_scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        final_scores = cf_scores.copy()

        if self.content_similarity is not None and known_items is not None and len(known_items) > 0:
            for i in range(len(final_scores)):
                cb_score = self._get_content_score(known_items, i)
                final_scores[i] = (self.combine_weight * cf_scores[i]) + ((1 - self.combine_weight) * cb_score)

        all_items = np.arange(len(final_scores))
        if exclude_seen and known_items is not None and len(known_items) > 0:
            mask = np.ones_like(final_scores, dtype=bool)
            mask[known_items] = False
            all_items = all_items[mask]
            final_scores = final_scores[mask]

        top_indices = np.argsort(-final_scores)[:n]
        top_items = all_items[top_indices]
        top_scores = final_scores[top_indices]

        return list(zip(top_items, top_scores))
