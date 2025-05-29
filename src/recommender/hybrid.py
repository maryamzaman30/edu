
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from typing import List, Tuple, Dict, Optional
import time

class HybridRecommender:
    """
    Hybrid recommendation system combining content-based (TF-IDF) and 
    collaborative filtering (SVD) approaches
    """
    
    def __init__(self, n_factors: int = 20, content_weight: float = 0.3):
        """
        Initialize the hybrid recommender
        
        Args:
            n_factors: Number of latent factors for SVD
            content_weight: Weight for content-based component (0-1)
        """
        self.n_factors = n_factors
        self.content_weight = content_weight
        self.collab_weight = 1 - content_weight
        
        # Model components
        self.svd_model = None
        self.tfidf_vectorizer = None
        self.content_similarity = None
        
        # Data mappings
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        
        # Cached data
        self.user_factors = None
        self.item_factors = None
        self.mean_rating = None
        self.bundle_features = None
        
    def fit(self, interaction_data: pd.DataFrame, content_features: pd.DataFrame):
        """
        Train the hybrid model
        
        Args:
            interaction_data: User-item interaction data
            content_features: Content metadata for TF-IDF
        """
        print("Training hybrid recommender...")
        start_time = time.time()
        
        # Create mappings
        self._create_mappings(interaction_data)
        
        # Prepare interaction matrix
        ratings_matrix = self._create_ratings_matrix(interaction_data)
        
        # Train SVD component
        self._train_svd(ratings_matrix)
        
        # Train content-based component
        self._train_content_based(content_features)
        
        end_time = time.time()
        print(f"Hybrid model trained in {end_time - start_time:.2f} seconds")
        
    def _create_mappings(self, interaction_data: pd.DataFrame):
        """Create user and item ID mappings"""
        unique_users = interaction_data['user_id'].unique()
        unique_items = interaction_data['bundle_id'].unique()
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
        
    def _create_ratings_matrix(self, interaction_data: pd.DataFrame) -> np.ndarray:
        """Create user-item ratings matrix"""
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        
        # Aggregate interactions per user-item pair
        user_item_data = interaction_data.groupby(['user_id', 'bundle_id']).agg({
            'is_correct': 'mean',
            'solving_id': 'count',  # Number of interactions
            'elapsed_time': 'mean'
        }).reset_index()
        
        user_item_data.columns = ['user_id', 'bundle_id', 'correctness_rate', 
                                 'interaction_count', 'avg_time']
        
        # Create ratings matrix
        ratings_matrix = np.zeros((n_users, n_items))
        
        for _, row in user_item_data.iterrows():
            user_id = row['user_id']
            item_id = row['bundle_id']
            
            if user_id in self.user_map and item_id in self.item_map:
                user_idx = self.user_map[user_id]
                item_idx = self.item_map[item_id]
                
                # Create rating based on interaction count and correctness
                base_rating = row['interaction_count']
                correctness_boost = 1 + (row['correctness_rate'] * 0.5)
                
                ratings_matrix[user_idx, item_idx] = base_rating * correctness_boost
        
        return ratings_matrix
        
    def _train_svd(self, ratings_matrix: np.ndarray):
        """Train SVD component"""
        print("Training SVD component...")
        
        # Calculate mean rating
        self.mean_rating = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Fill missing values with mean for SVD
        ratings_filled = ratings_matrix.copy()
        ratings_filled[ratings_filled == 0] = self.mean_rating
        
        # Apply SVD
        k = min(self.n_factors, min(ratings_filled.shape) - 1)
        u, sigma, vt = svds(ratings_filled, k=k)
        
        # Store factors
        self.user_factors = u
        self.item_factors = vt.T
        
    def _train_content_based(self, content_features: pd.DataFrame):
        """Train content-based component using TF-IDF"""
        print("Training content-based component...")
        
        # Store bundle features
        self.bundle_features = content_features.copy()
        
        # Create content text for TF-IDF
        content_text = (
            content_features['part_name'].fillna('') + ' ' +
            content_features['subject_category'].fillna('') + ' ' +
            content_features['tags'].fillna('')
        )
        
        # Apply TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_text)
        
        # Calculate content similarity
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair using SVD"""
        if self.user_factors is None or self.item_factors is None:
            return self.mean_rating
        
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def get_content_score(self, known_items: List[int], item_idx: int) -> float:
        """Get content-based similarity score"""
        if self.content_similarity is None or len(known_items) == 0:
            return 0.0
        
        # Calculate average similarity to known items
        similarities = [self.content_similarity[item_idx, idx] for idx in known_items]
        return np.mean(similarities)
    
    def get_recommendations(self, user_id: str, n: int = 10, 
                          exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Get hybrid recommendations for a user
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            exclude_seen: Whether to exclude items user has seen
            
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_map:
            return self._get_popular_items(n)
        
        user_idx = self.user_map[user_id]
        
        # Get user's known items for content filtering
        known_items = self._get_user_known_items(user_id)
        known_item_indices = [self.item_map[item] for item in known_items 
                             if item in self.item_map]
        
        # Calculate scores for all items
        all_scores = {}
        
        for item_idx in range(len(self.item_map)):
            item_id = self.reverse_item_map[item_idx]
            
            # Skip seen items if requested
            if exclude_seen and item_id in known_items:
                continue
            
            # Get collaborative filtering score
            collab_score = self.predict_rating(user_idx, item_idx)
            
            # Get content-based score
            content_score = self.get_content_score(known_item_indices, item_idx)
            
            # Combine scores
            hybrid_score = (self.collab_weight * collab_score + 
                           self.content_weight * content_score)
            
            all_scores[item_id] = hybrid_score
        
        # Sort and return top N
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    def _get_user_known_items(self, user_id: str) -> List[str]:
        """Get items that user has interacted with"""
        # This would typically query the original interaction data
        # For now, return empty list (implement based on your data structure)
        return []
    
    def _get_popular_items(self, n: int) -> List[Tuple[str, float]]:
        """Get popular items for cold start users"""
        # Return items with highest average ratings
        if self.bundle_features is not None:
            popular_items = self.bundle_features.nlargest(n, 'interaction_count')
            return [(item_id, 1.0 - i*0.1) for i, item_id in 
                   enumerate(popular_items['bundle_id'])]
        return []
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test interaction data
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating hybrid model...")
        
        # Implement evaluation metrics
        # This is a placeholder - implement based on your evaluation framework
        
        return {
            'rmse': 0.0,
            'precision@10': 0.0,
            'recall@10': 0.0,
            'coverage': 0.0
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'content_similarity': self.content_similarity,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'mean_rating': self.mean_rating,
            'bundle_features': self.bundle_features,
            'n_factors': self.n_factors,
            'content_weight': self.content_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            n_factors=model_data['n_factors'],
            content_weight=model_data['content_weight']
        )
        
        # Load components
        instance.user_factors = model_data['user_factors']
        instance.item_factors = model_data['item_factors']
        instance.content_similarity = model_data['content_similarity']
        instance.tfidf_vectorizer = model_data['tfidf_vectorizer']
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        instance.mean_rating = model_data['mean_rating']
        instance.bundle_features = model_data['bundle_features']
        
        print(f"Model loaded from {filepath}")
        return instance
