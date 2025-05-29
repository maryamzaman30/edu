"""
Collaborative Filtering Module

This module implements collaborative filtering using SVD for the educational content recommendation system.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import logging
from typing import List, Dict, Tuple, Optional
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommendation system using SVD.
    
    This class implements a recommendation system that suggests content
    based on user-item interactions, using Singular Value Decomposition (SVD).
    """
    
    def __init__(self, user_item_matrix: pd.DataFrame, item_features: Optional[pd.DataFrame] = None):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            user_item_matrix: DataFrame with user-item interaction matrix
            item_features: Optional DataFrame with item features for hybrid factorization
        """
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        
        # Create mappings between user/item IDs and internal indices
        self.user_id_map = {user: i for i, user in enumerate(user_item_matrix.index)}
        self.item_id_map = {item: i for i, item in enumerate(user_item_matrix.columns)}
        self.reverse_user_map = {i: user for user, i in self.user_id_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_id_map.items()}
        
        # Convert user-item matrix to sparse format
        self.user_item_sparse = csr_matrix(user_item_matrix.values)
    
    def create_feature_matrices(self) -> Tuple[Optional[csr_matrix], Optional[csr_matrix]]:
        """
        Create feature matrices for LightFM model if item features are available.
        
        Returns:
            Tuple of (user_features, item_features) as sparse matrices, or (None, None)
        """
        if self.item_features is None:
            return None, None
            
        logger.info("Creating feature matrices for LightFM model")
        
        try:
            # Create one-hot encoded features for subjects and parts
            subjects = pd.get_dummies(self.item_features['subject_category'], prefix='subject')
            parts = pd.get_dummies(self.item_features['part_name'], prefix='part')
            
            # Combine features
            features_df = pd.concat([subjects, parts], axis=1)
            
            # Ensure features align with item indices
            item_features = np.zeros((len(self.item_id_map), features_df.shape[1]))
            
            for item_id, item_idx in self.item_id_map.items():
                # Get row index in item_features for this item_id
                item_rows = self.item_features[self.item_features['bundle_id'] == item_id]
                if len(item_rows) > 0:
                    item_idx_in_features = item_rows.index[0]
                    if item_idx_in_features in features_df.index:
                        item_features[item_idx, :] = features_df.loc[item_idx_in_features].values
            
            # Convert to sparse matrix
            item_features_sparse = csr_matrix(item_features)
            
            logger.info(f"Created item features matrix with shape {item_features_sparse.shape}")
            return None, item_features_sparse
            
        except Exception as e:
            logger.error(f"Error creating feature matrices: {str(e)}")
            return None, None
    
    def fit(self, num_components: int = 30, random_state: int = 42) -> None:
        """
        Fit the SVD model to the user-item matrix.
        
        Args:
            num_components: Number of latent factors
            random_state: Random seed for reproducibility
        """
        logger.info(f"Fitting SVD model with {num_components} components")
        
        # Initialize and fit SVD
        self.svd = TruncatedSVD(n_components=num_components, random_state=random_state)
        
        # Fit and transform the user-item matrix
        self.user_factors = self.svd.fit_transform(self.user_item_sparse)
        self.item_factors = self.svd.components_.T
        
        logger.info("SVD model fitting complete")
    
    def recommend_for_user(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """
        Recommend items for a user based on collaborative filtering.
        
        Args:
            user_id: The user ID to make recommendations for
            n: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended items with scores
        """
        try:
            # Get user index
            user_idx = self.user_id_map.get(user_id)
            if user_idx is None:
                logger.warning(f"User {user_id} not found in user mapping")
                return []
                
            # Get item scores using SVD factors
            user_vector = self.user_factors[user_idx]
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Create list of (item_idx, score) tuples
            item_scores = list(enumerate(scores))
            
            # Exclude seen items if requested
            if exclude_seen:
                try:
                    user_interactions = self.user_item_matrix.loc[user_id]
                    seen_item_indices = [self.item_id_map[item] for item in user_interactions.index[user_interactions > 0]]
                    item_scores = [(i, score) for i, score in item_scores if i not in seen_item_indices]
                except:
                    # If user not found in matrix, continue with all items
                    pass
            
            # Sort items by score and take top N
            item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)[:n]
            
            # Convert back to item IDs and format the recommendations
            recommendations = []
            for item_idx, score in item_scores:
                item_id = self.reverse_item_map.get(item_idx)
                if item_id is not None:
                    recommendations.append({
                        'bundle_id': item_id,
                        'score': float(score)
                    })
            
            return recommendations
                        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def recommend_similar_items(self, item_id: str, n: int = 10) -> List[Dict]:
        """
        Recommend items similar to a given item based on latent factors.
        
        Args:
            item_id: The item (bundle) ID to find similar items for
            n: Number of recommendations to return
            
        Returns:
            List of dictionaries with similar items and similarity scores
        """
        if self.svd is None:
            logger.error("Model not fitted. Call fit() before making recommendations.")
            return []
        
        # Check if item exists in the model
        if item_id not in self.item_id_map:
            logger.warning(f"Item {item_id} not found in the dataset")
            return []
            
        item_idx = self.item_id_map[item_id]
        
        try:
            # Get item embedding
            item_embedding = self.item_factors[item_idx]
            similarities = np.dot(self.item_factors, item_embedding)

            # Calculate similarity with all items
            similarities = np.dot(self.svd.item_embeddings, item_embedding)
            
            # Create a list of (item_idx, similarity) tuples
            item_similarities = list(enumerate(similarities))
            
            # Sort items by similarity and take top N+1 (including the item itself)
            item_similarities = sorted(item_similarities, key=lambda x: x[1], reverse=True)[:n+1]
            
            # Remove the item itself (should be the first one)
            item_similarities = [x for x in item_similarities if self.reverse_item_map[x[0]] != item_id][:n]
            
            # Convert back to item IDs and format the recommendations
            similar_items = [
                {
                    'bundle_id': self.reverse_item_map[item_idx],
                    'similarity_score': float(score)
                }
                for item_idx, score in item_similarities
            ]
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error calculating similar items: {str(e)}")
            return []
    
    def save_model(self, path: str = "models") -> str:
        """
        Save the fitted model to disk.
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to the saved model
        """
        if self.svd is None:
            logger.error("Model not fitted. Call fit() before saving.")
            return ""
            
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "svd_hybrid_model.pkl")
        
        model_data = {
            'model': self.svd,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a fitted model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(model_path)
            
            self.svd = model_data['model']
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.reverse_user_map = model_data['reverse_user_map']
            self.reverse_item_map = model_data['reverse_item_map']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
