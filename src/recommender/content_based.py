"""
Content-Based Recommendation Module

This module implements content-based filtering using TF-IDF
for the educational content recommendation system.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF vectorization.
    
    This class implements a recommendation system that suggests content
    based on similarity between item features, using TF-IDF vectorization
    for text features and cosine similarity for calculating similarity scores.
    """
    
    def __init__(self, item_features: pd.DataFrame):
        """
        Initialize the content-based recommender.
        
        Args:
            item_features: DataFrame with item features
        """
        self.item_features = item_features
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.index_to_bundle = {}
        self.bundle_to_index = {}
        
    def preprocess_content(self) -> pd.DataFrame:
        """
        Preprocess item content for TF-IDF vectorization.
        
        Returns:
            DataFrame with processed content
        """
        logger.info("Preprocessing content for TF-IDF vectorization")
        
        # Create a text representation for each bundle
        self.item_features['content_text'] = (
            self.item_features['part_name'].fillna('') + ' ' +
            self.item_features['subject_category'].fillna('') + ' ' +
            self.item_features['tags'].fillna('')
        )
        
        # Create mappings between indices and bundle IDs
        self.index_to_bundle = {i: bundle_id for i, bundle_id in enumerate(self.item_features['bundle_id'])}
        self.bundle_to_index = {bundle_id: i for i, bundle_id in enumerate(self.item_features['bundle_id'])}
        
        return self.item_features
    
    def fit(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        """
        Fit the TF-IDF vectorizer to the item features.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
        """
        if 'content_text' not in self.item_features.columns:
            self.preprocess_content()
            
        logger.info(f"Fitting TF-IDF vectorizer with max_features={max_features}, ngram_range={ngram_range}")
        
        # Create and fit the TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range
        )
        
        # Transform content text into TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.item_features['content_text'])
        
        # Calculate cosine similarity between all items
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
    
    def recommend_similar_items(self, item_id: str, n: int = 10) -> List[Dict]:
        """
        Recommend items similar to a given item.
        
        Args:
            item_id: The item (bundle) ID to find similar items for
            n: Number of recommendations to return
            
        Returns:
            List of dictionaries with similar items and similarity scores
        """
        if self.similarity_matrix is None:
            logger.error("Model not fitted. Call fit() before making recommendations.")
            return []
        
        # Get the index of the item
        if item_id not in self.bundle_to_index:
            logger.warning(f"Item {item_id} not found in the dataset")
            return []
            
        item_idx = self.bundle_to_index[item_id]
        
        # Get similarity scores and indices of similar items
        similarity_scores = list(enumerate(self.similarity_matrix[item_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar items (excluding the item itself)
        similar_items = [
            {
                'bundle_id': self.index_to_bundle[idx],
                'similarity_score': score
            }
            for idx, score in similarity_scores[1:n+1]
        ]
        
        return similar_items
    
    def recommend_for_user(self, user_history: pd.DataFrame, n: int = 10, 
                            exclude_seen: bool = True, diversity_factor: float = 0.2) -> List[Dict]:
        """
        Recommend items for a user based on their interaction history.
        
        Args:
            user_history: DataFrame with user's interaction history
            n: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already interacted with
            diversity_factor: Factor to control diversity (0-1), higher means more diverse
            
        Returns:
            List of dictionaries with recommended items and scores
        """
        if self.similarity_matrix is None:
            logger.error("Model not fitted. Call fit() before making recommendations.")
            return []
        
        # Get unique bundles from user history
        user_bundles = user_history['bundle_id'].unique()
        
        # Items the user has already seen (to exclude)
        seen_items = set(user_bundles) if exclude_seen else set()
        
        # Calculate recommendation scores for each item based on user history
        recommendation_scores = {}
        
        # For each item in user history, get similar items and add their scores
        for bundle_id in user_bundles:
            if bundle_id not in self.bundle_to_index:
                continue
                
            similar_items = self.recommend_similar_items(bundle_id, n=50)
            
            # Calculate correctness for this bundle (as a weight)
            bundle_data = user_history[user_history['bundle_id'] == bundle_id]
            correctness = (bundle_data['user_answer'] == bundle_data['correct_answer']).mean()
            interaction_count = len(bundle_data)
            
            # Weight based on correctness and recency
            # Higher weight for more recent and more correct interactions
            weight = correctness * (0.5 + interaction_count / 10)  # Simple weighting scheme
            
            # Add scores for similar items
            for item in similar_items:
                item_id = item['bundle_id']
                
                # Skip if item already seen
                if item_id in seen_items:
                    continue
                    
                # Update recommendation score for this item
                if item_id not in recommendation_scores:
                    # Get item metadata
                    item_idx = self.bundle_to_index[item_id]
                    bundle_row = self.item_features.iloc[item_idx]
                    
                    recommendation_scores[item_id] = {
                        'bundle_id': item_id,
                        'score': 0,
                        'subject': bundle_row.get('subject_category', 'Unknown'),
                        'part': bundle_row.get('part_name', 'Unknown'),
                        'difficulty': bundle_row.get('difficulty', 'Unknown'),
                        'question_count': int(bundle_row.get('question_count', 0)),
                        'popularity': int(bundle_row.get('interaction_count', 0)),
                        'success_rate': float(bundle_row.get('success_rate', 0))
                    }
                
                # Add weighted similarity score
                recommendation_scores[item_id]['score'] += item['similarity_score'] * weight
        
        # Add diversity by considering popularity and subjects
        # Add diversity by considering popularity and subjects
        if diversity_factor > 0:
            user_subjects = set()
            for bundle_id in user_bundles:
                if bundle_id in self.bundle_to_index:
                    bundle_idx = self.bundle_to_index[bundle_id]
                    bundle_row = self.item_features.iloc[bundle_idx]
                    subject = bundle_row.get('subject_category', None)
                    if subject:
                        user_subjects.add(subject)

            for item_id, rec in list(recommendation_scores.items()):
                bundle_idx = self.bundle_to_index.get(item_id)
                if bundle_idx is not None:
                    bundle_row = self.item_features.iloc[bundle_idx]
                    subject = bundle_row.get('subject_category', None)
                    if subject and subject not in user_subjects:
                        recommendation_scores[item_id]['score'] = rec['score'] * (1 + diversity_factor)

        
        # Sort recommendations by score and take top N
        recommendations = sorted(
            [(item_id, rec['score']) for item_id, rec in recommendation_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        # Convert to list of dictionaries with all metadata
        return [{
            'bundle_id': item_id,
            'score': score,
            'subject': recommendation_scores[item_id]['subject'],
            'part': recommendation_scores[item_id]['part'],
            'difficulty': recommendation_scores[item_id]['difficulty'],
            'question_count': recommendation_scores[item_id]['question_count'],
            'popularity': recommendation_scores[item_id]['popularity'],
            'success_rate': recommendation_scores[item_id]['success_rate']
        } for item_id, score in recommendations]
    
    def save_model(self, path: str = "models") -> str:
        """
        Save the fitted model to disk.
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to the saved model
        """
        if self.tfidf_vectorizer is None or self.similarity_matrix is None:
            logger.error("Model not fitted. Call fit() before saving.")
            return ""
            
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "content_based_model.joblib")
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'similarity_matrix': self.similarity_matrix,
            'index_to_bundle': self.index_to_bundle,
            'bundle_to_index': self.bundle_to_index
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
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.index_to_bundle = model_data['index_to_bundle']
            self.bundle_to_index = model_data['bundle_to_index']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
