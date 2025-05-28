"""
Hybrid Recommendation Module

This module implements a hybrid recommendation system that combines
content-based filtering and collaborative filtering for educational content.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeFilteringRecommender
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hybrid recommendation system combining content-based and collaborative filtering.
    
    This class implements a recommendation system that combines recommendations
    from content-based filtering and collaborative filtering models.
    """
    
    def __init__(self, content_recommender: ContentBasedRecommender, 
                collab_recommender: CollaborativeFilteringRecommender,
                item_features: pd.DataFrame):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_recommender: Fitted content-based recommender
            collab_recommender: Fitted collaborative filtering recommender
            item_features: DataFrame with item features for enriching recommendations
        """
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.item_features = item_features
        self.alpha = 0.5  # Weight for content-based recommendations
        
    def set_weights(self, alpha: float) -> None:
        """
        Set weights for combining recommendations.
        
        Args:
            alpha: Weight for content-based recommendations (0-1)
                  Weight for collaborative recommendations will be (1-alpha)
        """
        self.alpha = max(0, min(1, alpha))  # Ensure alpha is between 0 and 1
        logger.info(f"Set hybrid weights: content_weight={self.alpha}, collab_weight={1-self.alpha}")
    
    def recommend_for_user(self, user_id: str, user_history: pd.DataFrame = None,
                          n: int = 10, exclude_seen: bool = True,
                          diversity_factor: float = 0.2, cold_start: bool = False) -> List[Dict]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: The user ID to make recommendations for
            user_history: Optional DataFrame with user's interaction history
            n: Number of recommendations to return
            exclude_seen: Whether to exclude items the user has already interacted with
            diversity_factor: Factor to control diversity in content-based recommendations
            cold_start: If True, use only content-based recommendations (for new users)
            
        Returns:
            List of dictionaries with recommended items and scores
        """
        try:
            # Get content-based recommendations
            content_recs = []
            if user_history is not None and len(user_history) > 0:
                content_recs = self.content_recommender.recommend_for_user(
                    user_history, 
                    n=n*2,  # Get more than needed to allow for merging
                    exclude_seen=exclude_seen,
                    diversity_factor=diversity_factor
                )
            
            # Get collaborative filtering recommendations
            collab_recs = []
            if not cold_start:
                collab_recs = self.collab_recommender.recommend_for_user(
                    user_id,
                    n=n*2,  # Get more than needed to allow for merging
                    exclude_seen=exclude_seen
                )
            
            # If either model failed to provide recommendations, use the other model's recommendations
            if len(content_recs) == 0 and len(collab_recs) == 0:
                logger.warning(f"No recommendations found for user {user_id}")
                return []
            elif len(content_recs) == 0:
                logger.info(f"Using only collaborative recommendations for user {user_id}")
                return self._enrich_recommendations(collab_recs[:n])
            elif len(collab_recs) == 0 or cold_start:
                logger.info(f"Using only content-based recommendations for user {user_id}")
                return self._enrich_recommendations(content_recs[:n])
            
            # Normalize scores within each recommendation set
            # content_recs = self._normalize_scores(content_recs, 'content_score')
            # collab_recs = self._normalize_scores(collab_recs, 'collab_score')
            for rec in content_recs:
                rec['content_score'] = rec.get('score', 0)

            for rec in collab_recs:
                rec['collab_score'] = rec.get('score', 0)

            
            # Combine recommendations
            hybrid_recs = self._combine_recommendations(content_recs, collab_recs, n)
            
            # Enrich with item metadata
            return self._enrich_recommendations(hybrid_recs)
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {str(e)}")
            return []
    
    def _normalize_scores(self, recommendations: List[Dict], score_key: str) -> List[Dict]:
        """
        Normalize scores in a list of recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            score_key: Key for the score to normalize
            
        Returns:
            List of recommendations with normalized scores
        """
        if not recommendations:
            return []
            
        # Extract scores
        # scores = [rec[score_key] for rec in recommendations]
        scores = [rec.get(score_key, 0) for rec in recommendations]

        
        # Find min and max scores
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            normalized_recs = [{**rec, f'{score_key}_normalized': 1.0} for rec in recommendations]
        else:
            # Normalize scores to [0, 1]
            normalized_recs = [
                {**rec, f'{score_key}_normalized': (rec[score_key] - min_score) / (max_score - min_score)}
                for rec in recommendations
            ]
            
        return normalized_recs
    
    def _combine_recommendations(self, content_recs: List[Dict], collab_recs: List[Dict], n: int) -> List[Dict]:
        """
        Combine content-based and collaborative recommendations.
        
        Args:
            content_recs: List of content-based recommendations with normalized scores
            collab_recs: List of collaborative recommendations with normalized scores
            n: Number of recommendations to return
            
        Returns:
            List of combined recommendations
        """
        # Create dictionaries for quick lookup
        content_dict = {rec['bundle_id']: rec for rec in content_recs}
        collab_dict = {rec['bundle_id']: rec for rec in collab_recs}
        
        # All unique bundle IDs
        all_bundles = set(content_dict.keys()) | set(collab_dict.keys())
        
        # Combine scores for each bundle
        hybrid_scores = {}
        for bundle_id in all_bundles:
            content_score = content_dict.get(bundle_id, {}).get('content_score_normalized', 0)
            collab_score = collab_dict.get(bundle_id, {}).get('collab_score_normalized', 0)
            
            # Weighted combination of scores
            hybrid_score = self.alpha * content_score + (1 - self.alpha) * collab_score
            
            hybrid_scores[bundle_id] = {
                'bundle_id': bundle_id,
                'hybrid_score': hybrid_score,
               # 'content_score': content_dict.get(bundle_id, {}).get('content_score', 0),
               # 'collab_score': collab_dict.get(bundle_id, {}).get('collab_score', 0)
            }
        
        # Sort by hybrid score and take top N
        sorted_recs = sorted(hybrid_scores.values(), key=lambda x: x['hybrid_score'], reverse=True)[:n]
        
        return sorted_recs
    
    def _enrich_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Enrich recommendations with item metadata.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            List of recommendations with added metadata
        """
        enriched_recs = []
        
        for rec in recommendations:
            bundle_id = rec['bundle_id']
            
            # Get item metadata
            bundle_rows = self.item_features[self.item_features['bundle_id'] == bundle_id]
            
            if len(bundle_rows) > 0:
                bundle_row = bundle_rows.iloc[0]
                
                # Create enriched recommendation
                enriched_rec = {
                    'bundle_id': bundle_id,
                    'title': f"Bundle {bundle_id}: {bundle_row.get('part_name', 'Unknown')}",
                    'part': bundle_row.get('part_name', 'Unknown'),
                    'subject_category': bundle_row.get('subject_category', 'Unknown'),
                    'difficulty': bundle_row.get('difficulty', 'Unknown'),
                    'question_count': int(bundle_row.get('question_count', 0)),
                    'popularity': int(bundle_row.get('interaction_count', 0)),
                    'success_rate': float(bundle_row.get('success_rate', 0)),
                    'hybrid_score': rec.get('hybrid_score', 0),
                    'content_score': rec.get('content_score', 0),
                    'collab_score': rec.get('collab_score', 0)
                }
                
                enriched_recs.append(enriched_rec)
            else:
                # If metadata not found, just add the bundle ID
                enriched_recs.append({
                    'bundle_id': bundle_id,
                    'title': f"Bundle {bundle_id}",
                    'part': "Unknown",
                    'subject_category': "Unknown",
                    'difficulty': "Unknown",
                    'popularity': 0,
                    'hybrid_score': rec.get('hybrid_score', 0),
                    'content_score': rec.get('content_score', 0),
                    'collab_score': rec.get('collab_score', 0)
                })
        
        return enriched_recs
    
    def tune_weights(self, validation_data: pd.DataFrame, 
                     weight_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> float:
        """
        Tune the weights for combining recommendations using validation data.
        
        Args:
            validation_data: DataFrame with validation interactions
            weight_range: List of alpha values to try
            
        Returns:
            Best alpha value
        """
        best_score = 0
        best_alpha = 0.5
        
        logger.info(f"Tuning hybrid weights with {len(weight_range)} values")
        
        for alpha in weight_range:
            self.set_weights(alpha)
            
            # Evaluate with this alpha
            score = self._evaluate_on_validation(validation_data)
            
            logger.info(f"Alpha={alpha}, validation score={score}")
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
        
        logger.info(f"Best alpha value: {best_alpha} with score {best_score}")
        self.set_weights(best_alpha)
        
        return best_alpha
    
    def _evaluate_on_validation(self, validation_data: pd.DataFrame) -> float:
        """
        Evaluate the hybrid recommender on validation data.
        
        Args:
            validation_data: DataFrame with validation interactions
            
        Returns:
            Evaluation score (higher is better)
        """
        # Group validation data by user
        user_groups = validation_data.groupby('user_id')
        
        total_score = 0
        count = 0
        
        for user_id, group in user_groups:
            try:
                # Split into history and future interactions
                history = group.iloc[:len(group)//2]
                future = group.iloc[len(group)//2:]
                
                # Get recommendations based on history
                recs = self.recommend_for_user(user_id, history, n=10)
                
                # See if recommendations match future interactions
                rec_ids = [rec['bundle_id'] for rec in recs]
                future_ids = future['bundle_id'].unique()
                
                # Calculate precision@K
                matches = len(set(rec_ids) & set(future_ids))
                precision = matches / len(rec_ids) if rec_ids else 0
                
                total_score += precision
                count += 1
                
            except Exception as e:
                logger.error(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Average precision@K
        return total_score / count if count > 0 else 0
    
    def save_model(self, path: str = "models") -> str:
        """
        Save the hybrid model configuration to disk.
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path to the saved model
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "hybrid_model.joblib")
        
        model_data = {
            'alpha': self.alpha
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Hybrid model configuration saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> bool:
        """
        Load hybrid model configuration from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(model_path)
            
            self.alpha = model_data['alpha']
            
            logger.info(f"Hybrid model configuration loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hybrid model: {str(e)}")
            return False
