"""
Evaluation Metrics Module

This module provides functions to evaluate the performance of recommendation models
using standard metrics like Precision@K, Recall@K, RMSE, and AUC.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """
    Evaluator for recommendation system models.
    
    This class provides functions to evaluate the performance of recommendation models
    using standard metrics and produce visualizations of the results.
    """
    
    def __init__(self, test_data: pd.DataFrame):
        """
        Initialize the evaluator with test data.
        
        Args:
            test_data: DataFrame with test interactions
        """
        self.test_data = test_data
        self.results = {}
        
    def precision_at_k(self, recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
        """
        Calculate Precision@K for a set of recommendations.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Precision@K score
        """
        # Take only the top K recommendations
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
            
        # Calculate precision
        if not recommended_items:
            return 0.0
            
        matches = len(set(recommended_items) & relevant_items)
        return matches / len(recommended_items)
    
    def recall_at_k(self, recommended_items: List[str], relevant_items: Set[str], k: int = 10) -> float:
        """
        Calculate Recall@K for a set of recommendations.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Recall@K score
        """
        # Take only the top K recommendations
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
            
        # Calculate recall
        if not relevant_items:
            return 0.0
            
        matches = len(set(recommended_items) & relevant_items)
        return matches / len(relevant_items)
    
    def ndcg_at_k(self, recommended_items: List[str], relevant_items: Dict[str, float], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Dictionary mapping item IDs to relevance scores
            k: Number of recommendations to consider
            
        Returns:
            NDCG@K score
        """
        # Take only the top K recommendations
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
            
        # If no recommendations or no relevant items, return 0
        if not recommended_items or not relevant_items:
            return 0.0
            
        # Calculate DCG
        dcg = 0
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                # Add relevance score with position discount
                dcg += relevant_items[item] / np.log2(i + 2)  # +2 because i is 0-indexed
                
        # Calculate ideal DCG (IDCG)
        ideal_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)
        idcg = 0
        for i, (item, score) in enumerate(ideal_items[:k]):
            idcg += score / np.log2(i + 2)
            
        # Calculate NDCG
        if idcg == 0:
            return 0.0
            
        return float(dcg) / float(idcg)

    
    def mean_average_precision(self, recommendations_by_user: Dict[str, List[str]], 
                              relevant_items_by_user: Dict[str, Set[str]], k: int = 10) -> float:
        """
        Calculate Mean Average Precision (MAP) across all users.
        
        Args:
            recommendations_by_user: Dictionary mapping user IDs to lists of recommended item IDs
            relevant_items_by_user: Dictionary mapping user IDs to sets of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            MAP score
        """
        # Calculate AP for each user
        average_precisions = []
        
        for user_id, recommended_items in recommendations_by_user.items():
            if user_id not in relevant_items_by_user:
                continue
                
            relevant_items = relevant_items_by_user[user_id]
            
            # Take only the top K recommendations
            if len(recommended_items) > k:
                recommended_items = recommended_items[:k]
                
            # Calculate precision at each position where a relevant item is found
            precisions = []
            hits = 0
            
            for i, item in enumerate(recommended_items):
                if item in relevant_items:
                    hits += 1
                    precisions.append(hits / (i + 1))
                    
            # Calculate AP
            if hits > 0:
                ap = sum(precisions) / len(relevant_items)
                average_precisions.append(ap)
                
        # Calculate MAP
        if not average_precisions:
            return 0.0
            
        return sum(average_precisions) / len(average_precisions)
    
    def catalog_coverage(self, all_recommendations: List[List[str]], catalog_size: int) -> float:
        """
        Calculate catalog coverage of the recommendations.
        
        Args:
            all_recommendations: List of lists containing recommended item IDs for all users
            catalog_size: Total number of items in the catalog
            
        Returns:
            Catalog coverage as a percentage
        """
        # Flatten all recommendations and count unique items
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
            
        # Calculate coverage
        return len(unique_items) / catalog_size * 100
    
    def diversity(self, recommendations: List[str], item_features: pd.DataFrame) -> float:
        """
        Calculate diversity of a set of recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            item_features: DataFrame with item features
            
        Returns:
            Diversity score (higher means more diverse)
        """
        if not recommendations or len(recommendations) < 2:
            return 0.0
            
        # Get features for recommended items
        rec_features = item_features[item_features['bundle_id'].isin(recommendations)]
        
        if len(rec_features) == 0:
            return 0.0
            
        # Count unique values in categorical features
        unique_parts = rec_features['part_name'].nunique() if 'part_name' in rec_features.columns else 0
        unique_subjects = rec_features['subject_category'].nunique() if 'subject_category' in rec_features.columns else 0
        
        # Calculate diversity as average of normalized unique counts
        max_parts = item_features['part_name'].nunique() if 'part_name' in item_features.columns else 1
        max_subjects = item_features['subject_category'].nunique() if 'subject_category' in item_features.columns else 1
        
        part_diversity = unique_parts / max_parts if max_parts > 0 else 0
        subject_diversity = unique_subjects / max_subjects if max_subjects > 0 else 0
        
        return (part_diversity + subject_diversity) / 2
    
    def rmse(self, predictions: List[float], actual: List[float]) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        
        Args:
            predictions: List of predicted values
            actual: List of actual values
            
        Returns:
            RMSE score
        """
        if len(predictions) != len(actual):
            logger.error(f"Length mismatch: predictions ({len(predictions)}), actual ({len(actual)})")
            return float('inf')
            
        return np.sqrt(mean_squared_error(actual, predictions))
    
    def auc_score(self, predictions: List[float], actual: List[int]) -> float:
        """
        Calculate Area Under the ROC Curve (AUC).
        
        Args:
            predictions: List of predicted probabilities
            actual: List of actual binary values (0 or 1)
            
        Returns:
            AUC score
        """
        if len(predictions) != len(actual):
            logger.error(f"Length mismatch: predictions ({len(predictions)}), actual ({len(actual)})")
            return 0.5  # Random classifier baseline
            
        try:
            return roc_auc_score(actual, predictions)
        except Exception as e:
            logger.error(f"Error calculating AUC: {str(e)}")
            return 0.5
    
    def evaluate_model(self, model_name: str, recommender_func, k_values: List[int] = [5, 10, 20],
                       item_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate a recommendation model using various metrics.
        
        Args:
            model_name: Name of the model being evaluated
            recommender_func: Function that takes a user_id and returns recommendations
            k_values: List of K values for Precision@K and Recall@K
            item_features: Optional DataFrame with item features for diversity calculation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Group test data by user
        user_groups = self.test_data.groupby('user_id')
        
        # Metrics to track
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        diversity_scores = []
        
        all_recommendations = []
        
        # Evaluate for each user
        for user_id, group in user_groups:
            try:
                # Split into history and future interactions
                history = group.iloc[:len(group)//2]
                future = group.iloc[len(group)//2:]
                
                # Skip users with no future interactions
                if len(future) == 0:
                    continue
                    
                # Get recommendations
                recommendations = recommender_func(user_id, history)
                if not recommendations:
                    continue
                    
                # Extract recommended item IDs
                rec_ids = [rec['bundle_id'] for rec in recommendations]
                all_recommendations.append(rec_ids)
                
                # Relevant items are those in future interactions
                relevant_items = set(future['bundle_id'].unique())
                
                # Calculate precision and recall for each K
                for k in k_values:
                    if rec_ids:
                        precision = self.precision_at_k(rec_ids, relevant_items, k)
                        recall = self.recall_at_k(rec_ids, relevant_items, k)

                        
                        precision_scores[k].append(precision)
                        recall_scores[k].append(recall)
                
                # Calculate diversity if item features are provided
                if item_features is not None:
                    diversity = self.diversity(rec_ids, item_features)
                    diversity_scores.append(diversity)
                    
                    
            except Exception as e:
                logger.error(f"Error evaluating user {user_id}: {str(e)}")
                continue
        
        # Calculate catalog coverage if we have item features
        coverage = 0
        if item_features is not None:
            catalog_size = len(item_features)
            coverage = self.catalog_coverage(all_recommendations, catalog_size)
        
        # Calculate average scores
        results = {
            'model_name': model_name,
            'user_count': len(user_groups),
            'evaluated_users': len(all_recommendations)

        }
        
        for k in k_values:
            if precision_scores[k]:
                results[f'precision@{k}'] = sum(precision_scores[k]) / len(precision_scores[k])
            else:
                results[f'precision@{k}'] = 0
                
            if recall_scores[k]:
                results[f'recall@{k}'] = sum(recall_scores[k]) / len(recall_scores[k])
            else:
                results[f'recall@{k}'] = 0
        
        if diversity_scores:
            results['diversity'] = sum(diversity_scores) / len(diversity_scores)
        
        if coverage > 0:
            results['coverage'] = coverage
        
        logger.info(f"Evaluation results for {model_name}: {results}")
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def compare_models(self, k: int = 10) -> pd.DataFrame:
        """
        Compare multiple models based on evaluation results.
        
        Args:
            k: K value for Precision@K and Recall@K comparison
            
        Returns:
            DataFrame with model comparison
        """
        # Create comparison DataFrame
        comparison = []
        
        for model_name, results in self.results.items():
            model_results = {
                'Model': model_name,
                f'Precision@{k}': results.get(f'precision@{k}', 0),
                f'Recall@{k}': results.get(f'recall@{k}', 0),
                'Diversity': results.get('diversity', 0),
                'Coverage (%)': results.get('coverage', 0)
            }
            comparison.append(model_results)
            
        return pd.DataFrame(comparison)
    
    def plot_precision_recall_curves(self, k_values: List[int] = [5, 10, 20], figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot precision and recall curves for different K values.
        
        Args:
            k_values: List of K values to plot
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Create subplots for precision and recall
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        
        # Precision plot
        precision_data = []
        for model_name in model_names:
            for k in k_values:
                precision_data.append({
                    'Model': model_name,
                    'K': k,
                    'Precision': self.results[model_name].get(f'precision@{k}', 0)
                })
                
        precision_df = pd.DataFrame(precision_data)
        sns.lineplot(data=precision_df, x='K', y='Precision', hue='Model', marker='o', ax=ax1)
        ax1.set_title('Precision@K for Different Models')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Precision')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Recall plot
        recall_data = []
        for model_name in model_names:
            for k in k_values:
                recall_data.append({
                    'Model': model_name,
                    'K': k,
                    'Recall': self.results[model_name].get(f'recall@{k}', 0)
                })
                
        recall_df = pd.DataFrame(recall_data)
        sns.lineplot(data=recall_df, x='K', y='Recall', hue='Model', marker='o', ax=ax2)
        ax2.set_title('Recall@K for Different Models')
        ax2.set_xlabel('K')
        ax2.set_ylabel('Recall')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
    def plot_diversity_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot diversity comparison between models.
        
        Args:
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Prepare data for plotting
        diversity_data = []
        for model_name, results in self.results.items():
            if 'diversity' in results:
                diversity_data.append({
                    'Model': model_name,
                    'Diversity Score': results['diversity']
                })
                
        if not diversity_data:
            logger.warning("No diversity data available for plotting")
            return
            
        diversity_df = pd.DataFrame(diversity_data)
        
        # Create bar plot
        ax = sns.barplot(data=diversity_df, x='Model', y='Diversity Score')
        ax.set_title('Recommendation Diversity by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Diversity Score (0-1)')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.3f}", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom')
            
        plt.tight_layout()
        
    def plot_coverage_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot catalog coverage comparison between models.
        
        Args:
            figsize: Figure size as (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Prepare data for plotting
        coverage_data = []
        for model_name, results in self.results.items():
            if 'coverage' in results:
                coverage_data.append({
                    'Model': model_name,
                    'Coverage (%)': results['coverage']
                })
                
        if not coverage_data:
            logger.warning("No coverage data available for plotting")
            return
            
        coverage_df = pd.DataFrame(coverage_data)
        
        # Create bar plot
        ax = sns.barplot(data=coverage_df, x='Model', y='Coverage (%)')
        ax.set_title('Catalog Coverage by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Coverage (%)')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.1f}%", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom')
            
        plt.tight_layout()
