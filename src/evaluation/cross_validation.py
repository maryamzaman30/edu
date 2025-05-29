"""
Recommendation Cross-Validation Framework

This module implements a comprehensive cross-validation framework for evaluating
recommendation models with various strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
import time
from collections import defaultdict
import warnings

class RecommendationEvaluator:
    """
    A class to evaluate recommendation models using various cross-validation strategies.
    
    This class provides methods for splitting data, training and evaluating models, and
    calculating relevant metrics for recommendation systems.
    """
    
    def __init__(self, data, user_col='user_id', item_col='item_id', 
                rating_col=None, timestamp_col=None):
        """
        Initialize the recommendation evaluator.
        
        Args:
            data: DataFrame with user-item interactions
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            rating_col: Optional column name for ratings
            timestamp_col: Optional column name for timestamps
        """
        self.data = data
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        
        # Check if data has required columns
        required_cols = [user_col, item_col]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must have columns {required_cols}")
            
        # If rating_col is not specified but exists in data, use it
        if rating_col is None and 'rating' in data.columns:
            self.rating_col = 'rating'
        
        # If timestamp_col is not specified but exists in data, use it
        if timestamp_col is None and 'timestamp' in data.columns:
            self.timestamp_col = 'timestamp'
            
        # Check if interactions are implicit or explicit
        self.is_implicit = self.rating_col is None
        
        # Get unique users and items
        self.users = data[user_col].unique()
        self.items = data[item_col].unique()
        
        # Print info about the dataset
        print(f"Dataset has {len(self.users)} users, {len(self.items)} items, "
              f"and {len(data)} interactions")
        print(f"Data sparsity: {len(data) / (len(self.users) * len(self.items)):.6f}")
    
    def user_level_split(self, n_splits=5, test_size=0.2, random_state=None):
        """
        Split data at the user level.
        
        This ensures all interactions from a user are either in training or test set.
        
        Args:
            n_splits: Number of folds
            test_size: Proportion of users in test set
            random_state: Random seed
            
        Returns:
            List of (train_data, test_data) tuples
        """
        # Create random generator
        rng = np.random.RandomState(random_state)
        
        # Shuffle users
        users = np.array(self.users)
        rng.shuffle(users)
        
        # Create folds
        folds = []
        fold_size = int(len(users) / n_splits)
        
        for i in range(n_splits):
            # Get test users for this fold
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(users)
            test_users = users[start_idx:end_idx]
            
            # Split data
            train_data = self.data[~self.data[self.user_col].isin(test_users)]
            test_data = self.data[self.data[self.user_col].isin(test_users)]
            
            folds.append((train_data, test_data))
        
        return folds
    
    def temporal_split(self, n_splits=5):
        """
        Split data temporally.
        
        This creates folds where training data comes before test data.
        
        Args:
            n_splits: Number of folds
            
        Returns:
            List of (train_data, test_data) tuples
        """
        if self.timestamp_col is None:
            raise ValueError("Temporal split requires timestamp column")
            
        # Sort data by timestamp
        sorted_data = self.data.sort_values(self.timestamp_col)
        
        # Create splitter
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Create folds
        folds = []
        for train_idx, test_idx in tscv.split(sorted_data):
            train_data = sorted_data.iloc[train_idx]
            test_data = sorted_data.iloc[test_idx]
            
            folds.append((train_data, test_data))
        
        return folds
    
    def stratified_item_split(self, n_splits=5, random_state=None):
        """
        Split data with stratification by item properties.
        
        This ensures similar distribution of item types across folds.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
            
        Returns:
            List of (train_data, test_data) tuples
        """
        # Create item-based strata
        # For demonstration, we'll use item popularity as the stratification variable
        item_counts = self.data[self.item_col].value_counts()
        
        # Divide items into quartiles by popularity
        quartiles = pd.qcut(item_counts, 4, labels=False, duplicates='drop')
        # item_strata = pd.Series(quartiles, index=item_counts.index)
        item_strata = quartiles  # already a Series

        
        # Add stratum to each interaction
        data_with_strata = self.data.copy()
        data_with_strata['stratum'] = data_with_strata[self.item_col].map(item_strata)
        
        # Fill missing strata
        data_with_strata['stratum'] = data_with_strata['stratum'].fillna(0)
        
        # Create stratified split
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Create folds
        folds = []
        for train_idx, test_idx in skf.split(data_with_strata, data_with_strata['stratum']):
            train_data = data_with_strata.iloc[train_idx]
            test_data = data_with_strata.iloc[test_idx]
            
            # Remove temporary column
            train_data = train_data.drop('stratum', axis=1)
            test_data = test_data.drop('stratum', axis=1)
            
            folds.append((train_data, test_data))
        
        return folds
    
    def leave_one_out_split(self, n_samples=None, random_state=None):
        """
        Create a leave-one-out split.
        
        This hides the last interaction for each user for testing.
        
        Args:
            n_samples: Optional number of users to sample
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.timestamp_col is None:
            # If no timestamp, we'll use a random interaction for each user
            rng = np.random.RandomState(random_state)
            
            # Get users to sample
            users = self.users
            if n_samples is not None and n_samples < len(users):
                rng.shuffle(users)
                users = users[:n_samples]
            
            # Create split
            train_data = []
            test_data = []
            
            for user in users:
                user_data = self.data[self.data[self.user_col] == user]
                
                if len(user_data) <= 1:
                    # Skip users with only one interaction
                    continue
                
                # Select a random interaction for testing
                test_idx = rng.randint(0, len(user_data))
                
                # Add to train and test data
                train_data.append(user_data.drop(user_data.index[test_idx]))
                test_data.append(user_data.iloc[test_idx:test_idx+1])
            
            # Combine data
            train_data = pd.concat(train_data)
            test_data = pd.concat(test_data)
        else:
            # If timestamp available, use the last interaction for each user
            # Get users to sample
            users = self.users
            if n_samples is not None and n_samples < len(users):
                rng = np.random.RandomState(random_state)
                rng.shuffle(users)
                users = users[:n_samples]
            
            # Create split
            train_data = []
            test_data = []
            
            for user in users:
                user_data = self.data[self.data[self.user_col] == user]
                
                if len(user_data) <= 1:
                    # Skip users with only one interaction
                    continue
                
                # Sort by timestamp
                user_data = user_data.sort_values(self.timestamp_col)
                
                # Add to train and test data
                train_data.append(user_data.iloc[:-1])
                test_data.append(user_data.iloc[-1:])
            
            # Combine data
            train_data = pd.concat(train_data)
            test_data = pd.concat(test_data)
        
        return train_data, test_data
    
    def evaluate_model(self, model, train_data, test_data, k=10, **kwargs):
        """
        Evaluate a recommendation model.
        
        Args:
            model: Model to evaluate
            train_data: Training data
            test_data: Test data
            k: Number of recommendations to consider
            **kwargs: Additional parameters for prediction
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Train model
        start_time = time.time()
        model.fit(train_data)
        training_time = time.time() - start_time
        
        # Get predictions
        start_time = time.time()
        predictions = model.predict(test_data, k=k, **kwargs)
        prediction_time = time.time() - start_time
        
        # Get ground truth
        ground_truth = self._extract_ground_truth(test_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truth, k=k)
        metrics['training_time'] = training_time
        metrics['prediction_time'] = prediction_time
        
        return metrics
    
    def _extract_ground_truth(self, test_data):
        """
        Extract ground truth from test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary mapping users to lists of relevant items
        """
        ground_truth = {}
        
        for user, user_data in test_data.groupby(self.user_col):
            # For explicit ratings, consider items with high ratings as relevant
            if not self.is_implicit and self.rating_col in user_data.columns:
                # Use median rating as threshold
                threshold = user_data[self.rating_col].median()
                relevant_items = user_data[user_data[self.rating_col] >= threshold][self.item_col].tolist()
            else:
                # For implicit feedback, all items are relevant
                relevant_items = user_data[self.item_col].tolist()
                
            ground_truth[user] = relevant_items
        
        return ground_truth
    
    def _calculate_metrics(self, predictions, ground_truth, k=10):
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted items for each user
            ground_truth: Ground truth items for each user
            k: Number of recommendations to consider
            
        Returns:
            Dictionary with metrics
        """
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        hit_rate = []
        
        for user, pred_items in predictions.items():
            if user not in ground_truth:
                continue
                
            relevant_items = ground_truth[user]
            
            # Get the top-k predicted items
            top_k_items = pred_items[:k]
            
            # Precision@k
            n_relevant_and_recommended = len(set(top_k_items) & set(relevant_items))
            precision = n_relevant_and_recommended / min(k, len(top_k_items)) if top_k_items else 0
            precision_at_k.append(precision)
            
            # Recall@k
            recall = n_relevant_and_recommended / len(relevant_items) if relevant_items else 0
            recall_at_k.append(recall)
            
            # NDCG@k
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k.append(ndcg)
            
            # Hit rate
            hit = 1.0 if n_relevant_and_recommended > 0 else 0.0
            hit_rate.append(hit)
        
        # Calculate coverage
        unique_rec_items = set()
        for pred_items in predictions.values():
            top_k_items = pred_items[:k]
            unique_rec_items.update(top_k_items)
        
        coverage = len(unique_rec_items) / len(self.items) * 100
        
        return {
            'precision@k': np.mean(precision_at_k) if precision_at_k else 0,
            'recall@k': np.mean(recall_at_k) if recall_at_k else 0,
            'ndcg@k': np.mean(ndcg_at_k) if ndcg_at_k else 0,
            'hit_rate': np.mean(hit_rate) if hit_rate else 0,
            'coverage': coverage
        }
    
    def cross_validate(self, model, cv_method='user_level', n_splits=5, k=10, 
                      random_state=None, **kwargs):
        """
        Perform cross-validation on a recommendation model.
        
        Args:
            model: Model to evaluate
            cv_method: Cross-validation method
            n_splits: Number of folds
            k: Number of recommendations to consider
            random_state: Random seed
            **kwargs: Additional parameters for prediction
            
        Returns:
            Dictionary with evaluation results
        """
        # Get CV splits
        if cv_method == 'user_level':
            folds = self.user_level_split(n_splits=n_splits, random_state=random_state)
        elif cv_method == 'temporal':
            folds = self.temporal_split(n_splits=n_splits)
        elif cv_method == 'stratified_item':
            folds = self.stratified_item_split(n_splits=n_splits, random_state=random_state)
        elif cv_method == 'leave_one_out':
            # Leave-one-out is a single split
            train_data, test_data = self.leave_one_out_split(random_state=random_state)
            folds = [(train_data, test_data)]
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        # Evaluate model on each fold
        results = {
            'precision@k': [],
            'recall@k': [],
            'ndcg@k': [],
            'hit_rate': [],
            'coverage': [],
            'training_time': [],
            'prediction_time': []
        }
        
        for i, (train_data, test_data) in enumerate(folds):
            print(f"Evaluating fold {i+1}/{len(folds)}")
            
            # Evaluate model
            fold_results = self.evaluate_model(model, train_data, test_data, k=k, **kwargs)
            
            # Add to results
            for metric, value in fold_results.items():
                results[metric].append(value)
        
        # Calculate statistics
        stats = {}
        for metric, values in results.items():
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
        
        return stats
    
    def plot_performance_comparison(self, results, metrics=None, figsize=(12, 8)):
        """
        Plot performance comparison between models.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            metrics: Optional list of metrics to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['precision@k_mean', 'recall@k_mean', 'ndcg@k_mean', 'hit_rate_mean']
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            model_names = []
            metric_values = []
            errors = []
            
            for model_name, model_results in results.items():
                if metric in model_results:
                    model_names.append(model_name)
                    metric_values.append(model_results[metric])
                    
                    # Get error if available
                    error_metric = metric.replace('_mean', '_std')
                    if error_metric in model_results:
                        errors.append(model_results[error_metric])
                    else:
                        errors.append(0)
            
            # Plot bar chart
            ax = axes[i]
            x_pos = np.arange(len(model_names))
            ax.bar(x_pos, metric_values, yerr=errors, align='center', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} comparison')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_training_time_comparison(self, results, figsize=(10, 6)):
        """
        Plot training time comparison between models.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        model_names = []
        training_times = []
        errors = []
        
        for model_name, model_results in results.items():
            if 'training_time_mean' in model_results:
                model_names.append(model_name)
                training_times.append(model_results['training_time_mean'])
                
                # Get error if available
                if 'training_time_std' in model_results:
                    errors.append(model_results['training_time_std'])
                else:
                    errors.append(0)
        
        # Plot bar chart
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, training_times, yerr=errors, align='center', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Training time (seconds)')
        ax.set_title('Training time comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_metric_heatmap(self, results, metric='precision@k_mean', figsize=(10, 8)):
        """
        Plot heatmap of a metric across models and CV methods.
        
        Args:
            results: Dictionary mapping (model_name, cv_method) to evaluation results
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract unique model names and CV methods
        model_names = list(set(model for model, _ in results.keys()))
        cv_methods = list(set(cv for _, cv in results.keys()))
        
        # Create data matrix
        data = np.zeros((len(model_names), len(cv_methods)))
        
        for i, model in enumerate(model_names):
            for j, cv in enumerate(cv_methods):
                        if (model, cv) in results and metric in results[(model, cv)]:
                            data[i, j] = results[(model, cv)][metric]
                        else:
                            data[i, j] = np.nan

        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=cv_methods, yticklabels=model_names)
        
        ax.set_xlabel('Cross-validation method')
        ax.set_ylabel('Model')
        ax.set_title(f'{metric} heatmap')
        
        plt.tight_layout()
        return fig
    
    def interactive_comparison_plot(self, results, metrics=None):
        """
        Create an interactive plot comparing models.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            metrics: Optional list of metrics to plot
            
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['precision@k_mean', 'recall@k_mean', 'ndcg@k_mean', 'hit_rate_mean']
        
        # Prepare data
        model_names = []
        plot_data = []
        
        for model_name, model_results in results.items():
            model_names.append(model_name)
            
            # Extract metrics for this model
            model_metrics = []
            model_errors = []
            
            for metric in metrics:
                if metric in model_results:
                    model_metrics.append(model_results[metric])
                    
                    # Get error if available
                    error_metric = metric.replace('_mean', '_std')
                    if error_metric in model_results:
                        model_errors.append(model_results[error_metric])
                    else:
                        model_errors.append(0)
                else:
                    model_metrics.append(0)
                    model_errors.append(0)
            
            # Create trace for this model
            trace = go.Bar(
                name=model_name,
                x=metrics,
                y=model_metrics,
                error_y=dict(
                    type='data',
                    array=model_errors,
                    visible=True
                )
            )
            
            plot_data.append(trace)
        
        # Create figure
        fig = go.Figure(data=plot_data)
        
        # Update layout
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=600
        )
        fig.update_yaxes(range=[0, max(max(metric_values) for metric_values in [trace.y for trace in plot_data]) * 1.1])

        
        return fig
    
    def interactive_radar_chart(self, results, metrics=None):
        """
        Create an interactive radar chart comparing models.
        
        Args:
            results: Dictionary mapping model names to evaluation results
            metrics: Optional list of metrics to plot
            
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ['precision@k_mean', 'recall@k_mean', 'ndcg@k_mean', 'hit_rate_mean', 'coverage_mean']
        
        # Prepare data
        plot_data = []
        
        # Normalize metrics
        max_values = {}
        for metric in metrics:
            max_values[metric] = max([results[model].get(metric, 0) for model in results])
        
        for model_name, model_results in results.items():
            # Extract metrics for this model
            model_metrics = []
            
            for metric in metrics:
                if metric in model_results and max_values[metric] > 0:
                    # Normalize value
                    model_metrics.append(model_results[metric] / max_values[metric])
                else:
                    model_metrics.append(0)
            
            # Add first point again to close the polygon
            model_metrics.append(model_metrics[0])
            metric_labels = metrics + [metrics[0]]
            
            # Create trace for this model
            trace = go.Scatterpolar(
                r=model_metrics,
                theta=metric_labels,
                fill='toself',
                name=model_name
            )
            
            plot_data.append(trace)
        
        # Create figure
        fig = go.Figure(data=plot_data)
        
        # Update layout
        fig.update_layout(
            title='Model Comparison',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=600
        )
        
        return fig

def compare_models(data, models, cv_method='user_level', n_splits=5, k=10, plot=True,
                 random_state=None, user_col='user_id', item_col='item_id',
                 rating_col=None, timestamp_col=None, **kwargs):
    """
    Compare multiple recommendation models.
    
    Args:
        data: DataFrame with user-item interactions
        models: Dictionary mapping model names to model instances
        cv_method: Cross-validation method
        n_splits: Number of folds
        k: Number of recommendations to consider
        random_state: Random seed
        user_col: Column name for user IDs
        item_col: Column name for item IDs
        rating_col: Optional column name for ratings
        timestamp_col: Optional column name for timestamps
        **kwargs: Additional parameters for prediction
        
    Returns:
        Dictionary mapping model names to evaluation results
    """
    # Create evaluator
    evaluator = RecommendationEvaluator(
        data, 
        user_col=user_col, 
        item_col=item_col,
        rating_col=rating_col,
        timestamp_col=timestamp_col
    )
    
    # Evaluate each model
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Cross-validate model
        model_results = evaluator.cross_validate(
            model,
            cv_method=cv_method,
            n_splits=n_splits,
            k=k,
            random_state=random_state,
            **kwargs
        )
        
        results[model_name] = model_results
        
        print(f"Results for {model_name}:")
        for metric, value in model_results.items():
            if 'mean' in metric:
                std_metric = metric.replace('mean', 'std')
                std_value = model_results[std_metric]
                print(f"  {metric}: {value:.4f} ± {std_value:.4f}")
    
    # Create comparison plots
    if plot:
        fig1 = evaluator.plot_performance_comparison(results)
        fig2 = evaluator.plot_training_time_comparison(results)
        fig3 = evaluator.interactive_comparison_plot(results)
        fig4 = evaluator.interactive_radar_chart(results)
        return results, (fig1, fig2, fig3, fig4)
    else:
        return results, ()

    
    
