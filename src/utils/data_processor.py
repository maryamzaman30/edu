"""
Data Processor Module for EdNet Recommendation System

This module provides utility functions for loading, processing,
and transforming the EdNet dataset for recommendation models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EdNetDataProcessor:
    """
    Data processor for the EdNet dataset.
    
    This class handles loading, cleaning, and preprocessing data for
    the recommendation system.
    """
    
    def __init__(self, merged_data_path: str, lectures_data_path: str):
        """
        Initialize the data processor with paths to the datasets.
        
        Args:
            merged_data_path: Path to the merged_cleaned_data.csv file
            lectures_data_path: Path to the cleaned_lectures.csv file
        """
        self.merged_data_path = merged_data_path
        self.lectures_data_path = lectures_data_path
        self.merged_data = None
        self.lectures_data = None
        self.user_item_matrix = None
        self.item_features = None
        
        # Maps for content types and subject areas (based on TOEIC structure)
        self.part_names = {
            0: "Introduction",
            1: "Listening Comprehension",
            2: "Reading Comprehension",
            3: "Grammar & Vocabulary",
            4: "Speaking Assessment",
            5: "Writing Exercises",
            6: "Practice Tests",
            7: "Additional Resources"
        }
        
        # Tag mapping (simplified for interpretability)
        # In a real system, this would be more comprehensive based on TOEIC documentation
        self.tag_categories = {
            range(1, 23): "Listening Skills",
            range(23, 52): "Reading Skills",
            range(52, 70): "Speaking Skills",
            range(70, 150): "Writing Skills",
            range(150, 200): "Test Preparation",
            range(200, 300): "Grammar & Vocabulary"
        }
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the data from CSV files.
        
        Returns:
            Tuple containing merged_data and lectures_data DataFrames
        """
        logger.info(f"Loading data from {self.merged_data_path} and {self.lectures_data_path}")
        
        try:
            self.merged_data = pd.read_csv(self.merged_data_path)
            self.lectures_data = pd.read_csv(self.lectures_data_path)
            
            logger.info(f"Loaded merged data: {self.merged_data.shape} rows")
            logger.info(f"Loaded lectures data: {self.lectures_data.shape} rows")
            
            return self.merged_data, self.lectures_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and preprocess the data for the recommendation models.
        
        Returns:
            Tuple containing cleaned merged_data and lectures_data
        """
        if self.merged_data is None or self.lectures_data is None:
            self.load_data()
            
        logger.info("Cleaning and preprocessing data")
        
        # Process merged data
        # Convert timestamp to datetime
        self.merged_data['timestamp'] = pd.to_datetime(self.merged_data['timestamp'])
        
        # Handle missing values in lectures data
        self.lectures_data['video_length'] = self.lectures_data['video_length'].fillna(0)
        self.lectures_data['deployed_at'] = pd.to_datetime(self.lectures_data['deployed_at'])
        
        # Convert tags to string to ensure type consistency
        self.lectures_data['tags'] = self.lectures_data['tags'].astype(str)
        
        # Map part numbers to human-readable names
        self.lectures_data['part_name'] = self.lectures_data['part'].map(self.part_names)
                
       # self.lectures_data['subject_category'] = self.lectures_data['tags'].apply(get_tag_category)
        self.lectures_data['subject_category'] = self.lectures_data['tags'].apply(self.get_tag_category)

        
        logger.info("Data cleaning and preprocessing complete")
        return self.merged_data, self.lectures_data
    
        # Create subject category based on tags
    def get_tag_category(tag_val, self):
        try:
            tag_num = float(tag_val)
            for range_obj, category in self.tag_categories.items():
                if tag_num in range_obj:
                    return category
            return "General"
        except:
            return "General"
            
    def create_user_item_matrix(self, min_interactions: int = 5) -> pd.DataFrame:
        """
        Create a user-item interaction matrix from the merged data.
        
        Args:
            min_interactions: Minimum number of interactions a user must have to be included
            
        Returns:
            User-item interaction matrix as a DataFrame
        """
        if self.merged_data is None:
            self.clean_and_preprocess()
            
        logger.info("Creating user-item interaction matrix")
        
        # Filter users with minimum interactions
        user_counts = self.merged_data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        filtered_data = self.merged_data[self.merged_data['user_id'].isin(valid_users)]
        
        # Create correctness as a measure of interaction quality
        # 1 if user_answer matches correct_answer, otherwise 0
       # filtered_data['correct'] = (filtered_data['user_answer'] == filtered_data['correct_answer']).astype(int)
        if 'correct' not in filtered_data.columns:
            if 'user_answer' in filtered_data.columns and 'correct_answer' in filtered_data.columns:
                filtered_data['correct'] = (filtered_data['user_answer'] == filtered_data['correct_answer']).astype(int)
            else:
                raise ValueError("Missing required columns: 'user_answer' or 'correct_answer' to compute 'correct'")


            
        # Use bundle_id as the item identifier for recommendations
        # This groups questions into logical bundles
        user_item_data = filtered_data.groupby(['user_id', 'bundle_id']).agg({
            'correct': ['mean', 'count'],
            'elapsed_time': 'mean',
            'part': 'first',
            'tags': lambda x: ';'.join(set(str(i) for i in x))
        })
        
        user_item_data.columns = ['correctness_rate', 'interaction_count', 'avg_time', 'part', 'tags']
        user_item_data = user_item_data.reset_index()
        
        # Create pivot table with interaction count as values
        self.user_item_matrix = user_item_data.pivot_table(
            index='user_id', 
            columns='bundle_id', 
            values='interaction_count',
            fill_value=0
        )
        
        logger.info(f"Created user-item matrix with {self.user_item_matrix.shape[0]} users and {self.user_item_matrix.shape[1]} items")
        return self.user_item_matrix
    
    def create_item_features(self) -> pd.DataFrame:
        """
        Create item features for content-based filtering.
        
        Returns:
            DataFrame with item features
        """
        if self.merged_data is None or self.lectures_data is None:
            self.clean_and_preprocess()
            
        logger.info("Creating item features")
        
        # Extract unique bundle information from merged data
        bundle_info = self.merged_data.groupby('bundle_id').agg({
            'part': 'first',
            'tags': lambda x: ';'.join(set(str(i) for i in x if pd.notna(i))),
            'question_id': lambda x: len(set(x))  # Number of questions in bundle
        }).reset_index()
        
        # Rename columns for clarity
        bundle_info.columns = ['bundle_id', 'part', 'tags', 'question_count']
        
        # Map part to human-readable names
        bundle_info['part_name'] = bundle_info['part'].map(self.part_names)
        
        # Create subject category based on tags
        bundle_info['subject_category'] = bundle_info['tags'].apply(self.get_tag_category)
        
        # Calculate bundle popularity from interaction data
        bundle_popularity = self.merged_data['bundle_id'].value_counts().reset_index()
        bundle_popularity.columns = ['bundle_id', 'interaction_count']
        
        # Calculate bundle difficulty from correct answer rates
        bundle_difficulty = self.merged_data.groupby('bundle_id').apply(
            lambda x: (x['user_answer'] == x['correct_answer']).mean()
        ).reset_index()
        bundle_difficulty.columns = ['bundle_id', 'success_rate']
        
        # Merge all features
        self.item_features = bundle_info.merge(bundle_popularity, on='bundle_id', how='left')
        self.item_features = self.item_features.merge(bundle_difficulty, on='bundle_id', how='left')
        
        # Fill missing values
        self.item_features['interaction_count'] = self.item_features['interaction_count'].fillna(0)
        self.item_features['success_rate'] = self.item_features['success_rate'].fillna(0.5)
        
        logger.info(f"Created item features for {len(self.item_features)} bundles")
        return self.item_features
    
    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """
        Get the interaction history for a specific user.
        
        Args:
            user_id: The user ID to get history for
            
        Returns:
            DataFrame with the user's interaction history
        """
        if self.merged_data is None:
            self.clean_and_preprocess()
            
        user_data = self.merged_data[self.merged_data['user_id'] == user_id]
        if len(user_data) == 0:
            logger.warning(f"No history found for user {user_id}")
            return pd.DataFrame()
            
        return user_data
    
    def train_test_split_interactions(self, test_size: float = 0.2, time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the interaction data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            time_based: If True, split based on time (latest interactions in test set)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.merged_data is None:
            self.clean_and_preprocess()
            
        logger.info(f"Splitting data with test_size={test_size}, time_based={time_based}")
        
        if time_based:
            # Time-based split (most recent interactions as test set)
            self.merged_data = self.merged_data.sort_values('timestamp')
            split_idx = int(len(self.merged_data) * (1 - test_size))
            train_df = self.merged_data.iloc[:split_idx]
            test_df = self.merged_data.iloc[split_idx:]
        else:
            # Random split
            train_df, test_df = train_test_split(self.merged_data, test_size=test_size, random_state=42)
            
        logger.info(f"Split data into train set ({len(train_df)} rows) and test set ({len(test_df)} rows)")
        return train_df, test_df
    
    def get_tag_category(self, tag_val):
        """Helper function to get tag category from tag value"""
        try:
            tag_num = float(tag_val)
            for range_obj, category in self.tag_categories.items():
                if tag_num in range_obj:
                    return category
            return "General"
        except:
            return "General"

# Helper function to get human-readable names for a bundle
def get_bundle_info(bundle_id: str, item_features: pd.DataFrame) -> Dict:
    """
    Get human-readable information for a bundle.
    
    Args:
        bundle_id: The bundle ID
        item_features: DataFrame with item features
        
    Returns:
        Dictionary with human-readable information
    """
    bundle_row = item_features[item_features['bundle_id'] == bundle_id]
    if len(bundle_row) == 0:
        return {
            'bundle_id': bundle_id,
            'title': f"Bundle {bundle_id}",
            'part': "Unknown",
            'subject': "Unknown",
            'difficulty': "Unknown",
            'popularity': 0
        }
        
    row = bundle_row.iloc[0]
    
    # Calculate difficulty level based on success rate
    if row['success_rate'] < 0.3:
        difficulty = "Hard"
    elif row['success_rate'] < 0.7:
        difficulty = "Medium"
    else:
        difficulty = "Easy"
    
    return {
        'bundle_id': bundle_id,
        'title': f"Bundle {bundle_id}: {row['part_name']}",
        'part': row['part_name'],
        'subject': row['subject_category'],
        'difficulty': difficulty,
        'question_count': row['question_count'],
        'popularity': int(row['interaction_count']),
        'success_rate': float(row['success_rate'])
    }
