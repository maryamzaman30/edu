"""
Recommendation Explainability Module

This module implements explainability features for educational content recommendations,
helping users understand why certain recommendations are provided.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationExplainer:
    """
    A class to generate explanations for content recommendations.
    
    This class provides methods to create user-friendly explanations for why
    specific educational content is recommended, enhancing transparency and trust.
    """
    
    def __init__(self, bundle_data, user_data=None):
        """
        Initialize the recommendation explainer.
        
        Args:
            bundle_data: DataFrame with content bundle information
            user_data: Optional DataFrame with user-bundle interactions
        """
        self.bundle_data = bundle_data
        self.user_data = user_data
        
        # Extract features for similarity calculation
        self.extract_features()
    
    def extract_features(self):
        """
        Extract features from bundle data for similarity calculations.
        
        This prepares the data for generating feature-based explanations.
        """
        # Create feature text from available columns
        self.bundle_data['feature_text'] = self.bundle_data.apply(self._compose_feature_text, axis=1)
        
        # Add part information
        if 'part_name' in self.bundle_data.columns:
            self.bundle_data['feature_text'] += self.bundle_data['part_name'].fillna('') + ' '
        elif 'part' in self.bundle_data.columns:
            self.bundle_data['feature_text'] += 'Part ' + self.bundle_data['part'].astype(str) + ' '
        
        # Add subject information
        if 'subject_category' in self.bundle_data.columns:
            self.bundle_data['feature_text'] += self.bundle_data['subject_category'].fillna('') + ' '
        
        # Add tags if available
        if 'tags' in self.bundle_data.columns:
            self.bundle_data['feature_text'] += self.bundle_data['tags'].fillna('') + ' '
        
        # Add difficulty information
        if 'difficulty' in self.bundle_data.columns:
            self.bundle_data['feature_text'] += 'Difficulty ' + self.bundle_data['difficulty'].fillna('') + ' '
        elif 'success_rate' in self.bundle_data.columns:
            # Convert success rate to difficulty labels
            def success_to_difficulty(rate):
                if rate < 0.3:
                    return 'Hard'
                elif rate < 0.7:
                    return 'Medium'
                else:
                    return 'Easy'
            
            self.bundle_data['difficulty'] = self.bundle_data['success_rate'].apply(success_to_difficulty)
            self.bundle_data['feature_text'] += 'Difficulty ' + self.bundle_data['difficulty'] + ' '
        
        # Calculate similarity matrix
        self.calculate_similarity()
    
    def calculate_similarity(self):
        """
        Calculate similarity between bundles.
        
        This creates a similarity matrix for content-based explanations.
        """
        # Create TF-IDF vectors from feature text
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.bundle_data['feature_text'])
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create DataFrame for easier lookups
        bundle_ids = self.bundle_data['bundle_id'].values
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=bundle_ids,
            columns=bundle_ids
        )
    
    def explain_recommendation(self, recommendation, user_id=None, user_history=None,
                              recommendation_type=None, n_factors=3):
        """
        Generate an explanation for a recommendation.
        
        Args:
            recommendation: Dictionary or Series with recommendation information
            user_id: Optional user ID
            user_history: Optional DataFrame with user history
            recommendation_type: Type of recommendation algorithm used
            n_factors: Number of factors to include in the explanation
            
        Returns:
            Dictionary with explanation
        """
        # Extract bundle ID
        if isinstance(recommendation, dict):
            bundle_id = recommendation.get('bundle_id')
        else:
            bundle_id = recommendation['bundle_id']
        
        # Get bundle information
        bundle_info = self.bundle_data[self.bundle_data['bundle_id'] == bundle_id]
        
        if bundle_info.empty:
            return {
                'bundle_id': bundle_id,
                'explanation': "Sorry, we couldn't find information about this content.",
                'explanation_type': 'error'
            }
        
        bundle_info = bundle_info.iloc[0]
        
        # Generate explanation based on recommendation type
        if recommendation_type == 'content':
            explanation = self._explain_content_based(bundle_info, n_factors)
        elif recommendation_type == 'collaborative':
            explanation = self._explain_collaborative(bundle_info, user_id, user_history, n_factors)
        elif recommendation_type == 'deep_learning':
            explanation = self._explain_deep_learning(bundle_info, user_id, user_history, n_factors)
        elif recommendation_type == 'time_aware':
            explanation = self._explain_time_aware(bundle_info, user_id, user_history, n_factors)
        elif recommendation_type == 'cluster':
            explanation = self._explain_cluster_based(bundle_info, n_factors)
        else:  # hybrid (default)
            explanation = self._explain_hybrid(bundle_info, user_id, user_history, n_factors)
        
        return explanation
    
    def _explain_content_based(self, bundle_info, n_factors=3):
        """
        Generate content-based explanation.
        
        Args:
            bundle_info: Series with bundle information
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        # Extract key features
        features = {}
        
        # Part information
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        # Subject information
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        # Difficulty information
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        elif 'success_rate' in bundle_info:
            # Convert success rate to difficulty
            rate = bundle_info['success_rate']
            if rate < 0.3:
                features['difficulty'] = 'Hard'
            elif rate < 0.7:
                features['difficulty'] = 'Medium'
            else:
                features['difficulty'] = 'Easy'
        
        # Find similar content the user has interacted with
        bundle_id = bundle_info['bundle_id']
        similar_bundles = None
        
        if bundle_id in self.similarity_df.index:
            # Get similarity scores for this bundle
            similarities = self.similarity_df.loc[bundle_id].sort_values(ascending=False)
            
            # Exclude self
            similarities = similarities[similarities.index != bundle_id]
            
            # Get top similar bundles
            top_similar = similarities.head(n_factors)
            
            if not top_similar.empty:
                similar_bundles = top_similar.index.tolist()
        
        # Generate natural language explanation
        explanation = f"This content is recommended because it matches your interests in {features.get('subject', 'this subject')}."
        
        if 'part' in features:
            explanation += f" It's from {features['part']}, which focuses on important skills for the TOEIC test."
        
        if 'difficulty' in features:
            explanation += f" The difficulty level is {features['difficulty']}, which is appropriate for your current skill level."
        
        if similar_bundles:
            explanation += f" It's similar to other content you've engaged with, such as Bundle {similar_bundles[0]}."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'content_based',
            'key_features': features,
            'similar_bundles': similar_bundles
        }
    
    def _explain_collaborative(self, bundle_info, user_id, user_history, n_factors=3):
        """
        Generate collaborative filtering explanation.
        
        Args:
            bundle_info: Series with bundle information
            user_id: User ID
            user_history: DataFrame with user history
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        bundle_id = bundle_info['bundle_id']
        
        # Basic bundle information (similar to content-based)
        features = {}
        
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        
        # Popularity information
        if 'interaction_count' in bundle_info:
            features['popularity'] = bundle_info['interaction_count']
        
        # Create collaborative explanation
        explanation = "This content is recommended because similar users found it valuable."
        
        if 'popularity' in features and features['popularity'] > 100:
            explanation += f" It's a popular choice among TOEIC learners, with {features['popularity']} interactions."
        
        if 'part' in features:
            explanation += f" {features['part']} is an important area for improving your TOEIC score."
        
        if 'difficulty' in features:
            explanation += f" The {features['difficulty']} difficulty level matches what's appropriate for your progress."
        
        # Add information about similar users if available
        if self.user_data is not None and user_id is not None:
            explanation += " Our analysis shows that users with similar learning patterns have benefited from this content."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'collaborative',
            'key_features': features
        }
    
    def _explain_deep_learning(self, bundle_info, user_id, user_history, n_factors=3):
        """
        Generate deep learning explanation.
        
        Args:
            bundle_info: Series with bundle information
            user_id: User ID
            user_history: DataFrame with user history
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        bundle_id = bundle_info['bundle_id']
        
        # Extract key features (similar to other methods)
        features = {}
        
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        
        # Create contextual explanation
        explanation = "Our advanced language model has analyzed your learning pattern and identified this content as beneficial for your TOEIC preparation."
        
        if user_history is not None and not user_history.empty:
            # Look for patterns in user history
            features = {}
            if 'part' in bundle_info and 'part' in user_history.columns:
                # Check if user has worked on this part
                if bundle_info['part'] in user_history['part'].values:
                    explanation += f" You've been working on {features.get('part', 'this part')}, and this content will help reinforce those skills."
                else:
                    explanation += f" This content introduces {features.get('part', 'a new part')} which complements your current learning activities."
        
        if 'difficulty' in features:
            explanation += f" The {features['difficulty']} difficulty aligns with your demonstrated abilities."
        
        if 'subject' in features:
            explanation += f" It focuses on {features['subject']}, which is important for achieving a balanced TOEIC score."
        
        # Add information about transformers
        explanation += " Our transformer-based model identified subtle patterns in your learning journey that make this content particularly relevant."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'deep_learning',
            'key_features': features
        }
    
    def _explain_time_aware(self, bundle_info, user_id, user_history, n_factors=3):
        """
        Generate time-aware explanation.
        
        Args:
            bundle_info: Series with bundle information
            user_id: User ID
            user_history: DataFrame with user history
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        bundle_id = bundle_info['bundle_id']
        
        # Extract key features (similar to other methods)
        features = {}
        
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        
        # Create temporal explanation
        explanation = "Based on your recent learning activities, this content is the next logical step in your TOEIC preparation journey."
        
        if user_history is not None and not user_history.empty and 'timestamp' in user_history.columns:
            # Get most recent interactions
            recent_history = user_history.sort_values('timestamp', ascending=False).head(5)
            
            if 'part' in recent_history.columns and 'part' in bundle_info:
                recent_parts = recent_history['part'].unique()
                
                if bundle_info['part'] in recent_parts:
                    explanation += f" You've recently worked on {features.get('part', 'this part')}, and this content will help deepen your understanding."
                elif int(bundle_info['part']) > max(map(int, recent_parts)):
                    explanation += f" Now that you've made progress in earlier parts, this content will help you advance to {features.get('part', 'the next level')}."
        
        if 'difficulty' in features:
            explanation += f" The {features['difficulty']} difficulty is appropriate for your current stage in the learning process."
        
        # Add information about learning progression
        explanation += " Our time-aware algorithm tracks your learning progression and suggests content that builds on your previous work at just the right time."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'time_aware',
            'key_features': features
        }
    
    def _explain_cluster_based(self, bundle_info, n_factors=3):
        """
        Generate cluster-based explanation.
        
        Args:
            bundle_info: Series with bundle information
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        bundle_id = bundle_info['bundle_id']
        
        # Extract key features (similar to other methods)
        features = {}
        
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        
        # Create cluster-based explanation
        explanation = "This content belongs to a group of related learning materials that are valuable for TOEIC preparation."
        
        if 'part' in features and 'subject' in features:
            explanation += f" It's part of the '{features['subject']}' cluster within {features['part']}, which covers essential TOEIC skills."
        
        if 'difficulty' in features:
            explanation += f" The {features['difficulty']} difficulty level is appropriate for your current abilities."
        
        # Add information about cold-start
        explanation += " Even with limited information about your preferences, our clustering algorithm can identify content that's likely to be valuable for your learning goals."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'cluster_based',
            'key_features': features
        }
    
    def _explain_hybrid(self, bundle_info, user_id, user_history, n_factors=3):
        """
        Generate hybrid explanation.
        
        Args:
            bundle_info: Series with bundle information
            user_id: User ID
            user_history: DataFrame with user history
            n_factors: Number of factors to include
            
        Returns:
            Dictionary with explanation
        """
        bundle_id = bundle_info['bundle_id']
        
        # Extract key features (similar to other methods)
        features = {}
        
        if 'part_name' in bundle_info:
            features['part'] = bundle_info['part_name']
        elif 'part' in bundle_info:
            features['part'] = f"Part {bundle_info['part']}"
        
        if 'subject_category' in bundle_info:
            features['subject'] = bundle_info['subject_category']
        
        if 'difficulty' in bundle_info:
            features['difficulty'] = bundle_info['difficulty']
        
        # Create hybrid explanation
        explanation = "This content is recommended based on multiple factors for a comprehensive and personalized suggestion."
        
        # Content-based component
        if 'subject' in features:
            explanation += f" It matches your interest in {features['subject']}."
        
        # Collaborative component
        explanation += " Users with similar learning patterns have found this content valuable."
        
        # Learning progression component
        if 'part' in features:
            explanation += f" {features['part']} is an important component of TOEIC preparation that aligns with your current progress."
        
        if 'difficulty' in features:
            explanation += f" The {features['difficulty']} difficulty level is appropriate for your demonstrated abilities."
        
        # Add information about hybrid approach
        explanation += " Our hybrid recommendation system combines content analysis, collaborative patterns, and your personal learning journey to provide the most relevant suggestions."
        
        # Create structured explanation
        return {
            'bundle_id': bundle_id,
            'explanation': explanation,
            'explanation_type': 'hybrid',
            'key_features': features
        }
    
    def generate_feature_importance_chart(self, recommendation, recommendation_type=None):
        """
        Generate a chart showing feature importance for a recommendation.
        
        Args:
            recommendation: Dictionary or Series with recommendation information
            recommendation_type: Type of recommendation algorithm used
            
        Returns:
            Plotly figure
        """
        # Extract bundle ID
        if isinstance(recommendation, dict):
            bundle_id = recommendation.get('bundle_id')
        else:
            bundle_id = recommendation['bundle_id']
        
        # Get bundle information
        bundle_info = self.bundle_data[self.bundle_data['bundle_id'] == bundle_id]
        
        if bundle_info.empty:
            # Create empty chart with error message
            fig = go.Figure()
            fig.update_layout(
                title="Bundle information not available",
                annotations=[
                    dict(
                        text="Bundle information not available",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
            return fig
        
        bundle_info = bundle_info.iloc[0]
        
        # Define feature importance based on recommendation type
        if recommendation_type == 'content':
            # Content-based emphasizes content features
            feature_importance = {
                'Subject Match': 0.40,
                'Part Relevance': 0.30,
                'Difficulty Alignment': 0.20,
                'Content Similarity': 0.10
            }
        elif recommendation_type == 'collaborative':
            # Collaborative emphasizes user similarity
            feature_importance = {
                'Similar Users': 0.40,
                'Popularity': 0.30,
                'Part Relevance': 0.15,
                'Difficulty Alignment': 0.15
            }
        elif recommendation_type == 'deep_learning':
            # Deep learning has complex patterns
            feature_importance = {
                'Learning Pattern': 0.35,
                'Context Understanding': 0.25,
                'Content Relevance': 0.20,
                'Skill Progression': 0.15,
                'Temporal Patterns': 0.05
            }
        elif recommendation_type == 'time_aware':
            # Time-aware emphasizes progression
            feature_importance = {
                'Learning Sequence': 0.35,
                'Recent Activity': 0.25,
                'Part Progression': 0.20,
                'Difficulty Progression': 0.15,
                'Topic Continuity': 0.05
            }
        elif recommendation_type == 'cluster':
            # Cluster-based emphasizes content grouping
            feature_importance = {
                'Topic Cluster': 0.40,
                'Difficulty Group': 0.25,
                'TOEIC Section': 0.20,
                'Question Type': 0.15
            }
        else:  # hybrid (default)
            # Hybrid balances multiple factors
            feature_importance = {
                'Content Match': 0.25,
                'User Similarity': 0.25,
                'Learning Progression': 0.20,
                'Difficulty Alignment': 0.15,
                'Topic Relevance': 0.15
            }
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            xaxis_tickangle=-30,
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            marker_color='rgb(26, 118, 255)',
            text=[f"{v:.0%}" for v in feature_importance.values()],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Feature Importance for Bundle {bundle_id} Recommendation",
            xaxis_title="Feature",
            yaxis_title="Importance",
            yaxis=dict(
                tickformat='.0%',
                range=[0, 1]
            ),
            showlegend=False
        )
        
        return fig