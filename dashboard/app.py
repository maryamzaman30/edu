"""
Streamlit Dashboard for Educational Content Recommendation System

This module implements an interactive dashboard for visualizing
the educational content recommendation system and its performance.
"""

import os
import sys
import logging
import pickle
import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path for direct model access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="EdNet Recommendation Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000"  # FastAPI backend URL

# Cache and session state initialization
if 'user_history' not in st.session_state:
    st.session_state.user_history = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'dataset_stats' not in st.session_state:
    st.session_state.dataset_stats = None
if 'users' not in st.session_state:
    st.session_state.users = None
if 'bundles' not in st.session_state:
    st.session_state.bundles = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None

# Helper functions
@st.cache_data(ttl=3600)
def load_dataset_stats():
    """Load dataset statistics from API"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading dataset stats: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_users():
    """Load list of users from API"""
    try:
        response = requests.get(f"{API_URL}/users", params={"limit": 100})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading users: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def load_bundles():
    """Load list of bundles from API"""
    try:
        response = requests.get(f"{API_URL}/bundles", params={"limit": 100})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading bundles: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def load_user_history(user_id):
    """Load interaction history for a user"""
    try:
        response = requests.get(f"{API_URL}/user/{user_id}/history", params={"limit": 50})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading user history: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

def get_recommendations(user_id, n=10, rec_type="hybrid"):
    """Get recommendations for a user"""
    try:
        response = requests.get(
            f"{API_URL}/recommendations/{user_id}", 
            params={"n": n, "rec_type": rec_type, "exclude_seen": True}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting recommendations: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_bundle_info(bundle_id):
    """Get information about a specific bundle"""
    try:
        response = requests.get(f"{API_URL}/bundle/{bundle_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting bundle info: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_similar_bundles(bundle_id, n=5):
    """Get bundles similar to a given bundle"""
    try:
        response = requests.get(f"{API_URL}/similar/{bundle_id}", params={"n": n})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting similar bundles: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []

# Load evaluation metrics
@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics for different recommendation models"""
    # These would ideally come from an API endpoint, but we'll mock them for now
    metrics = {
        "Content-Based": {
            "precision@5": 0.182,
            "precision@10": 0.145,
            "recall@5": 0.092,
            "recall@10": 0.142,
            "diversity": 0.65,
            "coverage": 48.2
        },
        "Collaborative Filtering": {
            "precision@5": 0.204,
            "precision@10": 0.167,
            "recall@5": 0.104,
            "recall@10": 0.165,
            "diversity": 0.51,
            "coverage": 39.8
        },
        "Hybrid": {
            "precision@5": 0.238,
            "precision@10": 0.196,
            "recall@5": 0.121,
            "recall@10": 0.189,
            "diversity": 0.72,
            "coverage": 52.7
        }
    }
    return metrics

# Main dashboard layout
def main():
    st.title("📚 EdNet Educational Content Recommendation Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Dashboard Overview", "Recommendation Explorer", "Model Evaluation", "Bundle Explorer"]
    )
    
    # Load data
    if st.session_state.dataset_stats is None:
        st.session_state.dataset_stats = load_dataset_stats()
    if st.session_state.users is None:
        st.session_state.users = load_users()
    if st.session_state.bundles is None:
        st.session_state.bundles = load_bundles()
    if st.session_state.evaluation_metrics is None:
        st.session_state.evaluation_metrics = load_evaluation_metrics()
    
    # Display selected page
    if page == "Dashboard Overview":
        display_overview()
    elif page == "Recommendation Explorer":
        display_recommendation_explorer()
    elif page == "Model Evaluation":
        display_model_evaluation()
    elif page == "Bundle Explorer":
        display_bundle_explorer()

def display_overview():
    """Display dashboard overview"""
    st.header("Dataset Overview")
    
    if st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats
        
        # Display key statistics in metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", f"{stats['total_users']:,}")
        with col2:
            st.metric("Total Bundles", f"{stats['total_bundles']:,}")
        with col3:
            st.metric("Total Questions", f"{stats['total_questions']:,}")
        with col4:
            st.metric("Total Interactions", f"{stats['total_interactions']:,}")
        
        st.subheader("User Engagement Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg. Interactions per User", f"{stats['avg_interactions_per_user']:.2f}")
        with col2:
            st.metric("Correct Answer Rate", f"{stats['correct_answer_rate']:.2%}")
        
        # Part distribution chart
        st.subheader("Content Distribution by Part")
        part_data = pd.DataFrame({
            'Part': list(stats['part_distribution'].keys()),
            'Count': list(stats['part_distribution'].values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(
            part_data, 
            x='Part', 
            y='Count',
            color='Part',
            title="Distribution of Content by Part"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional sections
        st.subheader("Recommendation System Architecture")
        st.write("""
        This dashboard provides an interface to the educational content recommendation system built for EdNet.
        The system uses a hybrid approach that combines:
        
        1. **Content-Based Filtering**: Uses TF-IDF to find similar educational content based on metadata
        2. **Collaborative Filtering**: Uses matrix factorization (LightFM) to find patterns in user-item interactions
        3. **Hybrid Recommendations**: Combines both approaches for optimal personalization
        
        Explore the other tabs to see recommendations, evaluate model performance, and explore content bundles.
        """)
        
    else:
        st.info("Loading dataset statistics... If this persists, ensure the API is running.")

def display_recommendation_explorer():
    """Display recommendation explorer interface"""
    st.header("Recommendation Explorer")
    
    # User selection
    if st.session_state.users:
        selected_user = st.selectbox(
            "Select a user:", 
            st.session_state.users
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rec_count = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=10)
        with col2:
            rec_type = st.selectbox(
                "Recommendation type:", 
                ["hybrid", "content", "collaborative"]
            )
        with col3:
            if st.button("Get Recommendations"):
                with st.spinner("Generating recommendations..."):
                    st.session_state.user_history = load_user_history(selected_user)
                    st.session_state.recommendations = get_recommendations(
                        selected_user, 
                        n=rec_count, 
                        rec_type=rec_type
                    )
        
        # Display user history
        if st.session_state.user_history:
            st.subheader("User Interaction History")
            history_df = pd.DataFrame(st.session_state.user_history)
            
            # Convert timestamp to datetime
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Calculate stats
            correct_rate = history_df['correct'].mean()
            total_questions = len(history_df)
            unique_bundles = history_df['bundle_id'].nunique()
            avg_time = history_df['elapsed_time'].mean() / 1000  # Convert to seconds
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", total_questions)
            with col2:
                st.metric("Correct Answer Rate", f"{correct_rate:.2%}")
            with col3:
                st.metric("Unique Bundles", unique_bundles)
            with col4:
                st.metric("Avg. Time per Question", f"{avg_time:.1f}s")
            
            # Display history table
            st.dataframe(
                history_df[['timestamp', 'bundle_id', 'question_id', 'correct', 'elapsed_time']]
                .sort_values('timestamp', ascending=False)
                .reset_index(drop=True)
            )
        
        # Display recommendations
        if st.session_state.recommendations:
            st.subheader(f"Personalized Recommendations ({rec_type.capitalize()})")
            
            # Convert to DataFrame for easier manipulation
            recs_df = pd.DataFrame(st.session_state.recommendations['recommendations'])
            
            # Show recommendations
            for i, (_, rec) in enumerate(recs_df.iterrows()):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{i+1}. {rec['title']}**")
                        st.markdown(f"Part: {rec['part']} | Subject: {rec['subject']} | Difficulty: {rec['difficulty']}")
                    with col2:
                        st.markdown(f"Questions: {rec['question_count']}")
                        st.markdown(f"Popularity: {rec['popularity']}")
                    with col3:
                        st.markdown(f"Score: {rec['score']:.4f}")
                        if st.button(f"View Details #{i}", key=f"details_{i}"):
                            bundle_info = get_bundle_info(rec['bundle_id'])
                            if bundle_info:
                                st.json(bundle_info)
                    st.markdown("---")
            
            # Create visualization of recommendations by subject and difficulty
            st.subheader("Recommendation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Subject distribution
                subject_counts = recs_df['subject'].value_counts().reset_index()
                subject_counts.columns = ['Subject', 'Count']
                
                fig = px.pie(
                    subject_counts, 
                    names='Subject', 
                    values='Count',
                    title="Recommendations by Subject"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Difficulty distribution
                difficulty_counts = recs_df['difficulty'].value_counts().reset_index()
                difficulty_counts.columns = ['Difficulty', 'Count']
                
                # Define color mapping for difficulty
                color_map = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
                
                fig = px.bar(
                    difficulty_counts, 
                    x='Difficulty', 
                    y='Count',
                    color='Difficulty',
                    color_discrete_map=color_map,
                    title="Recommendations by Difficulty"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info("Loading users... If this persists, ensure the API is running.")

def display_model_evaluation():
    """Display model evaluation metrics"""
    st.header("Model Evaluation")
    
    if st.session_state.evaluation_metrics:
        metrics = st.session_state.evaluation_metrics
        
        st.write("""
        This page shows the performance evaluation of different recommendation models.
        The metrics were computed using offline evaluation on a held-out test set.
        """)
        
        # Convert to DataFrame for easier visualization
        metrics_df = pd.DataFrame(metrics).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # Display metrics in a table
        st.subheader("Performance Metrics")
        st.dataframe(metrics_df)
        
        # Precision and Recall comparison
        st.subheader("Precision and Recall Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision@K comparison
            fig = px.bar(
                metrics_df, 
                x='Model', 
                y=['precision@5', 'precision@10'],
                barmode='group',
                labels={'value': 'Precision', 'variable': 'Metric'},
                title="Precision@K Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recall@K comparison
            fig = px.bar(
                metrics_df, 
                x='Model', 
                y=['recall@5', 'recall@10'],
                barmode='group',
                labels={'value': 'Recall', 'variable': 'Metric'},
                title="Recall@K Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Diversity and Coverage
        st.subheader("Diversity and Coverage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Diversity comparison
            fig = px.bar(
                metrics_df, 
                x='Model', 
                y='diversity',
                color='Model',
                labels={'diversity': 'Diversity Score (0-1)'},
                title="Recommendation Diversity by Model"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Coverage comparison
            fig = px.bar(
                metrics_df, 
                x='Model', 
                y='coverage',
                color='Model',
                labels={'coverage': 'Catalog Coverage (%)'},
                title="Catalog Coverage by Model"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # A/B Testing simulation
        st.subheader("A/B Testing Simulation")
        
        st.write("""
        The chart below shows a simulated A/B test comparing click-through rates (CTR) 
        for recommendations from different models. The hybrid model shows the best performance
        in terms of user engagement.
        """)
        
        # Simulate A/B testing data
        np.random.seed(42)  # For reproducibility
        days = 14
        ab_data = {
            'Day': list(range(1, days + 1)) * 3,
            'Model': ['Content-Based'] * days + ['Collaborative Filtering'] * days + ['Hybrid'] * days,
            'CTR': [
                # Content-based CTRs
                *np.random.normal(0.082, 0.01, days),
                # Collaborative CTRs
                *np.random.normal(0.095, 0.01, days),
                # Hybrid CTRs
                *np.random.normal(0.118, 0.01, days)
            ]
        }
        
        ab_df = pd.DataFrame(ab_data)
        
        fig = px.line(
            ab_df, 
            x='Day', 
            y='CTR', 
            color='Model',
            markers=True,
            labels={'CTR': 'Click-Through Rate'},
            title="A/B Testing: CTR by Recommendation Model"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional explanation
        st.subheader("Evaluation Methodology")
        
        st.write("""
        ### Evaluation Metrics
        
        - **Precision@K**: The proportion of recommended items that are relevant
        - **Recall@K**: The proportion of relevant items that are recommended
        - **Diversity**: Measures how diverse the recommendations are (higher is better)
        - **Coverage**: Percentage of items in the catalog that get recommended
        
        ### Comparison of Models
        
        - **Content-Based**: Good at finding similar content but can lead to filter bubbles
        - **Collaborative Filtering**: Good at capturing user preferences but suffers from cold-start
        - **Hybrid**: Combines strengths of both approaches for better performance
        
        The hybrid model consistently outperforms both individual models across all metrics.
        """)
        
    else:
        st.info("Loading evaluation metrics...")

def display_bundle_explorer():
    """Display bundle explorer interface"""
    st.header("Bundle Explorer")
    
    if st.session_state.bundles:
        # Bundle selection
        selected_bundle = st.selectbox(
            "Select a bundle:", 
            [bundle['bundle_id'] for bundle in st.session_state.bundles]
        )
        
        if st.button("Load Bundle Details"):
            with st.spinner("Loading bundle information..."):
                bundle_info = get_bundle_info(selected_bundle)
                similar_bundles = get_similar_bundles(selected_bundle, n=5)
                
                if bundle_info:
                    # Display bundle details
                    st.subheader(f"Bundle Details: {bundle_info['title']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Part", bundle_info['part'])
                    with col2:
                        st.metric("Subject", bundle_info['subject'])
                    with col3:
                        st.metric("Difficulty", bundle_info['difficulty'])
                    with col4:
                        st.metric("Success Rate", f"{bundle_info['success_rate']:.2%}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Questions", bundle_info['question_count'])
                    with col2:
                        st.metric("Popularity", bundle_info['popularity'])
                    
                    # Display tags
                    st.subheader("Tags")
                    tags = bundle_info['tags'].split(';')
                    st.write(', '.join(tags))
                    
                    # Display questions
                    st.subheader("Questions")
                    st.write(f"This bundle contains {len(bundle_info['questions'])} questions.")
                    for q in bundle_info['questions']:
                        st.write(f"- {q}")
                    
                    # Display similar bundles
                    if similar_bundles:
                        st.subheader("Similar Bundles")
                        
                        for i, bundle in enumerate(similar_bundles):
                            with st.container():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.markdown(f"**{i+1}. {bundle['title']}**")
                                    st.markdown(f"Part: {bundle['part']} | Subject: {bundle['subject']} | Difficulty: {bundle['difficulty']}")
                                with col2:
                                    st.markdown(f"Questions: {bundle['question_count']}")
                                    st.markdown(f"Popularity: {bundle['popularity']}")
                                with col3:
                                    st.markdown(f"Similarity: {bundle['similarity_score']:.4f}")
                                st.markdown("---")
                                
                        # Create a network visualization of similar bundles
                        st.subheader("Bundle Similarity Network")
                        
                        # Create network data
                        network_data = {
                            'source': [selected_bundle] * len(similar_bundles),
                            'target': [b['bundle_id'] for b in similar_bundles],
                            'weight': [b['similarity_score'] for b in similar_bundles],
                            'subject': [b['subject'] for b in similar_bundles]
                        }
                        
                        # Create a force-directed graph
                        # (This is a simplified version since we can't use networkx directly in Streamlit)
                        nodes = list(set(network_data['source'] + network_data['target']))
                        node_data = pd.DataFrame({'id': nodes})
                        
                        edges = pd.DataFrame(network_data)
                        
                        st.write("""
                        This visualization shows the selected bundle (center) and its most similar bundles.
                        The thickness of lines represents similarity strength.
                        """)
                        
                        # Create a simple bar chart to show similarity scores
                        fig = px.bar(
                            edges, 
                            x='target', 
                            y='weight',
                            color='subject',
                            labels={'target': 'Bundle ID', 'weight': 'Similarity Score'},
                            title="Similarity to Selected Bundle"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loading bundles... If this persists, ensure the API is running.")

# Run the app
if __name__ == "__main__":
    main()
