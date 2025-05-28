"""
FastAPI Backend for Educational Content Recommendation System

This module implements the API endpoints for the educational content 
recommendation system, allowing clients to get personalized recommendations.
"""

import os
import sys
import logging
import pickle
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import recommendation modules
from src.utils.data_processor import EdNetDataProcessor
from src.recommender.content_based import ContentBasedRecommender
from src.recommender.collaborative import CollaborativeFilteringRecommender
from src.recommender.hybrid import HybridRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Educational Content Recommendation API",
    description="API for personalized educational content recommendations based on user interactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Data paths
MERGED_DATA_PATH = "data/cleaned/merged_cleaned_data.csv"
LECTURES_DATA_PATH = "data/cleaned/cleaned_lectures.csv"
MODELS_DIR = "models"

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    n_recommendations: int = Field(10, description="Number of recommendations to return")
    recommendation_type: str = Field("hybrid", description="Type of recommendation (content, collaborative, hybrid)")
    exclude_seen: bool = Field(True, description="Whether to exclude items the user has already interacted with")

class RecommendationItem(BaseModel):
    bundle_id: str = Field(..., description="Bundle ID")
    title: str = Field(..., description="Bundle title")
    part: str = Field(..., description="Part/section name")
    subject: str = Field(..., description="Subject category")
    difficulty: str = Field(..., description="Difficulty level (Easy, Medium, Hard)")
    question_count: int = Field(..., description="Number of questions in the bundle")
    popularity: int = Field(..., description="Interaction count as a measure of popularity")
    score: float = Field(..., description="Recommendation score")

class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    recommendation_type: str = Field(..., description="Type of recommendation used")
    timestamp: str = Field(..., description="Timestamp of recommendation")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")

# Global variables for loaded models and data
data_processor = None
content_recommender = None
collab_recommender = None
hybrid_recommender = None
merged_data = None
lectures_data = None
bundle_features = None
user_item_matrix = None

@app.on_event("startup")
async def startup_event():
    """Load data and models on startup"""
    global data_processor, content_recommender, collab_recommender, hybrid_recommender
    global merged_data, lectures_data, bundle_features, user_item_matrix
    
    logger.info("Loading data and models...")
    
    try:
        # Initialize data processor
        data_processor = EdNetDataProcessor(MERGED_DATA_PATH, LECTURES_DATA_PATH)
        
        # Load data
        merged_data, lectures_data = data_processor.load_data()
        data_processor.clean_and_preprocess()
        
        # Create user-item matrix
        user_item_matrix = data_processor.create_user_item_matrix()
        
        # Create item features
        bundle_features = data_processor.create_item_features()
        
        # Load individual models
        logger.info("Loading individual pre-trained models...")
        
        # Load content-based model
        content_model_path = os.path.join(MODELS_DIR, "content_based_model.pkl")
        if os.path.exists(content_model_path):
            with open(content_model_path, 'rb') as f:
                content_model_data = pickle.load(f)
            
            content_recommender = ContentBasedRecommender(bundle_features)
            content_recommender.tfidf_vectorizer = content_model_data.get('tfidf_vectorizer')
            content_recommender.tfidf_matrix = content_model_data.get('tfidf_matrix')
            # content_recommender.similarity_matrix = content_model_data.get('similarity_matrix')
            content_recommender.similarity_matrix = content_model_data.get('cosine_sim')
            content_recommender.bundle_to_index = {b: i for i, b in enumerate(bundle_features['bundle_id'])}
            content_recommender.index_to_bundle = {i: b for i, b in enumerate(bundle_features['bundle_id'])}
        else:
            logger.warning("Content-based model not found. Training new model...")
            content_recommender = ContentBasedRecommender(bundle_features)
            content_recommender.preprocess_content()
            content_recommender.fit()
            
        # Load SVD model
        svd_model_path = os.path.join(MODELS_DIR, "collaborative_model.joblib")
        if os.path.exists(svd_model_path):
            try:
                svd_model_data = joblib.load(svd_model_path)
                    
                # Try different ways to get the model components
                if isinstance(svd_model_data, dict):
                    # Try to load SVD components
                    if 'svd' in svd_model_data and 'user_factors' in svd_model_data and 'item_factors' in svd_model_data:
                        collab_recommender = CollaborativeFilteringRecommender(user_item_matrix, bundle_features)
                        collab_recommender.svd = svd_model_data['svd']
                        collab_recommender.user_factors = svd_model_data['user_factors']
                        collab_recommender.item_factors = svd_model_data['item_factors']
                        logger.info("Successfully loaded SVD model components")
                        return
                    
                logger.warning("Model format not recognized. Training new model...")
                collab_recommender = CollaborativeFilteringRecommender(user_item_matrix, bundle_features)
                collab_recommender.fit()
            except Exception as e:
                logger.warning(f"Error loading model: {str(e)}. Training new model...")
                collab_recommender = CollaborativeFilteringRecommender(user_item_matrix, bundle_features)
                collab_recommender.fit()
        else:
            logger.warning("SVD model not found. Training new model...")
            collab_recommender = CollaborativeFilteringRecommender(user_item_matrix, bundle_features)
            collab_recommender.fit()
            
        # Initialize hybrid recommender
        hybrid_recommender = HybridRecommender(
            content_recommender,
            collab_recommender,
            bundle_features
        )
        hybrid_recommender.set_weights(0.5)  # Default alpha value
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Educational Content Recommendation API",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "content_recommender": content_recommender is not None,
            "collab_recommender": collab_recommender is not None,
            "hybrid_recommender": hybrid_recommender is not None
        },
        "data_loaded": {
            "merged_data": merged_data is not None,
            "lectures_data": lectures_data is not None,
            "user_item_matrix": user_item_matrix is not None,
            "bundle_features": bundle_features is not None
        }
    }

@app.get("/users", response_model=List[str])
async def get_users(limit: int = Query(20, description="Maximum number of users to return")):
    """Get a list of user IDs"""
    if merged_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    users = merged_data['user_id'].unique().tolist()[:limit]
    return users

@app.get("/bundles", response_model=List[Dict])
async def get_bundles(limit: int = Query(20, description="Maximum number of bundles to return")):
    """Get a list of bundle information"""
    if bundle_features is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    bundles = []
    for _, row in bundle_features.head(limit).iterrows():
        # Calculate difficulty level
        success_rate = row['success_rate']
        if success_rate < 0.3:
            difficulty = "Hard"
        elif success_rate < 0.7:
            difficulty = "Medium"
        else:
            difficulty = "Easy"
            
        bundles.append({
            'bundle_id': row['bundle_id'],
            'title': f"Bundle {row['bundle_id']}",
            'part': row['part_name'],
            'subject': row['subject_category'],
            'difficulty': difficulty,
            'question_count': int(row['question_count']),
            'popularity': int(row['interaction_count'])
        })
    
    return bundles

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str = Path(..., description="User ID to get recommendations for"),
    n: int = Query(10, description="Number of recommendations to return"),
    rec_type: str = Query("hybrid", description="Type of recommendation (content, collaborative, hybrid)"),
    exclude_seen: bool = Query(True, description="Whether to exclude items the user has already interacted with")
):
    """Get recommendations for a user"""
    if merged_data is None or hybrid_recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Check if user exists
    if user_id not in merged_data['user_id'].unique():
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    logger.info(f"Getting {rec_type} recommendations for user {user_id}, n={n}")
    
    try:
        # Get user history
        user_history = merged_data[merged_data['user_id'] == user_id]
        
        recommendations = []
        
        if rec_type == "content":
            # Get content-based recommendations
            content_recs = content_recommender.recommend_for_user(user_history, n=n, exclude_seen=exclude_seen)
            logger.info(f"Content-based recommender returned {len(content_recs)} recommendations")
            
            recommendations = []
            for rec in content_recs:
                bundle_id = rec['bundle_id']
                bundle_info = bundle_features[bundle_features['bundle_id'] == bundle_id].iloc[0]
                
                # Calculate difficulty
                success_rate = bundle_info['success_rate']
                if success_rate < 0.3:
                    difficulty = "Hard"
                elif success_rate < 0.7:
                    difficulty = "Medium"
                else:
                    difficulty = "Easy"
                
                recommendations.append({
                    'bundle_id': bundle_id,
                    'title': f"Bundle {bundle_id}",
                    'part': bundle_info['part_name'],
                    'subject': bundle_info['subject_category'],
                    'difficulty': difficulty,
                    'question_count': int(bundle_info['question_count']),
                    'popularity': int(bundle_info['interaction_count']),
                    'score': float(rec['content_score'])
                })
                
        elif rec_type == "collaborative":
            # Get collaborative recommendations
            collab_recs = collab_recommender.recommend_for_user(user_id, n=n, exclude_seen=exclude_seen)
            logger.info(f"Collaborative recommender returned {len(collab_recs)} recommendations")
            
            recommendations = []
            for rec in collab_recs:
                bundle_id = rec['bundle_id']
                bundle_info = bundle_features[bundle_features['bundle_id'] == bundle_id].iloc[0]
                
                # Calculate difficulty
                success_rate = bundle_info['success_rate']
                if success_rate < 0.3:
                    difficulty = "Hard"
                elif success_rate < 0.7:
                    difficulty = "Medium"
                else:
                    difficulty = "Easy"
                
                recommendations.append({
                    'bundle_id': bundle_id,
                    'title': f"Bundle {bundle_id}",
                    'part': bundle_info['part_name'],
                    'subject': bundle_info['subject_category'],
                    'difficulty': difficulty,
                    'question_count': int(bundle_info['question_count']),
                    'popularity': int(bundle_info['interaction_count']),
                    'score': float(rec['collab_score'])
                })
                
        else:  # hybrid
            # Get hybrid recommendations
            hybrid_recs = hybrid_recommender.recommend_for_user(
                user_id, 
                user_history=user_history, 
                n=n, 
                exclude_seen=exclude_seen
            )
            logger.info(f"Hybrid recommender returned {len(hybrid_recs)} recommendations")
            
            recommendations = []
            for rec in hybrid_recs:
                recommendations.append({
                    'bundle_id': rec['bundle_id'],
                    'title': rec['title'],
                    'part': rec['part'],
                    'subject': rec['subject'],
                    'difficulty': rec['difficulty'],
                    'question_count': int(rec.get('question_count', 0)),
                    'popularity': int(rec.get('popularity', 0)),
                    'score': float(rec['hybrid_score'])
                })
        
        # Create response
        response = {
            'user_id': user_id,
            'recommendation_type': rec_type,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/user/{user_id}/history", response_model=List[Dict])
async def get_user_history(
    user_id: str = Path(..., description="User ID to get history for"),
    limit: int = Query(20, description="Maximum number of interactions to return")
):
    """Get interaction history for a user"""
    if merged_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Check if user exists
    if user_id not in merged_data['user_id'].unique():
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Get user history
    user_history = merged_data[merged_data['user_id'] == user_id].sort_values('timestamp', ascending=False).head(limit)
    
    history_items = []
    for _, row in user_history.iterrows():
        correct = row['user_answer'] == row['correct_answer']
        
        history_items.append({
            'timestamp': row['timestamp'].isoformat(),
            'bundle_id': row['bundle_id'],
            'question_id': row['question_id'],
            'part': int(row['part']),
            'elapsed_time': int(row['elapsed_time']),
            'correct': correct,
            'user_answer': row['user_answer'],
            'correct_answer': row['correct_answer']
        })
    
    return history_items

@app.get("/bundle/{bundle_id}", response_model=Dict)
async def get_bundle_info(
    bundle_id: str = Path(..., description="Bundle ID to get information for")
):
    """Get information about a specific bundle"""
    if bundle_features is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Check if bundle exists
    bundle_rows = bundle_features[bundle_features['bundle_id'] == bundle_id]
    if len(bundle_rows) == 0:
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found")
    
    bundle_row = bundle_rows.iloc[0]
    
    # Calculate difficulty
    success_rate = bundle_row['success_rate']
    if success_rate < 0.3:
        difficulty = "Hard"
    elif success_rate < 0.7:
        difficulty = "Medium"
    else:
        difficulty = "Easy"
    
    # Get questions for this bundle
    questions = merged_data[merged_data['bundle_id'] == bundle_id]['question_id'].unique().tolist()
    
    bundle_info = {
        'bundle_id': bundle_id,
        'title': f"Bundle {bundle_id}",
        'part': bundle_row['part_name'],
        'part_number': int(bundle_row['part']),
        'subject': bundle_row['subject_category'],
        'tags': bundle_row['tags'],
        'difficulty': difficulty,
        'success_rate': float(bundle_row['success_rate']),
        'question_count': int(bundle_row['question_count']),
        'popularity': int(bundle_row['interaction_count']),
        'questions': questions[:20]  # Limit to 20 questions
    }
    
    return bundle_info

@app.get("/similar/{bundle_id}", response_model=List[Dict])
async def get_similar_bundles(
    bundle_id: str = Path(..., description="Bundle ID to find similar bundles for"),
    n: int = Query(10, description="Number of similar bundles to return")
):
    """Get bundles similar to a given bundle"""
    if content_recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get similar bundles
        similar_bundles = content_recommender.recommend_similar_items(bundle_id, n=n)
        
        # Add more details to recommendations
        recommendations = []
        for rec in similar_bundles:
            bundle_id = rec['bundle_id']
            bundle_info = bundle_features[bundle_features['bundle_id'] == bundle_id].iloc[0]
            
            # Calculate difficulty
            success_rate = bundle_info['success_rate']
            if success_rate < 0.3:
                difficulty = "Hard"
            elif success_rate < 0.7:
                difficulty = "Medium"
            else:
                difficulty = "Easy"
            
            recommendations.append({
                'bundle_id': bundle_id,
                'title': f"Bundle {bundle_id}",
                'part': bundle_info['part_name'],
                'subject': bundle_info['subject_category'],
                'difficulty': difficulty,
                'question_count': int(bundle_info['question_count']),
                'popularity': int(bundle_info['interaction_count']),
                'similarity_score': float(rec['similarity_score'])
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error finding similar bundles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding similar bundles: {str(e)}")

@app.get("/stats", response_model=Dict)
async def get_dataset_stats():
    """Get statistics about the dataset"""
    if merged_data is None or lectures_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        stats = {
            'total_users': merged_data['user_id'].nunique(),
            'total_bundles': merged_data['bundle_id'].nunique(),
            'total_questions': merged_data['question_id'].nunique(),
            'total_interactions': len(merged_data),
            'total_lectures': len(lectures_data),
            'avg_interactions_per_user': len(merged_data) / merged_data['user_id'].nunique(),
            'avg_questions_per_bundle': merged_data.groupby('bundle_id')['question_id'].nunique().mean(),
            'correct_answer_rate': (merged_data['user_answer'] == merged_data['correct_answer']).mean(),
            'part_distribution': {
                part_name: int(count) for part_name, count in 
                merged_data['part'].map(data_processor.part_names).value_counts().items()
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating dataset stats: {str(e)}")

# Run the app
if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
