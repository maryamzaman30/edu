"""
Transformer-based Recommendation Model Implementation

This module implements an actual transformer-based model for educational content recommendations
using PyTorch, rather than simulating the behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TransformerRecommender(nn.Module):
    """
    Transformer-based model for educational content recommendation.
    
    This model uses a transformer encoder architecture to process sequences of user interactions
    and predict the next most relevant content items.
    """
    
    def __init__(self, n_items, n_factors=64, n_heads=4, n_layers=2, dropout=0.1):
        """
        Initialize the transformer-based recommendation model.
        
        Args:
            n_items: Number of unique items in the dataset
            n_factors: Dimension of the embedding space
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerRecommender, self).__init__()
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(n_items + 1, n_factors, padding_idx=0)
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(n_factors, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_factors,
            nhead=n_heads,
            dim_feedforward=n_factors * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(n_factors, n_items + 1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights of the model."""
        initrange = 0.1
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_key_padding_mask=None):
        """
        Forward pass of the model.
        
        Args:
            src: Input sequence of item indices [batch_size, seq_len]
            src_key_padding_mask: Boolean mask for padding tokens
            
        Returns:
            Output predictions for next items
        """
        # Get item embeddings [batch_size, seq_len, n_factors]
        src = self.item_embeddings(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoder [batch_size, seq_len, n_factors]
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Get the representation of the last item in the sequence
        last_item_representation = output[:, -1, :]
        
        # Output layer [batch_size, n_items]
        output = self.output_layer(last_item_representation)
        
        return output
    
    def get_recommendations(self, user_seq, k=10, exclude_seen=True):
        """
        Get top-k recommendations for a user.
        
        Args:
            user_seq: Sequence of item indices for a user
            k: Number of recommendations to return
            exclude_seen: Whether to exclude items that the user has already interacted with
            
        Returns:
            List of top-k recommended item indices
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            seq_tensor = torch.LongTensor(user_seq).unsqueeze(0)
            
            # Forward pass
            logits = self.forward(seq_tensor)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Set probabilities of seen items to 0 if exclude_seen is True
            if exclude_seen and len(user_seq) > 0:
                probs[0, user_seq] = 0
                
            # Get top-k items
            top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
            
        return top_k_indices.squeeze(0).tolist(), top_k_probs.squeeze(0).tolist()
        
class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    
    This module adds positional information to the input embeddings to give the model
    information about the order of items in the sequence.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding module.
        
        Args:
            d_model: Dimension of the embedding space
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Forward pass of the positional encoding module.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def prepare_data_for_transformer(merged_data):
    """
    Prepare data for the transformer model from the EdNet dataset.
    
    Args:
        merged_data: DataFrame with user interactions
        
    Returns:
        user_item_sequences: Dictionary mapping user IDs to sequences of item indices
        item_id_map: Dictionary mapping original item IDs to indices
        id_item_map: Dictionary mapping indices to original item IDs
    """
    # Sort by user and timestamp
    sorted_data = merged_data.sort_values(['user_id', 'timestamp'])
    
    # Create item ID mapping
    unique_items = sorted_data['bundle_id'].unique()
    item_id_map = {item: idx+1 for idx, item in enumerate(unique_items)}  # Start from 1 (0 is padding)
    id_item_map = {idx+1: item for idx, item in enumerate(unique_items)}
    
    # Create user-item sequences
    user_item_sequences = {}
    for user_id, user_df in sorted_data.groupby('user_id'):
        user_item_sequences[user_id] = [item_id_map[item] for item in user_df['bundle_id']]
    
    return user_item_sequences, item_id_map, id_item_map

def get_transformer_recommendations(model, user_id, user_item_sequences, id_item_map, k=10):
    """
    Get recommendations for a user using the transformer model.
    
    Args:
        model: Trained transformer model
        user_id: User ID to get recommendations for
        user_item_sequences: Dictionary mapping user IDs to sequences of item indices
        id_item_map: Dictionary mapping indices to original item IDs
        k: Number of recommendations to return
        
    Returns:
        List of recommended item IDs and their scores
    """
    # Get user sequence
    if user_id not in user_item_sequences or len(user_item_sequences[user_id]) == 0:
        # Cold start: return most popular items
        return [], []
    
    user_seq = user_item_sequences[user_id]
    
    # Get recommendations
    item_indices, scores = model.get_recommendations(user_seq, k=k)
    
    # Convert indices back to original item IDs
    recommended_items = [id_item_map.get(idx, None) for idx in item_indices]
    
    # Filter out None values (in case the model predicts indices not in the mapping)
    recommendations = [(item, score) for item, score in zip(recommended_items, scores) if item is not None]
    
    return recommendations