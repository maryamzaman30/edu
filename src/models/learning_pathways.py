"""
Interactive Learning Pathways Module

This module implements interactive learning pathways based on the TOEIC test structure,
allowing for personalized learning journeys tailored to individual student needs.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

class LearningPathway:
    """
    A class to create interactive learning pathways for educational content.
    
    This class uses the TOEIC test structure to generate personalized learning paths
    based on user performance, learning goals, and content relationships.
    """
    
    def __init__(self, bundle_data, interaction_data=None):
        """
        Initialize the learning pathway generator.
        
        Args:
            bundle_data: DataFrame with content bundle information
            interaction_data: Optional DataFrame with user-bundle interactions
        """
        self.bundle_data = bundle_data
        self.interaction_data = interaction_data
        
        # TOEIC test structure
        self.toeic_structure = {
            # Listening Section
            1: {
                "name": "Photographs", 
                "section": "Listening",
                "description": "Look at photographs and listen to four statements. Select the statement that best describes the photograph.",
                "skills": ["Visual recognition", "Listening comprehension"],
                "difficulty": "Easy",
                "time_allocation": 15  # minutes
            },
            2: {
                "name": "Question-Response", 
                "section": "Listening",
                "description": "Listen to a question and three possible responses. Select the most appropriate response.",
                "skills": ["Listening comprehension", "Quick response"],
                "difficulty": "Medium",
                "time_allocation": 25
            },
            3: {
                "name": "Conversations", 
                "section": "Listening",
                "description": "Listen to conversations and answer questions about their content.",
                "skills": ["Listening comprehension", "Context understanding"],
                "difficulty": "Medium",
                "time_allocation": 30
            },
            4: {
                "name": "Talks", 
                "section": "Listening",
                "description": "Listen to talks such as announcements and answer questions about their content.",
                "skills": ["Listening comprehension", "Note-taking"],
                "difficulty": "Hard",
                "time_allocation": 30
            },
            # Reading Section
            5: {
                "name": "Incomplete Sentences", 
                "section": "Reading",
                "description": "Read sentences with missing words and select the best word or phrase to complete them.",
                "skills": ["Grammar", "Vocabulary"],
                "difficulty": "Medium",
                "time_allocation": 20
            },
            6: {
                "name": "Text Completion", 
                "section": "Reading",
                "description": "Read passages with missing words and select the best word or phrase to complete them.",
                "skills": ["Reading comprehension", "Grammar", "Vocabulary"],
                "difficulty": "Medium",
                "time_allocation": 20
            },
            7: {
                "name": "Reading Comprehension", 
                "section": "Reading",
                "description": "Read passages and answer questions about them.",
                "skills": ["Reading comprehension", "Critical thinking"],
                "difficulty": "Hard",
                "time_allocation": 55
            }
        }
        
        # Prerequisites graph (part dependencies)
        self.prerequisites = {
            1: [],
            2: [1],
            3: [1, 2],
            4: [2, 3],
            5: [],
            6: [5],
            7: [5, 6]
        }
        
        # Build the content graph
        self.build_content_graph()
    
    def build_content_graph(self):
        """
        Build a graph representing content relationships.
        
        This creates a directed graph where nodes are content bundles and
        edges represent prerequisites or recommended next steps.
        """
        # Create a directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes (content bundles)
        for _, bundle in self.bundle_data.iterrows():
            # Extract bundle information
            bundle_id = bundle['bundle_id']
            part = bundle.get('part', 0)
            
            # Get TOEIC part information
            toeic_info = self.toeic_structure.get(part, {
                "name": f"Part {part}",
                "section": "General",
                "description": "General educational content",
                "skills": ["General"],
                "difficulty": "Medium",
                "time_allocation": 20
            })
            
            # Create node attributes
            node_attrs = {
                'bundle_id': bundle_id,
                'part': part,
                'part_name': toeic_info['name'],
                'section': toeic_info['section'],
                'subject': bundle.get('subject_category', 'General'),
                'difficulty': bundle.get('difficulty', toeic_info['difficulty']),
                'success_rate': bundle.get('success_rate', 0.5),
                'question_count': bundle.get('question_count', 10),
                'time_allocation': toeic_info['time_allocation'],
                'skills': toeic_info['skills'],
                'description': toeic_info['description']
            }
            
            # Add node to graph
            self.graph.add_node(bundle_id, **node_attrs)
        
        # Add edges based on part prerequisites and content similarity
        for bundle_id, data in self.graph.nodes(data=True):
            part = data['part']
            
            # Find prerequisite parts
            prereq_parts = self.prerequisites.get(part, [])
            
            # Find bundles in prerequisite parts
            for prereq_part in prereq_parts:
                prereq_bundles = [b for b, d in self.graph.nodes(data=True) 
                                if d['part'] == prereq_part]
                
                # Add edges to all prerequisite bundles
                for prereq_bundle in prereq_bundles:
                    # Calculate edge weight based on success rate
                    # Higher success rate means stronger recommendation
                    prereq_success = self.graph.nodes[prereq_bundle]['success_rate']
                    edge_weight = prereq_success if prereq_success > 0 else 0.1
                    
                    self.graph.add_edge(prereq_bundle, bundle_id, 
                                       weight=edge_weight,
                                       relation_type='prerequisite')
            
            # Add edges based on content similarity (same subject)
            same_subject_bundles = [b for b, d in self.graph.nodes(data=True)
                                    if d['subject'] == data['subject'] and b != bundle_id]
            
            for sim_bundle in same_subject_bundles:
                # Only connect to bundles with similar or higher difficulty
                sim_difficulty = self.graph.nodes[sim_bundle]['difficulty']
                current_difficulty = data['difficulty']
                
                difficulty_levels = ['Easy', 'Medium', 'Hard']
                current_idx = difficulty_levels.index(current_difficulty) if current_difficulty in difficulty_levels else 1
                sim_idx = difficulty_levels.index(sim_difficulty) if sim_difficulty in difficulty_levels else 1
                
                # Add edge if similar or higher difficulty
                if sim_idx >= current_idx:
                    # Calculate similarity based on attributes
                    edge_weight = 0.5  # Base similarity
                    
                    # Add edge
                    self.graph.add_edge(bundle_id, sim_bundle,
                                       weight=edge_weight,
                                       relation_type='similarity')
    
    def get_user_starting_point(self, user_id=None, user_history=None, assessment_results=None):
        """
        Determine the best starting point for a user.
        
        Args:
            user_id: User ID to get starting point for
            user_history: Optional DataFrame with user history
            assessment_results: Optional dictionary with assessment results
            
        Returns:
            Dictionary with recommended starting bundles
        """
        # If assessment results are provided, use them
        if assessment_results is not None:
            return self._get_starting_point_from_assessment(assessment_results)
        
        # If user history is provided, use it
        if user_history is not None and not user_history.empty:
            return self._get_starting_point_from_history(user_history)
        
        # If interaction data is available and user_id is provided, get user history
        if self.interaction_data is not None and user_id is not None:
            user_history = self.interaction_data[self.interaction_data['user_id'] == user_id]
            if not user_history.empty:
                return self._get_starting_point_from_history(user_history)
        
        # Default: start with easiest content from each section
        return self._get_default_starting_point()
    
    def _get_starting_point_from_assessment(self, assessment_results):
        """
        Determine starting point based on assessment results.
        
        Args:
            assessment_results: Dictionary with assessment scores by skill
            
        Returns:
            Dictionary with recommended starting bundles
        """
        # Extract scores from assessment
        listening_score = assessment_results.get('listening', 0.5)
        reading_score = assessment_results.get('reading', 0.5)
        grammar_score = assessment_results.get('grammar', 0.5)
        vocabulary_score = assessment_results.get('vocabulary', 0.5)
        
        # Define difficulty thresholds
        low_threshold = 0.3
        high_threshold = 0.7
        
        # Determine appropriate part for each section based on scores
        recommended_parts = {}
        
        # Listening section (Parts 1-4)
        if listening_score < low_threshold:
            # Low score: start with easiest
            recommended_parts['Listening'] = 1
        elif listening_score < high_threshold:
            # Medium score: start with medium difficulty
            recommended_parts['Listening'] = 2
        else:
            # High score: start with harder content
            recommended_parts['Listening'] = 3
        
        # Reading section (Parts 5-7)
        if reading_score < low_threshold:
            # Low score: start with easier reading content
            recommended_parts['Reading'] = 5
        elif reading_score < high_threshold:
            # Medium score: start with medium difficulty
            recommended_parts['Reading'] = 6
        else:
            # High score: start with harder content
            recommended_parts['Reading'] = 7
        
        # Find bundles for recommended parts
        recommendations = {}
        for section, part in recommended_parts.items():
            # Get bundles for this part
            part_bundles = [b for b, d in self.graph.nodes(data=True) 
                           if d['part'] == part]
            
            # Sort by success rate (higher is easier)
            part_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                             reverse=True)
            
            # Take top 3 bundles
            recommendations[section] = part_bundles[:3]
        
        return recommendations
    
    def _get_starting_point_from_history(self, user_history):
        """
        Determine starting point based on user history.
        
        Args:
            user_history: DataFrame with user's interaction history
            
        Returns:
            Dictionary with recommended starting bundles
        """
        # Calculate user performance by part
        part_performance = {}
        for part in range(1, 8):
            # Filter history for this part
            part_history = user_history[user_history['part'] == part]
            
            if not part_history.empty:
                # Calculate success rate
                if 'correct' in part_history.columns:
                    success_rate = part_history['correct'].mean()
                elif 'user_answer' in part_history.columns and 'correct_answer' in part_history.columns:
                    success_rate = (part_history['user_answer'] == part_history['correct_answer']).mean()
                else:
                    success_rate = 0.5  # Default
                
                part_performance[part] = {
                    'success_rate': success_rate,
                    'count': len(part_history)
                }
        
        # Determine next steps for each section
        recommendations = {'Listening': [], 'Reading': []}
        
        # Check if user has started each section
        listening_parts = [p for p in range(1, 5) if p in part_performance]
        reading_parts = [p for p in range(5, 8) if p in part_performance]
        
        # For Listening section
        if listening_parts:
            # Find highest part with good performance
            good_parts = [p for p in listening_parts 
                         if part_performance[p]['success_rate'] >= 0.7]
            
            if good_parts:
                # User is doing well, recommend next part
                current_part = max(good_parts)
                next_part = min(current_part + 1, 4)
            else:
                # User needs more practice, recommend same part
                current_part = max(listening_parts)
                next_part = current_part
            
            # Get bundles for recommended part
            part_bundles = [b for b, d in self.graph.nodes(data=True) 
                           if d['part'] == next_part]
            
            # Sort by success rate
            part_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                             reverse=True)
            
            recommendations['Listening'] = part_bundles[:3]
        else:
            # User hasn't started listening section, recommend part 1
            part_bundles = [b for b, d in self.graph.nodes(data=True) 
                           if d['part'] == 1]
            
            part_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                             reverse=True)
            
            recommendations['Listening'] = part_bundles[:3]
        
        # Similar logic for Reading section
        if reading_parts:
            good_parts = [p for p in reading_parts 
                         if part_performance[p]['success_rate'] >= 0.7]
            
            if good_parts:
                current_part = max(good_parts)
                next_part = min(current_part + 1, 7)
            else:
                current_part = max(reading_parts)
                next_part = current_part
            
            part_bundles = [b for b, d in self.graph.nodes(data=True) 
                           if d['part'] == next_part]
            
            part_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                             reverse=True)
            
            recommendations['Reading'] = part_bundles[:3]
        else:
            # User hasn't started reading section, recommend part 5
            part_bundles = [b for b, d in self.graph.nodes(data=True) 
                           if d['part'] == 5]
            
            part_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                             reverse=True)
            
            recommendations['Reading'] = part_bundles[:3]
        
        return recommendations
    
    def _get_default_starting_point(self):
        """
        Get default starting point for new users.
        
        Returns:
            Dictionary with recommended starting bundles
        """
        # Start with easiest content from each section
        recommendations = {'Listening': [], 'Reading': []}
        
        # For Listening section, start with Part 1
        listening_bundles = [b for b, d in self.graph.nodes(data=True) 
                            if d['part'] == 1]
        
        # Sort by success rate (higher is easier)
        listening_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                              reverse=True)
        
        recommendations['Listening'] = listening_bundles[:3]
        
        # For Reading section, start with Part 5
        reading_bundles = [b for b, d in self.graph.nodes(data=True) 
                          if d['part'] == 5]
        
        reading_bundles.sort(key=lambda b: self.graph.nodes[b].get('success_rate', 0.5), 
                            reverse=True)
        
        recommendations['Reading'] = reading_bundles[:3]
        
        return recommendations
    
    def generate_learning_pathway(self, user_id=None, user_history=None, 
                                 assessment_results=None, target_score=None, 
                                 max_bundles=10, time_constraint=None):
        """
        Generate a personalized learning pathway.
        
        Args:
            user_id: User ID to generate pathway for
            user_history: Optional DataFrame with user history
            assessment_results: Optional dictionary with assessment results
            target_score: Optional target TOEIC score
            max_bundles: Maximum number of bundles in the pathway
            time_constraint: Optional time constraint in hours
            
        Returns:
            Dictionary with learning pathway
        """
        # Get starting point
        starting_point = self.get_user_starting_point(user_id, user_history, assessment_results)
        
        # Calculate target level based on target score
        target_level = self._calculate_target_level(target_score)
        
        # Build pathway
        pathway = {
            'user_id': user_id,
            'target_score': target_score,
            'target_level': target_level,
            'generated_at': datetime.now().isoformat(),
            'sections': []
        }
        
        # Process each section
        total_bundles = 0
        total_time = 0
        
        for section in ['Listening', 'Reading']:
            # Get starting bundles for this section
            starting_bundles = starting_point.get(section, [])
            
            if not starting_bundles:
                continue
            
            # Create section pathway
            section_pathway = {
                'name': section,
                'bundles': []
            }
            
            # Start with first recommended bundle
            current_bundle = starting_bundles[0]
            visited_bundles = set()
            
            # Add bundles until we reach max or run out of options
            while (total_bundles < max_bundles and 
                  (time_constraint is None or total_time < time_constraint)):
                
                # Avoid cycles
                if current_bundle in visited_bundles:
                    break
                    
                visited_bundles.add(current_bundle)
                
                # Get bundle information
                bundle_info = self.graph.nodes[current_bundle]
                
                # Add to pathway
                bundle_entry = {
                    'bundle_id': current_bundle,
                    'part': bundle_info['part'],
                    'part_name': bundle_info['part_name'],
                    'difficulty': bundle_info['difficulty'],
                    'question_count': bundle_info['question_count'],
                    'time_allocation': bundle_info['time_allocation'],
                    'skills': bundle_info['skills'],
                    'description': bundle_info['description']
                }
                
                section_pathway['bundles'].append(bundle_entry)
                
                # Update counters
                total_bundles += 1
                total_time += bundle_info['time_allocation'] / 60  # Convert to hours
                
                # Find next bundle
                next_bundle = self._find_next_bundle(current_bundle, visited_bundles, target_level)
                
                if next_bundle is None:
                    break
                    
                current_bundle = next_bundle
            
            # Add section to pathway
            pathway['sections'].append(section_pathway)
        
        # Add additional metadata
        pathway['total_bundles'] = total_bundles
        pathway['estimated_hours'] = total_time
        pathway['difficulty_distribution'] = self._calculate_difficulty_distribution(pathway)
        
        return pathway
    
    def _calculate_target_level(self, target_score):
        """
        Calculate target difficulty level based on target score.
        
        Args:
            target_score: Target TOEIC score (0-990)
            
        Returns:
            Dictionary with target level information
        """
        if target_score is None:
            # Default to medium level
            return {
                'listening_part': 2,
                'reading_part': 6,
                'difficulty': 'Medium'
            }
        
        # TOEIC score ranges
        # 0-400: Beginner
        # 405-600: Intermediate
        # 605-780: Advanced
        # 785-990: Proficient
        
        if target_score < 400:
            return {
                'listening_part': 1,
                'reading_part': 5,
                'difficulty': 'Easy'
            }
        elif target_score < 600:
            return {
                'listening_part': 2,
                'reading_part': 6,
                'difficulty': 'Medium'
            }
        elif target_score < 780:
            return {
                'listening_part': 3,
                'reading_part': 7,
                'difficulty': 'Medium'
            }
        else:
            return {
                'listening_part': 4,
                'reading_part': 7,
                'difficulty': 'Hard'
            }
    
    def _find_next_bundle(self, current_bundle, visited_bundles, target_level):
        """
        Find the next bundle in the learning pathway.
        
        Args:
            current_bundle: Current bundle ID
            visited_bundles: Set of already visited bundle IDs
            target_level: Target level information
            
        Returns:
            Next bundle ID or None if no suitable bundle found
        """
        # Get outgoing edges
        neighbors = list(self.graph.successors(current_bundle))
        
        # Filter out already visited bundles
        neighbors = [n for n in neighbors if n not in visited_bundles]
        
        if not neighbors:
            return None
        
        # Get current bundle information
        current_info = self.graph.nodes[current_bundle]
        current_part = current_info['part']
        current_section = current_info['section']
        
        # Strategy: prioritize progression through parts
        target_part = (target_level['listening_part'] if current_section == 'Listening' 
                      else target_level['reading_part'])
        
        # If below target part, try to progress to next part
        if current_part < target_part:
            next_part_neighbors = [n for n in neighbors 
                                  if self.graph.nodes[n]['part'] == current_part + 1]
            
            if next_part_neighbors:
                # Sort by success rate (higher is easier)
                next_part_neighbors.sort(
                    key=lambda n: self.graph.nodes[n].get('success_rate', 0.5),
                    reverse=True
                )
                return next_part_neighbors[0]
        
        # Otherwise, find similar difficulty in same part
        same_part_neighbors = [n for n in neighbors 
                              if self.graph.nodes[n]['part'] == current_part]
        
        if same_part_neighbors:
            # Sort by similarity to target difficulty
            target_difficulty = target_level['difficulty']
            
            def difficulty_distance(node):
                node_difficulty = self.graph.nodes[node]['difficulty']
                difficulty_levels = ['Easy', 'Medium', 'Hard']
                
                if node_difficulty not in difficulty_levels or target_difficulty not in difficulty_levels:
                    return 1  # Maximum distance
                
                node_idx = difficulty_levels.index(node_difficulty)
                target_idx = difficulty_levels.index(target_difficulty)
                
                return abs(node_idx - target_idx)
            
            same_part_neighbors.sort(key=difficulty_distance)
            return same_part_neighbors[0]
        
        # If no good match, just take any unvisited neighbor
        return neighbors[0]
    
    def _calculate_difficulty_distribution(self, pathway):
        """
        Calculate difficulty distribution in the pathway.
        
        Args:
            pathway: Learning pathway dictionary
            
        Returns:
            Dictionary with difficulty distribution
        """
        difficulties = {
            'Easy': 0,
            'Medium': 0,
            'Hard': 0
        }
        
        # Count bundles by difficulty
        for section in pathway['sections']:
            for bundle in section['bundles']:
                difficulty = bundle['difficulty']
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        # Calculate percentages
        total = sum(difficulties.values())
        if total > 0:
            difficulties = {k: (v / total) * 100 for k, v in difficulties.items()}
        
        return difficulties
    
    def visualize_pathway(self, pathway):
        """
        Create a visual representation of the learning pathway.
        
        Args:
            pathway: Learning pathway dictionary
            
        Returns:
            Plotly figure object
        """
        # Extract data for visualization
        sections = []
        parts = []
        bundle_ids = []
        start_times = []
        durations = []
        difficulties = []
        
        # Current time for simulation
        current_time = datetime.now()
        
        for section in pathway['sections']:
            section_name = section['name']
            
            for bundle in section['bundles']:
                sections.append(section_name)
                parts.append(f"Part {bundle['part']}: {bundle['part_name']}")
                bundle_ids.append(str(bundle['bundle_id']))
                
                # Add time
                start_times.append(current_time)
                duration_hours = bundle['time_allocation'] / 60  # Convert minutes to hours
                durations.append(duration_hours)
                
                # Update current time
                current_time += timedelta(hours=duration_hours)
                
                # Add difficulty
                difficulties.append(bundle['difficulty'])
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        task_names = [f"{s}: {p}<br>Bundle {b}" for s, p, b in zip(sections, parts, bundle_ids)]
        
        # Set colors based on difficulty
        color_map = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
        colors = [color_map.get(d, 'blue') for d in difficulties]
        
        # Add Gantt chart
        for i in range(len(task_names)):
            fig.add_trace(go.Bar(
                x=[durations[i]],
                y=[task_names[i]],
                orientation='h',
                marker_color=colors[i],
                name=difficulties[i],
                text=difficulties[i],
                textposition='auto',
                hoverinfo='text',
                hovertext=f"Bundle {bundle_ids[i]}<br>Difficulty: {difficulties[i]}<br>Duration: {durations[i]:.1f} hours"
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Learning Pathway",
            xaxis_title="Estimated Time (hours)",
            yaxis_title="Learning Activities",
            height=600,
            barmode='stack',
            showlegend=True
        )
        
        return fig
    
    def export_pathway(self, pathway, format='json'):
        """
        Export the learning pathway to a specified format.
        
        Args:
            pathway: Learning pathway dictionary
            format: Export format ('json', 'html', 'calendar')
            
        Returns:
            String or file path depending on format
        """
        if format == 'json':
            return json.dumps(pathway, indent=2)
        elif format == 'html':
            # Create HTML representation
            html = "<html><body>"
            html += f"<h1>Learning Pathway for User {pathway['user_id']}</h1>"
            html += f"<p>Target Score: {pathway['target_score']}</p>"
            html += f"<p>Total Bundles: {pathway['total_bundles']}</p>"
            html += f"<p>Estimated Hours: {pathway['estimated_hours']:.1f}</p>"
            
            for section in pathway['sections']:
                html += f"<h2>{section['name']} Section</h2>"
                html += "<ol>"
                
                for bundle in section['bundles']:
                    html += "<li>"
                    html += f"<strong>Bundle {bundle['bundle_id']}</strong>: "
                    html += f"{bundle['part_name']} ({bundle['difficulty']})<br>"
                    html += f"Time: {bundle['time_allocation']} minutes<br>"
                    html += f"Skills: {', '.join(bundle['skills'])}<br>"
                    html += f"Description: {bundle['description']}"
                    html += "</li>"
                
                html += "</ol>"
            
            html += "</body></html>"
            return html
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_toeic_structure_description(self):
        """
        Get a description of the TOEIC test structure.
        
        Returns:
            Dictionary with TOEIC structure information
        """
        # Create a summary of the TOEIC test
        toeic_summary = {
            "test_name": "TOEIC (Test of English for International Communication)",
            "total_time": "2 hours",
            "total_questions": 200,
            "score_range": "10-990",
            "sections": [
                {
                    "name": "Listening",
                    "parts": [self.toeic_structure[i] for i in range(1, 5)],
                    "time": "45 minutes",
                    "questions": 100
                },
                {
                    "name": "Reading",
                    "parts": [self.toeic_structure[i] for i in range(5, 8)],
                    "time": "75 minutes",
                    "questions": 100
                }
            ]
        }
        
        return toeic_summary