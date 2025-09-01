"""
Adaptive grid optimization for insulator extraction
Translated from MATLAB adaptive_grid_tension.m and calcuV.m
"""
import numpy as np
from ..utils.math_utils import calc_verticality
from ..core.constants import VERTICALITY_THRESHOLD, MIN_INSULATOR_LENGTH, MAX_INSULATOR_LENGTH

def adaptive_grid_tension(all_insulators, all_lengths):
    """
    Adaptive grid selection for tension towers
    Translated from adaptive_grid_tension.m
    
    Args:
        all_insulators: List of lists containing insulator results for each grid width
                       Shape: (n_insulators, n_grid_widths) where each element is point cloud
        all_lengths: Array of insulator lengths for each (insulator, grid_width) combination
                    Shape: (n_insulators, n_grid_widths)
    Returns:
        tuple: (final_insulators, final_lengths, final_grid_indices, final_verticalities)
    """
    if not all_insulators or len(all_insulators) == 0:
        return np.empty((0, 3)), np.array([]), np.array([]), np.array([])
    
    # Convert to numpy arrays for easier processing
    all_lengths = np.array(all_lengths)
    
    # Get dimensions
    n_insulators = len(all_insulators)
    n_grid_widths = len(all_insulators[0]) if n_insulators > 0 else 0
    
    if n_grid_widths == 0:
        return np.empty((0, 3)), np.array([]), np.array([]), np.array([])
    
    # Calculate verticality for each insulator at each grid width
    verticalities = np.zeros((n_insulators, n_grid_widths))
    
    for i in range(n_insulators):
        for j in range(n_grid_widths):
            insulator_points = all_insulators[i][j]
            if len(insulator_points) > 0:
                try:
                    verticalities[i, j] = calc_verticality(insulator_points)
                except:
                    verticalities[i, j] = 0.0
            else:
                verticalities[i, j] = 0.0
    
    # Determine horizontal vs vertical insulators
    # Condition: sum(verticality < 0.8 & verticality > 0 & length > 1) - sum(verticality > 0) / 2 >= 0
    horizontal_mask = np.zeros(n_insulators, dtype=bool)
    
    for i in range(n_insulators):
        # Count valid horizontal measurements (low verticality, reasonable length)
        horizontal_count = np.sum(
            (verticalities[i, :] < VERTICALITY_THRESHOLD) & 
            (verticalities[i, :] > 0) & 
            (all_lengths[i, :] > MIN_INSULATOR_LENGTH)
        )
        
        # Count valid vertical measurements  
        vertical_count = np.sum(verticalities[i, :] > 0)
        
        # Determine if this insulator is predominantly horizontal
        horizontal_mask[i] = horizontal_count - vertical_count / 2 >= 0
    
    # Calculate optimal grid width for each type
    # For horizontal insulators: find grid width closest to mean length of horizontal insulators
    horizontal_insulators = verticalities < VERTICALITY_THRESHOLD
    valid_horizontal = (horizontal_insulators & 
                       (all_lengths > MAX_INSULATOR_LENGTH))
    
    if np.any(valid_horizontal):
        mean_horizontal_length = np.mean(all_lengths[valid_horizontal])
    else:
        mean_horizontal_length = 2.0  # Default value
    
    # For vertical insulators: find grid width closest to mean length of vertical insulators  
    vertical_insulators = verticalities >= VERTICALITY_THRESHOLD
    valid_vertical = (vertical_insulators & 
                     (all_lengths < MAX_INSULATOR_LENGTH))
    
    if np.any(valid_vertical):
        mean_vertical_length = np.mean(all_lengths[valid_vertical])
    else:
        mean_vertical_length = 1.5  # Default value
    
    # Select optimal grid width for each insulator
    optimal_grid_indices = np.zeros(n_insulators, dtype=int)
    
    for i in range(n_insulators):
        if horizontal_mask[i]:
            # For horizontal insulators, find grid width closest to horizontal mean
            length_diffs = np.abs(all_lengths[i, :] - mean_horizontal_length)
            optimal_grid_indices[i] = np.argmin(length_diffs)
        else:
            # For vertical insulators, find grid width closest to vertical mean  
            length_diffs = np.abs(all_lengths[i, :] - mean_vertical_length)
            optimal_grid_indices[i] = np.argmin(length_diffs)
    
    # Extract final results using optimal grid indices
    final_lengths = np.zeros(n_insulators)
    final_verticalities = np.zeros(n_insulators)
    final_insulators = []
    
    for i in range(n_insulators):
        grid_idx = optimal_grid_indices[i]
        final_lengths[i] = all_lengths[i, grid_idx]
        final_verticalities[i] = verticalities[i, grid_idx]
        final_insulators.append(all_insulators[i][grid_idx])
    
    # Combine all final insulator points with labels
    combined_insulators = np.empty((0, 4))  # X, Y, Z, Label
    current_label = 1
    
    for i, insulator_points in enumerate(final_insulators):
        if len(insulator_points) > 1:  # Only include non-empty results
            # Add label column
            labeled_points = np.column_stack([
                insulator_points,
                np.full(len(insulator_points), current_label)
            ])
            combined_insulators = np.vstack([combined_insulators, labeled_points])
            current_label += 1
    
    return combined_insulators, final_lengths, optimal_grid_indices, final_verticalities

def calculate_grid_quality_metrics(insulator_points, grid_width):
    """
    Calculate quality metrics for a specific grid width
    
    Args:
        insulator_points: Nx3 array of insulator points
        grid_width: Grid width used for extraction
    Returns:
        dict: Quality metrics including verticality, completeness, etc.
    """
    metrics = {
        'verticality': 0.0,
        'length': 0.0,
        'point_count': 0,
        'completeness': 0.0,
        'grid_width': grid_width
    }
    
    if len(insulator_points) == 0:
        return metrics
    
    # Calculate verticality
    metrics['verticality'] = calc_verticality(insulator_points)
    
    # Calculate length (Z-direction span)
    if len(insulator_points) > 1:
        metrics['length'] = np.max(insulator_points[:, 2]) - np.min(insulator_points[:, 2])
    
    # Point count
    metrics['point_count'] = len(insulator_points)
    
    # Completeness heuristic (points per unit length)
    if metrics['length'] > 0:
        metrics['completeness'] = metrics['point_count'] / metrics['length']
    
    return metrics

def select_optimal_grid_width(grid_widths, insulator_results, length_results):
    """
    Select optimal grid width based on multiple criteria
    
    Args:
        grid_widths: List of tested grid widths
        insulator_results: List of insulator extraction results for each grid width
        length_results: List of insulator lengths for each grid width
    Returns:
        tuple: (optimal_grid_width, optimal_index, quality_scores)
    """
    n_grids = len(grid_widths)
    quality_scores = np.zeros(n_grids)
    
    for i in range(n_grids):
        if len(insulator_results[i]) > 0:
            metrics = calculate_grid_quality_metrics(insulator_results[i], grid_widths[i])
            
            # Combined quality score (can be adjusted based on requirements)
            completeness_score = min(metrics['completeness'] / 100.0, 1.0)  # Normalize
            length_score = 1.0 if MIN_INSULATOR_LENGTH <= metrics['length'] <= MAX_INSULATOR_LENGTH else 0.5
            verticality_score = metrics['verticality'] if metrics['verticality'] < VERTICALITY_THRESHOLD else 1.0
            
            quality_scores[i] = (completeness_score * 0.4 + 
                               length_score * 0.4 + 
                               verticality_score * 0.2)
        else:
            quality_scores[i] = 0.0
    
    # Select grid width with highest quality score
    optimal_index = np.argmax(quality_scores)
    optimal_grid_width = grid_widths[optimal_index]
    
    return optimal_grid_width, optimal_index, quality_scores

def merge_insulator_results(insulator_lists, length_lists):
    """
    Merge insulator results from multiple processing rounds
    Translated from mergeCell3.m functionality
    
    Args:
        insulator_lists: List of insulator point arrays
        length_lists: List of corresponding lengths
    Returns:
        tuple: (merged_points, merged_lengths)
    """
    if not insulator_lists:
        return np.empty((0, 3)), np.array([])
    
    valid_insulators = []
    valid_lengths = []
    
    for insulators, lengths in zip(insulator_lists, length_lists):
        if len(insulators) > 0 and lengths > 0:
            valid_insulators.append(insulators)
            valid_lengths.append(lengths)
    
    if valid_insulators:
        merged_points = np.vstack(valid_insulators)
        merged_lengths = np.array(valid_lengths)
    else:
        merged_points = np.empty((0, 3))
        merged_lengths = np.array([])
    
    return merged_points, merged_lengths