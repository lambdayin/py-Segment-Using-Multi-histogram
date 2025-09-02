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
    Exact 1:1 translation from MATLAB adaptive_grid_tension.m
    
    Args:
        all_insulators: 2D cell array (n_insulators x n_grid_widths)
        all_lengths: 2D array (n_insulators x n_grid_widths)
    Returns:
        tuple: (final_insulators, final_lengths, final_grid_indices, final_verticalities)
    """
    print(f"DEBUG adaptive_grid_tension: Input shape - {len(all_insulators)} insulators")
    
    if not all_insulators or len(all_insulators) == 0:
        return np.empty((0, 4)), np.array([]), np.array([]), np.array([])
    
    # Convert input to match MATLAB matrix format
    all_lengths = np.array(all_lengths)
    
    # Get dimensions (matching MATLAB lines 3-4)
    n_insulators = len(all_insulators)  # InsNum
    n_grid_widths = len(all_insulators[0]) if n_insulators > 0 else 0  # GridNum
    
    print(f"DEBUG: Processing {n_insulators} insulators across {n_grid_widths} grid widths")
    
    if n_grid_widths == 0:
        return np.empty((0, 4)), np.array([]), np.array([]), np.array([])
    
    # Calculate verticality matrix (MATLAB lines 5-15)
    verticalities = np.zeros((n_insulators, n_grid_widths))  # VeI
    
    for i in range(n_insulators):
        for j in range(n_grid_widths):
            try:
                insulator_points = all_insulators[i][j]
                if len(insulator_points) > 0:
                    verticalities[i, j] = calc_verticality(insulator_points)
                else:
                    verticalities[i, j] = 0.0
            except:
                verticalities[i, j] = 0.0
    
    print(f"DEBUG: Verticality matrix shape: {verticalities.shape}")
    print(f"DEBUG: Verticality range: [{np.min(verticalities):.3f}, {np.max(verticalities):.3f}]")
    
    # Determine horizontal vs vertical insulators (MATLAB line 17)
    # XYInsInd = sum(VeI < 0.8 & VeI > 0 & AllLen > 1,2) - sum(VeI > 0,2) / 2 >= 0
    horizontal_mask = np.zeros(n_insulators, dtype=bool)
    
    for i in range(n_insulators):
        # Count conditions that favor horizontal classification
        horizontal_count = np.sum(
            (verticalities[i, :] < 0.8) & 
            (verticalities[i, :] > 0) & 
            (all_lengths[i, :] > 1.0)
        )
        
        # Count total valid verticality measurements
        valid_count = np.sum(verticalities[i, :] > 0)
        
        # Apply MATLAB condition exactly
        horizontal_mask[i] = horizontal_count - valid_count / 2.0 >= 0
    
    print(f"DEBUG: Horizontal insulators: {np.sum(horizontal_mask)}/{n_insulators}")
    
    # Calculate target lengths for optimization (MATLAB lines 19-20)
    # Find mean length for horizontal insulators (VeI < 0.8 & AllLen > 2.5)
    horizontal_condition = (verticalities < 0.8) & (all_lengths > 2.5)
    if np.any(horizontal_condition):
        mean_horizontal_length = np.mean(all_lengths[horizontal_condition])
    else:
        mean_horizontal_length = 2.5  # Default for horizontal
    
    # Find mean length for vertical insulators (VeI >= 0.8 & AllLen < 2.5)
    vertical_condition = (verticalities >= 0.8) & (all_lengths < 2.5)
    if np.any(vertical_condition):
        mean_vertical_length = np.mean(all_lengths[vertical_condition])
    else:
        mean_vertical_length = 1.5  # Default for vertical
    
    print(f"DEBUG: Target lengths - Horizontal: {mean_horizontal_length:.3f}, Vertical: {mean_vertical_length:.3f}")
    
    # Calculate optimal grid indices for each insulator (MATLAB lines 19-21)
    optimal_indices = np.zeros(n_insulators, dtype=int)  # ResInd
    
    # For horizontal insulators: find grid closest to horizontal mean
    xy_indices = np.zeros(n_insulators, dtype=int)  # XYInd
    for i in range(n_insulators):
        length_diffs = np.abs(all_lengths[i, :] - mean_horizontal_length)
        xy_indices[i] = np.argmin(length_diffs)
    
    # For vertical insulators: find grid closest to vertical mean
    z_indices = np.zeros(n_insulators, dtype=int)  # ZInd
    for i in range(n_insulators):
        length_diffs = np.abs(all_lengths[i, :] - mean_vertical_length)
        z_indices[i] = np.argmin(length_diffs)
    
    # Assign based on horizontal/vertical classification
    for i in range(n_insulators):
        if horizontal_mask[i]:
            optimal_indices[i] = xy_indices[i]
        else:
            optimal_indices[i] = z_indices[i]
    
    print(f"DEBUG: Optimal grid indices: {optimal_indices}")
    
    # Extract final results (MATLAB lines 23-29)
    final_lengths = np.zeros(n_insulators)
    final_verticalities = np.zeros(n_insulators)
    
    for i in range(n_insulators):
        grid_idx = optimal_indices[i]
        final_lengths[i] = all_lengths[i, grid_idx]
        final_verticalities[i] = verticalities[i, grid_idx]
    
    # Combine insulator points with labels (MATLAB lines 30-39)
    final_insulators = np.empty((0, 4))  # [X, Y, Z, Label]
    current_label = 1
    
    for i in range(n_insulators):
        grid_idx = optimal_indices[i]
        insulator_points = all_insulators[i][grid_idx]
        
        if len(insulator_points) > 1:  # Only non-empty results
            # Add label column
            labeled_points = np.column_stack([
                insulator_points,
                np.full(len(insulator_points), current_label, dtype=int)
            ])
            
            if final_insulators.size == 0:
                final_insulators = labeled_points
            else:
                final_insulators = np.vstack([final_insulators, labeled_points])
            
            current_label += 1
    
    print(f"DEBUG: Final result - {len(final_insulators)} points, {current_label-1} insulators")
    print(f"DEBUG: Final lengths: {final_lengths}")
    print(f"DEBUG: Final verticalities: {final_verticalities}")
    
    return final_insulators, final_lengths, optimal_indices, final_verticalities

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

def merge_cell3(insulator_cells):
    """
    Merge cell array of insulators and calculate lengths
    Exact 1:1 translation from MATLAB mergeCell3.m
    
    Args:
        insulator_cells: Cell array of insulator point clouds
    Returns:
        tuple: (merged_points, total_length)
    """
    if not insulator_cells:
        return np.empty((0, 3)), 0.0
    
    valid_insulators = []
    lengths = []
    
    for cell in insulator_cells:
        if isinstance(cell, np.ndarray) and len(cell) > 0:
            valid_insulators.append(cell)
            # Calculate length as Z-range
            if len(cell) > 1:
                length = np.max(cell[:, 2]) - np.min(cell[:, 2])
            else:
                length = 0.0
            lengths.append(length)
    
    if valid_insulators:
        merged_points = np.vstack(valid_insulators)
        total_length = np.sum(lengths)
    else:
        merged_points = np.empty((0, 3))
        total_length = 0.0
    
    return merged_points, total_length