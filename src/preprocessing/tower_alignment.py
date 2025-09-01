"""
Tower alignment and preprocessing functions
Translated from MATLAB RTower.m and related functions
"""
import numpy as np
from ..utils.math_utils import rotate_with_axle, maxk, rotz

def r_tower(tower_points):
    """
    Rotate tower point cloud to standard orientation
    Translated from RTower.m
    
    Args:
        tower_points: Nx3 array of tower point cloud
    Returns:
        tuple: (rotated_points, rotation_angle_rad)
    """
    if len(tower_points) == 0:
        return tower_points, 0.0
    
    # Remove noise points above the tower (similar to MATLAB implementation)
    top_z_values, top_indices = maxk(tower_points[:, 2], min(10, len(tower_points)))
    
    # Find significant drops in height to detect noise
    if len(top_z_values) > 1:
        z_diffs = np.diff(top_z_values)
        # Find the last significant drop (> 1m)
        significant_drops = np.where(z_diffs < -1.0)[0]
        
        if len(significant_drops) > 0:
            cut_index = top_indices[significant_drops[-1] + 1]
            # Remove points above the cut threshold
            cut_height = tower_points[cut_index, 2]
            tower_points = tower_points[tower_points[:, 2] <= cut_height]
    
    # Use top 3m of tower points to calculate main direction
    if len(tower_points) > 0:
        max_z = np.max(tower_points[:, 2])
        top_points = tower_points[tower_points[:, 2] > max_z - 3.0]
        
        if len(top_points) >= 3:
            rotated_points, theta = rotate_with_axle(top_points, axis=3)
            
            # Apply the same rotation to all tower points
            rotation_matrix = rotz(np.degrees(theta))
            rotated_tower = tower_points @ rotation_matrix.T
            
            return rotated_tower, theta
        else:
            return tower_points, 0.0
    else:
        return tower_points, 0.0

def preprocess_point_clouds(tower_points, line_points):
    """
    Preprocess tower and line point clouds
    
    Args:
        tower_points: Nx3 array of tower point cloud
        line_points: Mx3 array of line point cloud  
    Returns:
        tuple: (processed_tower_points, processed_line_points, rotation_angle)
    """
    # Rotate tower to standard orientation
    rotated_tower, theta = r_tower(tower_points)
    
    # Apply same rotation to line points
    if len(line_points) > 0:
        rotation_matrix = rotz(np.degrees(theta))
        rotated_line = line_points @ rotation_matrix.T
    else:
        rotated_line = line_points
    
    return rotated_tower, rotated_line, theta

def filter_noise_points(points, z_threshold_std=3.0):
    """
    Remove statistical outliers based on Z-coordinate
    
    Args:
        points: Nx3 point cloud
        z_threshold_std: Number of standard deviations for outlier detection
    Returns:
        Filtered point cloud
    """
    if len(points) == 0:
        return points
    
    z_coords = points[:, 2]
    z_mean = np.mean(z_coords)
    z_std = np.std(z_coords)
    
    # Remove points that are too far from the mean in Z direction
    threshold_high = z_mean + z_threshold_std * z_std
    threshold_low = z_mean - z_threshold_std * z_std
    
    valid_mask = (z_coords >= threshold_low) & (z_coords <= threshold_high)
    return points[valid_mask]

def normalize_point_cloud(points):
    """
    Normalize point cloud to have origin at minimum coordinates
    
    Args:
        points: Nx3 point cloud
    Returns:
        tuple: (normalized_points, translation_vector)
    """
    if len(points) == 0:
        return points, np.array([0, 0, 0])
    
    min_coords = np.min(points, axis=0)
    normalized_points = points - min_coords
    
    return normalized_points, min_coords