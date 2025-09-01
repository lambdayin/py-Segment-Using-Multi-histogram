"""
Specific insulator extraction methods
Translated from various MATLAB extraction functions
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from ..utils.math_utils import rotate_with_axle, calc_verticality, distance_point_to_line
from ..utils.projection_utils import bin_projection, sine_projection
from ..utils.pointcloud_utils import cluster_points_dbscan, remove_duplicates

def extract_insulators_vertical(tower_points, cross_line_clusters, cross_locations, grid_width, tower_type):
    """
    Extract vertical insulators (ZL method)
    Translated from InsExtract_ZL.m
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of power line clusters
        cross_locations: Cross-arm locations
        grid_width: Grid cell size
        tower_type: Type of tower
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    if not cross_line_clusters or len(tower_points) == 0:
        return insulator_points, insulator_length
    
    # Use the largest line cluster for processing
    if cross_line_clusters:
        # Find largest cluster by point count
        cluster_sizes = [len(cluster) for cluster in cross_line_clusters]
        largest_idx = np.argmax(cluster_sizes)
        main_line_cluster = cross_line_clusters[largest_idx]
        
        # Cluster the line points using DBSCAN
        if len(main_line_cluster) > 0:
            labels = cluster_points_dbscan(main_line_cluster, eps=1.0, min_samples=1)
            
            # Process each line segment
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]  # Remove noise
            
            all_insulators = []
            
            for label in unique_labels:
                line_segment = main_line_cluster[labels == label]
                
                # Rotate line segment to standard orientation
                if len(line_segment) >= 3:
                    rotated_line, theta = rotate_with_axle(line_segment, axis=3)
                    
                    # Extract insulator from this segment
                    segment_insulators = extract_insulator_from_line_segment(
                        tower_points, rotated_line, grid_width
                    )
                    
                    if len(segment_insulators) > 0:
                        all_insulators.append(segment_insulators)
            
            if all_insulators:
                insulator_points = np.vstack(all_insulators)
                # Calculate total length
                if len(insulator_points) > 1:
                    insulator_length = np.max(insulator_points[:, 2]) - np.min(insulator_points[:, 2])
    
    return insulator_points, insulator_length

def extract_insulator_from_line_segment(tower_points, line_segment, grid_width):
    """
    Extract insulator points from a single line segment
    Simplified version of InsExtrat_Partone.m
    
    Args:
        tower_points: Nx3 tower point cloud
        line_segment: Nx3 line segment points
        grid_width: Grid cell size
    Returns:
        Insulator points for this segment
    """
    if len(tower_points) == 0 or len(line_segment) == 0:
        return np.empty((0, 3))
    
    # Create XY projection for hole detection
    bin_image_xy, grid_cells = bin_projection(tower_points, grid_width, axis_x=0, axis_y=1)
    
    # Find cut positions using hole detection (simplified)
    cut_positions = find_cut_positions_from_holes(bin_image_xy, grid_width)
    
    # Extract points between tower and line
    if cut_positions:
        # Find points between tower structure and power lines
        tower_z_range = [np.min(tower_points[:, 2]), np.max(tower_points[:, 2])]
        line_z_range = [np.min(line_segment[:, 2]), np.max(line_segment[:, 2])]
        
        # Insulator should be between tower and line in Z direction
        z_min = max(tower_z_range[0], line_z_range[0])
        z_max = min(tower_z_range[1], line_z_range[1])
        
        if z_max > z_min:
            # Find points in the insulator region
            candidate_points = tower_points[
                (tower_points[:, 2] >= z_min) & 
                (tower_points[:, 2] <= z_max)
            ]
            
            if len(candidate_points) > 0:
                # Additional filtering based on distance to line
                return filter_insulator_candidates(candidate_points, line_segment)
    
    return np.empty((0, 3))

def extract_insulators_horizontal(tower_points, cross_line_clusters, cross_locations, grid_width):
    """
    Extract horizontal insulators 
    Placeholder for horizontal extraction methods
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of power line clusters
        cross_locations: Cross-arm locations
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    # This would implement the horizontal insulator extraction logic
    # from functions like InsExtractType4.m, InsExtractType51.m etc.
    
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    return insulator_points, insulator_length

def extract_insulators_type4(tower_points, cross_line_clusters, cross_locations, grid_width, tower_type):
    """
    Extract insulators for Type 4 towers (horizontal insulators)
    Translated from InsExtractType4.m
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of power line clusters
        cross_locations: Cross-arm locations  
        grid_width: Grid cell size
        tower_type: Type of tower
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    if not cross_line_clusters or len(tower_points) == 0:
        return insulator_points, insulator_length
    
    # Placeholder for Type 4 specific extraction
    # This would involve plane fitting, line splitting, etc.
    
    return insulator_points, insulator_length

def extract_insulators_type51(tower_points, cross_line_clusters, cross_locations, grid_width, tower_type):
    """
    Extract insulators for Type 5.1 towers (tension drum towers)
    Translated from InsExtractType51.m
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of power line clusters
        cross_locations: Cross-arm locations
        grid_width: Grid cell size  
        tower_type: Type of tower
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    # Placeholder for Type 5.1 specific extraction
    return insulator_points, insulator_length

def find_cut_positions_from_holes(binary_image, grid_width):
    """
    Find cut positions using hole detection in binary image
    Simplified version of getCutpos1.m functionality
    
    Args:
        binary_image: 2D binary image
        grid_width: Grid cell size
    Returns:
        List of cut positions
    """
    if binary_image.size == 0:
        return []
    
    # Create histogram along one axis
    histogram = np.sum(binary_image, axis=0)
    
    # Find empty regions (holes)
    empty_positions = np.where(histogram == 0)[0]
    
    if len(empty_positions) == 0:
        return []
    
    # Group consecutive empty positions
    cut_positions = []
    current_group = [empty_positions[0]]
    
    for i in range(1, len(empty_positions)):
        if empty_positions[i] - empty_positions[i-1] == 1:
            current_group.append(empty_positions[i])
        else:
            if len(current_group) > 2:  # Significant hole
                cut_positions.append(np.mean(current_group) * grid_width)
            current_group = [empty_positions[i]]
    
    # Add last group
    if len(current_group) > 2:
        cut_positions.append(np.mean(current_group) * grid_width)
    
    return cut_positions

def filter_insulator_candidates(candidate_points, line_segment):
    """
    Filter candidate insulator points based on proximity to power line
    
    Args:
        candidate_points: Nx3 candidate points
        line_segment: Mx3 power line points
    Returns:
        Filtered insulator points
    """
    if len(candidate_points) == 0 or len(line_segment) == 0:
        return candidate_points
    
    # Calculate distances from candidates to line points
    min_distances = []
    
    for candidate in candidate_points:
        distances_to_line = np.linalg.norm(line_segment - candidate, axis=1)
        min_dist = np.min(distances_to_line)
        min_distances.append(min_dist)
    
    min_distances = np.array(min_distances)
    
    # Keep points that are reasonably close to the power line
    # but not too close (indicating they're part of the line)
    distance_threshold_min = 0.5  # 0.5m minimum distance
    distance_threshold_max = 5.0  # 5.0m maximum distance
    
    valid_mask = (min_distances >= distance_threshold_min) & (min_distances <= distance_threshold_max)
    
    return candidate_points[valid_mask]

def fit_plane_ransac(points, max_trials=1000, residual_threshold=0.1):
    """
    Fit plane to points using RANSAC
    Simplified version of plane fitting functionality from fitPlane1.m
    
    Args:
        points: Nx3 point cloud
        max_trials: Maximum RANSAC iterations
        residual_threshold: Residual threshold for inliers
    Returns:
        tuple: (plane_normal, plane_point, inlier_mask)
    """
    if len(points) < 3:
        return None, None, None
    
    # Use RANSAC to fit plane (z = ax + by + c)
    X = points[:, :2]  # X, Y coordinates
    y = points[:, 2]   # Z coordinates
    
    ransac = RANSACRegressor(
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        random_state=42
    )
    
    try:
        ransac.fit(X, y)
        
        # Extract plane parameters
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Plane normal vector [a, b, -1] normalized
        normal = np.array([a, b, -1])
        normal = normal / np.linalg.norm(normal)
        
        # A point on the plane
        plane_point = np.array([0, 0, c])
        
        return normal, plane_point, ransac.inlier_mask_
        
    except Exception:
        return None, None, None