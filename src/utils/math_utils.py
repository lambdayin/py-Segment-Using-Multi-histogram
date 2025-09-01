"""
Mathematical utility functions
Translated from MATLAB implementation
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import open3d as o3d

def rotz(angle_deg):
    """
    Create rotation matrix around Z axis
    Args:
        angle_deg: Rotation angle in degrees
    Returns:
        3x3 rotation matrix
    """
    return R.from_euler('z', angle_deg, degrees=True).as_matrix()

def roty(angle_deg):
    """
    Create rotation matrix around Y axis
    Args:
        angle_deg: Rotation angle in degrees  
    Returns:
        3x3 rotation matrix
    """
    return R.from_euler('y', angle_deg, degrees=True).as_matrix()

def rotx(angle_deg):
    """
    Create rotation matrix around X axis
    Args:
        angle_deg: Rotation angle in degrees
    Returns:
        3x3 rotation matrix
    """
    return R.from_euler('x', angle_deg, degrees=True).as_matrix()

def rotate_with_axle(points, axis=3):
    """
    Rotate point cloud to align with specified axis using PCA
    Translated from RotawithAxle.m
    
    Args:
        points: Nx3 point cloud array
        axis: Axis to align with (1=X, 2=Y, 3=Z)
    Returns:
        tuple: (rotated_points, rotation_angle_rad)
    """
    if len(points) < 3:
        return points, 0.0
    
    # Downsample points for PCA calculation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(0.2)
    pts_down = np.asarray(downsampled.points)
    
    if axis == 3:  # Z-axis alignment
        # Use XY plane for direction calculation
        pts_2d = pts_down[:, :2]
        if len(pts_2d) < 2:
            return points, 0.0
            
        center = np.mean(pts_2d, axis=0)
        centered = pts_2d - center
        
        # Covariance matrix
        cov_matrix = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Principal direction vector
        principal_idx = np.argmax(eigenvals)
        direction_vector = eigenvecs[:, principal_idx]
        
        # Calculate rotation angle
        angle = np.arccos(np.abs(direction_vector[0]) / np.linalg.norm(direction_vector))
        
        # Adjust angle direction based on quadrant
        if direction_vector[0] * direction_vector[1] < 0:
            angle = -angle
            
        # Apply rotation
        rotation_matrix = rotz(np.degrees(angle))
        rotated_points = points @ rotation_matrix.T
        
        return rotated_points, angle
    
    else:
        # For other axes, use full 3D PCA
        pca = PCA(n_components=3)
        pca.fit(pts_down)
        
        # Get principal component
        principal_direction = pca.components_[0]
        
        if axis == 1:  # X-axis alignment
            target = np.array([1, 0, 0])
        elif axis == 2:  # Y-axis alignment  
            target = np.array([0, 1, 0])
        else:
            target = np.array([0, 0, 1])
        
        # Calculate rotation to align principal direction with target
        cross_product = np.cross(principal_direction, target)
        angle = np.arccos(np.clip(np.dot(principal_direction, target), -1.0, 1.0))
        
        if np.linalg.norm(cross_product) > 1e-6:
            axis_of_rotation = cross_product / np.linalg.norm(cross_product)
            rotation = R.from_rotvec(angle * axis_of_rotation)
            rotation_matrix = rotation.as_matrix()
            rotated_points = points @ rotation_matrix.T
        else:
            rotated_points = points
            angle = 0.0
        
        return rotated_points, angle

def maxk(array, k):
    """
    Find k largest values and their indices
    Equivalent to MATLAB's maxk function
    
    Args:
        array: Input array
        k: Number of maximum values to find
    Returns:
        tuple: (values, indices)
    """
    array = np.asarray(array)
    if k >= len(array):
        sorted_idx = np.argsort(array)[::-1]
        return array[sorted_idx], sorted_idx
    
    # Get k largest indices
    indices = np.argpartition(array, -k)[-k:]
    # Sort them by value in descending order
    sorted_indices = indices[np.argsort(array[indices])[::-1]]
    values = array[sorted_indices]
    
    return values, sorted_indices

def max_label(labels):
    """
    Find the most frequent label (equivalent to MaxLabel.m)
    
    Args:
        labels: Array of cluster labels
    Returns:
        Most frequent label
    """
    labels = labels[labels >= 0]  # Remove noise points (-1)
    if len(labels) == 0:
        return -1
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_idx = np.argmax(counts)
    return unique_labels[max_idx]

def calc_verticality(points):
    """
    Calculate verticality of point cloud using PCA
    Translated from calcuV.m
    
    Args:
        points: Nx3 point cloud
    Returns:
        Verticality value (0-1, higher = more vertical)
    """
    if len(points) < 3:
        return 0.0
    
    # Remove duplicate points
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return 0.0
    
    # Calculate PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Get the ratio of the largest eigenvalue (main direction)
    # to the sum of all eigenvalues
    eigenvals = pca.explained_variance_
    
    # Verticality is related to how much variation is in Z direction
    # vs. horizontal directions
    z_component = np.abs(pca.components_[0, 2])  # Z component of first PC
    
    return z_component

def distance_point_to_line(points, line_params):
    """
    Calculate distance from points to a 2D line
    
    Args:
        points: Nx2 array of points
        line_params: [slope, intercept] of line y = slope*x + intercept
    Returns:
        Array of distances
    """
    slope, intercept = line_params
    # Distance formula: |ax + by + c| / sqrt(a^2 + b^2)
    # For y = mx + b -> mx - y + b = 0, so a=m, b=-1, c=b
    distances = np.abs(slope * points[:, 0] - points[:, 1] + intercept)
    distances /= np.sqrt(slope**2 + 1)
    return distances