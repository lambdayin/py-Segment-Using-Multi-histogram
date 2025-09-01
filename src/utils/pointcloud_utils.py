"""
Point cloud processing utilities
Translated from MATLAB implementation
"""
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def load_point_cloud(filepath):
    """
    Load point cloud from text file
    
    Args:
        filepath: Path to .txt file containing point cloud data
    Returns:
        Nx3 numpy array of points [X, Y, Z]
    """
    try:
        # First try to load with default delimiter (space/tab)
        points = np.loadtxt(filepath)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points[:, :3]  # Take only first 3 columns (X, Y, Z)
    except Exception:
        try:
            # Try with comma delimiter
            points = np.loadtxt(filepath, delimiter=',')
            if points.ndim == 1:
                points = points.reshape(1, -1)
            return points[:, :3]  # Take only first 3 columns (X, Y, Z)
        except Exception as e:
            print(f"Error loading point cloud from {filepath}: {e}")
            return np.empty((0, 3))

def downsample_point_cloud(points, voxel_size=0.2):
    """
    Downsample point cloud using voxel grid
    
    Args:
        points: Nx3 point cloud
        voxel_size: Size of voxel for downsampling
    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points)

def remove_duplicates(points, tolerance=1e-6):
    """
    Remove duplicate points from point cloud
    
    Args:
        points: Nx3 point cloud
        tolerance: Distance tolerance for considering points as duplicates
    Returns:
        Point cloud with duplicates removed
    """
    if len(points) == 0:
        return points
    
    # Use numpy unique with tolerance
    # Round to handle floating point precision
    decimal_places = int(-np.log10(tolerance))
    rounded = np.round(points, decimal_places)
    unique_points = np.unique(rounded, axis=0)
    
    return unique_points

def cluster_points_dbscan(points, eps=1.0, min_samples=1):
    """
    Cluster points using DBSCAN algorithm
    
    Args:
        points: Nx3 point cloud
        eps: Maximum distance between samples in same neighborhood
        min_samples: Minimum number of samples in neighborhood for core point
    Returns:
        Array of cluster labels (-1 for noise)
    """
    if len(points) < min_samples:
        return np.array([-1] * len(points))
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points)
    return labels

def extract_largest_cluster(points, eps=1.0, min_samples=1):
    """
    Extract the largest cluster from point cloud
    
    Args:
        points: Nx3 point cloud  
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
    Returns:
        Points belonging to largest cluster
    """
    if len(points) == 0:
        return points
    
    labels = cluster_points_dbscan(points, eps, min_samples)
    
    # Find largest cluster (excluding noise points)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return points  # Return all points if no valid clusters
    
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    
    return points[labels == largest_label]

def split_point_cloud_2d(points, eps=0.1, min_clusters=1, max_clusters=10):
    """
    Split point cloud into clusters using 2D DBSCAN
    Translated from SplitPowerLine_2D.m
    
    Args:
        points: Nx3 point cloud
        eps: DBSCAN eps parameter  
        min_clusters: Minimum number of clusters expected
        max_clusters: Maximum number of clusters to return
    Returns:
        List of point cloud clusters
    """
    if len(points) == 0:
        return []
    
    # Use only X,Y coordinates for 2D clustering
    points_2d = points[:, :2]
    labels = cluster_points_dbscan(points_2d, eps=eps, min_samples=1)
    
    # Group points by cluster label
    clusters = []
    unique_labels = np.unique(labels)
    
    # Remove noise label if present
    unique_labels = unique_labels[unique_labels >= 0]
    
    for label in unique_labels[:max_clusters]:
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    
    return clusters

def split_point_cloud_c4(points, n_clusters=2, eps=0.5):
    """
    Split point cloud using clustering (C4 method)
    Translated from SplitPowerLine_C4.m
    
    Args:
        points: Nx3 point cloud
        n_clusters: Expected number of clusters
        eps: DBSCAN eps parameter
    Returns:
        List of point cloud clusters
    """
    if len(points) == 0:
        return []
    
    labels = cluster_points_dbscan(points, eps=eps, min_samples=1)
    
    # Group by cluster
    clusters = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Remove noise
    
    # Sort clusters by size (largest first)
    cluster_sizes = []
    cluster_points_list = []
    
    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_sizes.append(len(cluster_points))
        cluster_points_list.append(cluster_points)
    
    # Sort by size descending
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    
    # Return top n_clusters
    for i in sorted_indices[:n_clusters]:
        clusters.append(cluster_points_list[i])
    
    return clusters

def split_point_cloud_c5(points, n_clusters=4, eps=0.5):
    """
    Split point cloud using clustering (C5 method)
    Translated from SplitPowerLine_C5.m
    
    Args:
        points: Nx3 point cloud
        n_clusters: Expected number of clusters
        eps: DBSCAN eps parameter  
    Returns:
        List of point cloud clusters
    """
    return split_point_cloud_c4(points, n_clusters, eps)

def merge_cell_results(cell_results):
    """
    Merge results from multiple cells
    Translated from mergeCell3.m
    
    Args:
        cell_results: List of point cloud arrays
    Returns:
        tuple: (merged_points, lengths_array)
    """
    if not cell_results or len(cell_results) == 0:
        return np.empty((0, 3)), np.array([])
    
    all_points = []
    lengths = []
    
    for cell in cell_results:
        if len(cell) > 0:
            all_points.append(cell)
            # Calculate approximate length as distance between min/max Z
            if len(cell) > 1:
                length = np.max(cell[:, 2]) - np.min(cell[:, 2])
            else:
                length = 0.0
            lengths.append(length)
        else:
            lengths.append(0.0)
    
    if all_points:
        merged_points = np.vstack(all_points)
    else:
        merged_points = np.empty((0, 3))
    
    return merged_points, np.array(lengths)