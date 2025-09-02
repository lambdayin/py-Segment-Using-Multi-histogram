"""
Core insulator extraction algorithm
Translated from MATLAB TypeInsdeTree.m
"""
import numpy as np
from ..core.constants import TowerType
from ..tower_detection.tower_classifier import detect_tower_type
from ..utils.projection_utils import bin_projection, cross_location_detection
from ..utils.pointcloud_utils import (
    split_point_cloud_2d, split_point_cloud_c4, split_point_cloud_c5,
    remove_duplicates
)
from .extraction_methods import (
    extract_insulators_vertical, extract_insulators_horizontal,
    extract_insulators_type4, extract_insulators_type51
)
from .precise_extraction import ins_extract_zl, ins_extract_zl1

def type_inside_tree(tower_points, line_points, grid_width):
    """
    Main insulator extraction algorithm dispatcher
    Translated from TypeInsdeTree.m
    
    Args:
        tower_points: Nx3 array of tower point cloud
        line_points: Mx3 array of line point cloud  
        grid_width: Grid cell size for processing
    Returns:
        tuple: (insulator_points, is_cable, insulator_length)
    """
    # Initialize return values
    insulator_points = np.empty((0, 3))
    is_cable = 0
    insulator_length = 0.0
    
    if len(tower_points) == 0:
        print(f"DEBUG: No tower points available")
        return insulator_points, is_cable, insulator_length
    
    print(f"DEBUG: Processing {len(tower_points)} tower points, {len(line_points)} line points, grid={grid_width}")
    
    # Detect tower type
    tower_type = detect_tower_type(tower_points, line_points)
    print(f"DEBUG: Detected tower type: {tower_type}")
    
    # Create binary projection for cross-arm detection
    bin_image_yz, _ = bin_projection(tower_points, grid_width, axis_x=1, axis_y=2)
    print(f"DEBUG: Binary projection size: {bin_image_yz.shape}")
    
    # Detect cross-arm locations
    cross_locations = cross_location_detection(bin_image_yz, ratio=3)
    print(f"DEBUG: Cross-arm locations: {cross_locations}")
    
    # Dispatch to appropriate extraction method based on tower type
    if tower_type in [TowerType.WINE_GLASS, TowerType.PORTAL]:
        # Wine glass tower, portal tower
        print(f"DEBUG: Processing wine glass/portal tower")
        cross_line_clusters = split_point_cloud_2d(line_points, eps=0.1, 
                                                  min_clusters=1, max_clusters=10)
        print(f"DEBUG: Line clusters: {len(cross_line_clusters)} clusters")
        for i, cluster in enumerate(cross_line_clusters):
            print(f"DEBUG: Cluster {i}: {len(cluster)} points")
        insulator_points, insulator_length = ins_extract_zl(
            tower_points, cross_line_clusters, cross_locations, grid_width, tower_type.value
        )
        print(f"DEBUG: Extracted {len(insulator_points)} insulator points, length={insulator_length}")
        
    elif tower_type == TowerType.CAT_HEAD:
        # Cat head tower
        print(f"DEBUG: Processing cat head tower")
        cross_line_clusters = split_point_cloud_c4(line_points, n_clusters=2, eps=0.6)
        print(f"DEBUG: Line clusters: {len(cross_line_clusters)} clusters")
        for i, cluster in enumerate(cross_line_clusters):
            print(f"DEBUG: Cluster {i}: {len(cluster)} points")
        insulator_points, is_cable, insulator_length = ins_extract_zl1(
            tower_points, cross_line_clusters, cross_locations, grid_width
        )
        print(f"DEBUG: Extracted {len(insulator_points)} insulator points, length={insulator_length}")
        
    elif tower_type == TowerType.SINGLE_CROSS:
        # Single cross-arm tower
        cross_line_clusters = split_point_cloud_c4(line_points, n_clusters=2, eps=0.5)
        # Append location information (simplified)
        extended_locations = cross_locations.copy() if cross_locations else []
        insulator_points, insulator_length = extract_insulators_type4(
            tower_points, cross_line_clusters, extended_locations, grid_width, tower_type
        )
        
    elif tower_type == TowerType.TENSION_DRY:
        # Tension resistant dry type tower
        # First split into 4 clusters, then further into 2
        line_clusters_c5 = split_point_cloud_c5(line_points, n_clusters=4, eps=0.5)
        if line_clusters_c5:
            # Merge clusters and split again
            merged_lines = np.vstack(line_clusters_c5) if line_clusters_c5 else np.empty((0, 3))
            cross_line_clusters = split_point_cloud_c4(merged_lines, n_clusters=2, eps=1.0)
        else:
            cross_line_clusters = []
            
        insulator_points, insulator_length = extract_insulators_type4(
            tower_points, cross_line_clusters, cross_locations, grid_width, tower_type
        )
        
    elif tower_type == TowerType.TENSION_DRUM:
        # Tension type drum tower
        cross_line_clusters = split_point_cloud_c4(line_points, n_clusters=3, eps=0.8)
        insulator_points, insulator_length = extract_insulators_type51(
            tower_points, cross_line_clusters, cross_locations, grid_width, tower_type
        )
        
    elif tower_type == TowerType.DC_DRUM:
        # DC drum tower
        print(f"DEBUG: Processing DC drum tower")
        cross_line_clusters = split_point_cloud_2d(line_points, eps=0.1, 
                                                  min_clusters=3, max_clusters=10)
        print(f"DEBUG: Line clusters: {len(cross_line_clusters)} clusters")
        insulator_points, insulator_length = ins_extract_zl(
            tower_points, cross_line_clusters, cross_locations, grid_width, tower_type.value
        )
        print(f"DEBUG: Extracted {len(insulator_points)} insulator points, length={insulator_length}")
    
    else:
        print(f"DEBUG: Unknown tower type {tower_type}, skipping extraction")
    
    print(f"DEBUG: type_inside_tree result: {len(insulator_points)} points, cable={is_cable}, length={insulator_length:.3f}")
    return insulator_points, is_cable, insulator_length

def extract_insulators_vertical_zl1(tower_points, cross_line_clusters, cross_locations, grid_width):
    """
    Vertical insulator extraction for cat head towers (ZL1 method)
    Simplified version of InsExtract_ZL1.m
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of line point clusters
        cross_locations: Cross-arm location information
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, is_cable, insulator_length)
    """
    # This is a placeholder for the complex ZL1 extraction method
    # The actual implementation would require detailed analysis of the
    # specific MATLAB functions InsExtract_ZL1.m, InsExtrat_Partone.m, etc.
    
    insulator_points = np.empty((0, 3))
    is_cable = 0
    insulator_length = 0.0
    
    if not cross_line_clusters or len(tower_points) == 0:
        return insulator_points, is_cable, insulator_length
    
    # Simplified extraction - would need full implementation
    # For now, return empty results
    return insulator_points, is_cable, insulator_length

def process_multi_grid_extraction(tower_points, line_points, grid_widths):
    """
    Process insulator extraction across multiple grid widths
    
    Args:
        tower_points: Nx3 tower point cloud
        line_points: Nx3 line point cloud
        grid_widths: List of grid widths to test
    Returns:
        tuple: (all_insulator_results, all_lengths)
    """
    all_insulator_results = []
    all_lengths = []
    
    for grid_width in grid_widths:
        ins_points, is_cable, ins_length = type_inside_tree(
            tower_points, line_points, grid_width
        )
        
        all_insulator_results.append(ins_points)
        all_lengths.append(ins_length)
    
    return all_insulator_results, np.array(all_lengths)