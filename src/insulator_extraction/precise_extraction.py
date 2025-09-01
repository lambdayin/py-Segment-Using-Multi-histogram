"""
Precise insulator extraction methods - exact 1:1 MATLAB translation
Translated from InsExtrat_Partone.m and related extraction functions
"""
import numpy as np
from ..utils.math_utils import rotate_with_axle, rotz, roty
from ..utils.projection_utils import bin_projection
from ..utils.histogram_utils import get_cut_pos1
from ..utils.pointcloud_utils import cluster_points_dbscan, remove_duplicates

def ins_extrat_partone(line_points, tower_points, grid_width):
    """
    Detailed extraction logic for individual phases
    Exact 1:1 translation from MATLAB InsExtrat_Partone.m
    
    Args:
        line_points: Nx3 array of power line points for this phase
        tower_points: Nx3 array of tower points
        grid_width: Grid cell size for projection
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    # Initialize empty results
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    print(f"DEBUG ins_extrat_partone: {len(line_points)} line points, {len(tower_points)} tower points, grid={grid_width}")
    
    # Skip if insufficient points
    if len(line_points) < 5 or len(tower_points) == 0:
        print(f"DEBUG ins_extrat_partone: Insufficient points, skipping")
        return insulator_points, insulator_length
    
    try:
        # Step 1: Dual-axis alignment (lines 72-73 in MATLAB)
        # First rotation: Align with Z-axis
        line_r3, theta1 = rotate_with_axle(line_points, axis=3)
        
        # Second rotation: Align with Y-axis  
        line_r32, theta2 = rotate_with_axle(line_r3, axis=2)
        
        # Step 2: Binary projection to X-Z plane (line 74)
        bin_xy, _ = bin_projection(line_r32, grid_width, axis_x=0, axis_y=2)
        
        # Step 3: Cut position detection (line 75)
        cut_pos = get_cut_pos1(bin_xy)
        
        if cut_pos <= 0:
            # No valid cut position found
            return insulator_points, insulator_length
        
        # Step 4: Extract line points above cut position (lines 77-81)
        # Convert cut_pos from 1-based to 0-based indexing
        cut_pos_0based = cut_pos - 1
        
        # Apply same rotations to tower points for consistency
        tower_r1 = tower_points @ rotz(np.degrees(theta1)).T
        tower_r2 = tower_r1 @ roty(np.degrees(theta2)).T
        
        # Extract line insulator points (above cut position)
        if cut_pos_0based < bin_xy.shape[0]:
            # Find line points corresponding to rows above cut position
            line_z_min = np.min(line_r32[:, 2])
            line_z_max = np.max(line_r32[:, 2])
            z_range = line_z_max - line_z_min
            
            if z_range > 0:
                # Calculate Z threshold based on cut position
                z_threshold = line_z_min + (cut_pos_0based / bin_xy.shape[0]) * z_range
                
                # Extract line points above threshold
                ins_line_mask = line_r32[:, 2] >= z_threshold
                ins_line_r = line_r32[ins_line_mask]
                
                # Step 5: Apply geometric constraints to tower points (lines 82-87)
                if len(ins_line_r) > 0:
                    # Define bounding box from line insulator points
                    line_x_min, line_x_max = np.min(ins_line_r[:, 0]), np.max(ins_line_r[:, 0])
                    line_y_min, line_y_max = np.min(ins_line_r[:, 1]), np.max(ins_line_r[:, 1])
                    line_z_min_ins = np.min(ins_line_r[:, 2])
                    
                    # Extract tower points within bounding box
                    tower_mask = (
                        (tower_r2[:, 0] >= line_x_min) & 
                        (tower_r2[:, 0] <= line_x_max) &
                        (tower_r2[:, 1] >= line_y_min) & 
                        (tower_r2[:, 1] <= line_y_max) &
                        (tower_r2[:, 2] >= line_z_min_ins)  # Above line insulator
                    )
                    
                    ins_tower_r = tower_r2[tower_mask]
                    
                    # Step 6: Merge insulator points (lines 88-90)
                    if len(ins_tower_r) > 0:
                        # Combine line and tower insulator points
                        all_insulator_points = np.vstack([ins_line_r, ins_tower_r])
                    else:
                        all_insulator_points = ins_line_r
                    
                    # Step 7: Reverse rotations to original coordinate system
                    # Reverse Y-axis rotation
                    insulator_r1 = all_insulator_points @ roty(-np.degrees(theta2)).T
                    
                    # Reverse Z-axis rotation  
                    insulator_original = insulator_r1 @ rotz(-np.degrees(theta1)).T
                    
                    # Remove duplicates
                    insulator_points = remove_duplicates(insulator_original, tolerance=1e-6)
                    
                    # Calculate insulator length
                    if len(insulator_points) > 1:
                        insulator_length = (np.max(insulator_points[:, 2]) - 
                                          np.min(insulator_points[:, 2]))
                    else:
                        insulator_length = 0.0
        
    except Exception as e:
        # Handle extraction failure silently (matching MATLAB try-catch)
        print(f"Warning: Insulator extraction failed: {e}")
        return np.empty((0, 3)), 0.0
    
    return insulator_points, insulator_length

def ins_extract_zl(tower_points, cross_line_clusters, cross_locations, grid_width, tower_type):
    """
    Main vertical insulator extraction algorithm
    Exact 1:1 translation from MATLAB InsExtract_ZL.m
    
    Args:
        tower_points: Nx3 array of tower point cloud
        cross_line_clusters: List of power line clusters
        cross_locations: Cross-arm location information
        grid_width: Grid cell size
        tower_type: Type of tower
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    # Initialize results
    all_insulator_points = []
    all_insulator_lengths = []
    
    print(f"DEBUG ins_extract_zl: {len(tower_points)} tower points, {len(cross_line_clusters)} line clusters, {len(cross_locations)} cross locations")
    
    if not cross_line_clusters or len(tower_points) == 0:
        print("DEBUG ins_extract_zl: Empty inputs, returning empty results")
        return np.empty((0, 3)), 0.0
    
    # Determine number of cross-arms to process based on tower type
    if tower_type in [1, 8]:  # Wine glass, portal towers
        cross_num = min(len(cross_locations), len(cross_line_clusters))
    elif tower_type == 2:  # Cat head tower
        cross_num = min(2, len(cross_locations), len(cross_line_clusters))
    elif tower_type == 6:  # DC drum tower
        cross_num = min(len(cross_locations), len(cross_line_clusters))
    else:
        cross_num = min(len(cross_locations), len(cross_line_clusters))
    
    print(f"DEBUG ins_extract_zl: Processing {cross_num} cross-arms for tower type {tower_type}")
    
    # Process each cross-arm
    for cross_arm_idx in range(cross_num):
        try:
            print(f"DEBUG ins_extract_zl: Processing cross-arm {cross_arm_idx}")
            # Get power line points for this cross-arm
            if cross_arm_idx < len(cross_line_clusters):
                cross_line = cross_line_clusters[cross_arm_idx]
                print(f"DEBUG ins_extract_zl: Cross-arm {cross_arm_idx} has {len(cross_line)} line points")
            else:
                print(f"DEBUG ins_extract_zl: Cross-arm {cross_arm_idx} not available in clusters")
                continue
                
            if len(cross_line) < 5:
                print(f"DEBUG ins_extract_zl: Cross-arm {cross_arm_idx} has insufficient points ({len(cross_line)})")
                continue
            
            # Apply DBSCAN clustering to group line segments
            labels = cluster_points_dbscan(cross_line, eps=0.5, min_samples=1)
            
            # Find largest cluster
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]  # Remove noise
            
            if len(unique_labels) == 0:
                continue
            
            # Get largest cluster
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            largest_cluster_label = unique_labels[np.argmax(cluster_sizes)]
            main_line_points = cross_line[labels == largest_cluster_label]
            
            # Rotate line to find transmission direction
            rotated_line, theta = rotate_with_axle(main_line_points, axis=3)
            
            # Define cross-arm boundaries using Z-coordinates
            if cross_arm_idx < len(cross_locations):
                loc_start, loc_end = cross_locations[cross_arm_idx]
                
                print(f"DEBUG ins_extract_zl: Cross-arm {cross_arm_idx} Z range: [{loc_start}, {loc_end}]")
                
                # Convert from image row indices to actual Z coordinates
                # The cross_locations are in image space, need to convert to world coordinates
                tower_z_min, tower_z_max = np.min(tower_points[:, 2]), np.max(tower_points[:, 2])
                total_z_range = tower_z_max - tower_z_min
                
                # Get the binary image height used for cross-arm detection
                bin_image_yz, _ = bin_projection(tower_points, grid_width, axis_x=1, axis_y=2)
                img_height = bin_image_yz.shape[0]
                
                # Convert image row indices to actual Z coordinates  
                # Note: image coordinates are from top to bottom, so flip
                z_actual_start = tower_z_min + ((img_height - loc_end) / img_height) * total_z_range
                z_actual_end = tower_z_min + ((img_height - loc_start) / img_height) * total_z_range
                
                print(f"DEBUG ins_extract_zl: Actual Z range: [{z_actual_start:.2f}, {z_actual_end:.2f}]")
                print(f"DEBUG ins_extract_zl: Tower Z range: [{tower_z_min:.2f}, {tower_z_max:.2f}]")
                
                # Extend boundaries by 3 grid widths (matching MATLAB logic)
                z_offset = 3 * grid_width
                z_min = z_actual_start - z_offset
                z_max = z_actual_end + z_offset
                
                # Filter tower points within cross-arm Z range
                tower_in_range = tower_points[
                    (tower_points[:, 2] >= z_min) & 
                    (tower_points[:, 2] <= z_max)
                ]
                
                print(f"DEBUG ins_extract_zl: Tower points in range: {len(tower_in_range)}")
            else:
                tower_in_range = tower_points
                print(f"DEBUG ins_extract_zl: Using all tower points: {len(tower_in_range)}")
            
            # Split into 3 phases based on Y-coordinate divisions
            if len(rotated_line) > 0:
                y_min, y_max = np.min(rotated_line[:, 1]), np.max(rotated_line[:, 1])
                third_len = (y_max - y_min) / 3.0
                
                # Process each of the 3 phases
                for phase in range(3):
                    phase_y_min = y_min + phase * third_len
                    phase_y_max = y_min + (phase + 1) * third_len
                    
                    # Extract line points for this phase
                    phase_line_mask = (
                        (rotated_line[:, 1] >= phase_y_min) & 
                        (rotated_line[:, 1] <= phase_y_max)
                    )
                    phase_line_points = rotated_line[phase_line_mask]
                    
                    if len(phase_line_points) < 5:
                        continue
                    
                    # Reverse rotation to get original coordinates
                    original_phase_line = phase_line_points @ rotz(-np.degrees(theta)).T
                    
                    # Extract insulators for this phase
                    phase_insulators, phase_length = ins_extrat_partone(
                        original_phase_line, tower_in_range, grid_width
                    )
                    
                    if len(phase_insulators) > 0:
                        all_insulator_points.append(phase_insulators)
                        all_insulator_lengths.append(phase_length)
        
        except Exception as e:
            # Handle cross-arm processing failure silently
            print(f"Warning: Cross-arm {cross_arm_idx} processing failed: {e}")
            continue
    
    # Combine all insulator points
    if all_insulator_points:
        combined_insulators = np.vstack(all_insulator_points)
        combined_length = np.sum(all_insulator_lengths)
        
        # Remove duplicates from final result
        final_insulators = remove_duplicates(combined_insulators, tolerance=1e-6)
        
        return final_insulators, combined_length
    else:
        return np.empty((0, 3)), 0.0

def ins_extract_zl1(tower_points, cross_line_clusters, cross_locations, grid_width):
    """
    Vertical insulator extraction for cat head towers (ZL1 method)
    Translated from InsExtract_ZL1.m
    
    Args:
        tower_points: Nx3 tower point cloud
        cross_line_clusters: List of line point clusters
        cross_locations: Cross-arm location information
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, is_cable, insulator_length)
    """
    # This is a specialized version of InsExtract_ZL for cat head towers
    # The main difference is in how it handles the positioning (vertical vs angled)
    
    insulator_points, insulator_length = ins_extract_zl(
        tower_points, cross_line_clusters, cross_locations, grid_width, tower_type=2
    )
    
    # For cat head towers, is_cable is typically 0 (indicating insulators, not cables)
    is_cable = 0
    
    return insulator_points, is_cable, insulator_length