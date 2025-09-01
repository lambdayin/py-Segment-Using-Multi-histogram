"""
Line splitting utilities for insulator extraction
Translated from MATLAB functions SplitOverline4Mid1.m, Cut_OverLine.m etc.
"""
import numpy as np
from .pointcloud_utils import remove_duplicates, cluster_points_dbscan

def split_overline_4_mid1(cross_line, grid_width):
    """
    Split overline for middle extraction
    Translated from SplitOverline4Mid1.m
    
    Args:
        cross_line: Nx3 power line points
        grid_width: Grid cell size
    Returns:
        Processed line points
    """
    if len(cross_line) == 0:
        return cross_line
    
    # This is a simplified version - the full MATLAB implementation
    # would involve more complex line splitting logic
    
    # Apply clustering to identify main line segments
    if len(cross_line) > 5:
        labels = cluster_points_dbscan(cross_line, eps=grid_width*2, min_samples=3)
        
        # Get largest cluster
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Remove noise
        
        if len(unique_labels) > 0:
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            largest_label = unique_labels[np.argmax(cluster_sizes)]
            main_line = cross_line[labels == largest_label]
            
            # Remove points that are too close to each other
            processed_line = remove_duplicates(main_line, tolerance=grid_width/2)
            return processed_line
    
    return cross_line

def cut_over_line(ql, grid_width, fit_line, mid_yt):
    """
    Cut overline into two parts
    Translated from Cut_OverLine.m
    
    Args:
        ql: Line points to cut
        grid_width: Grid cell size  
        fit_line: Fitted line parameters [slope, intercept]
        mid_yt: Middle Y coordinate of tower
    Returns:
        tuple: (qlc1, qlc2) - two parts of cut line
    """
    if len(ql) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # Use the fitted line to separate points
    # Line equation: y = slope * x + intercept
    slope, intercept = fit_line
    
    # Calculate which side of the line each point is on
    y_line = slope * ql[:, 0] + intercept
    
    # Points above and below the fitted line
    above_line = ql[ql[:, 1] >= y_line]
    below_line = ql[ql[:, 1] < y_line]
    
    # Determine which part is the insulator based on distance to tower center
    if len(above_line) > 0 and len(below_line) > 0:
        dist_above = np.mean(np.abs(above_line[:, 1] - mid_yt))
        dist_below = np.mean(np.abs(below_line[:, 1] - mid_yt))
        
        # The part closer to tower is likely insulator
        if dist_above < dist_below:
            qlc1 = above_line  # Insulator part
            qlc2 = below_line  # Jumper part
        else:
            qlc1 = below_line  # Insulator part  
            qlc2 = above_line  # Jumper part
    elif len(above_line) > 0:
        qlc1 = above_line
        qlc2 = np.empty((0, 3))
    elif len(below_line) > 0:
        qlc1 = below_line
        qlc2 = np.empty((0, 3))
    else:
        qlc1 = np.empty((0, 3))
        qlc2 = np.empty((0, 3))
    
    return qlc1, qlc2

def remove_duplicate_points(cross_line1, cross_line_pts1, tolerance):
    """
    Remove duplicate points between two point clouds
    Translated from removeDuplicatePoints function
    
    Args:
        cross_line1: First point cloud
        cross_line_pts1: Second point cloud  
        tolerance: Distance tolerance for duplicates
    Returns:
        Points from cross_line1 that are not duplicated in cross_line_pts1
    """
    if len(cross_line1) == 0 or len(cross_line_pts1) == 0:
        return cross_line1
    
    # Find points in cross_line1 that are far from all points in cross_line_pts1
    unique_points = []
    
    for point1 in cross_line1:
        # Calculate distances to all points in second cloud
        distances = np.linalg.norm(cross_line_pts1 - point1, axis=1)
        min_distance = np.min(distances)
        
        # Keep point if it's far enough from all points in second cloud
        if min_distance > tolerance:
            unique_points.append(point1)
    
    if unique_points:
        return np.array(unique_points)
    else:
        return np.empty((0, 3))

def find_mutations(bin_array, cut_threshold):
    """
    Find mutation positions in binary array
    Translated from findMuta function in InsExtractType4.m
    
    Args:
        bin_array: 1D binary array
        cut_threshold: Threshold for clipping
    Returns:
        tuple: (beg_pos, end_pos) - start and end positions of mutations
    """
    if len(bin_array) == 0:
        return [], []
    
    # Clipping value
    bin_c = bin_array - cut_threshold
    bin_c[bin_c < 0] = 0
    
    # Translate one unit
    d_bin_c = np.zeros_like(bin_c)
    if len(bin_c) > 1:
        d_bin_c[1:] = bin_c[:-1]
    
    # Dislocation addition
    df = d_bin_c + bin_c
    
    # Identify starting and ending positions
    beg_pos = np.where((bin_c == df) & (bin_c != 0))[0]
    end_pos = np.where((d_bin_c == df) & (d_bin_c != 0))[0]
    
    if len(beg_pos) == 0 or len(end_pos) == 0:
        return [], []
    
    # Calculate interval between each mutation interval
    if len(beg_pos) > 1 and len(end_pos) > 1:
        min_len = min(len(beg_pos) - 1, len(end_pos) - 1)
        if min_len > 0:
            gap1 = beg_pos[1:min_len+1] - end_pos[:min_len]
            
            # Merge intervals that are too small
            merge_indices = np.where(gap1 <= 2)[0]
            beg_pos = np.delete(beg_pos, merge_indices + 1)
            end_pos = np.delete(end_pos, merge_indices)
    
    # Adjust positions (MATLAB to Python indexing)
    beg_pos = beg_pos - 1
    end_pos = end_pos - 1
    
    # Delete mutations that are too small and close to both ends
    del_gap = np.where(
        (beg_pos < len(bin_c) / 4) | 
        (end_pos > len(bin_c) * 3 / 4)
    )[0]
    
    beg_pos = np.delete(beg_pos, del_gap)
    end_pos = np.delete(end_pos, del_gap)
    
    return beg_pos.tolist(), end_pos.tolist()

def extra_ins_with_line_h1(line_points, tower_points, fit_line, is_up_cross, grid_width):
    """
    Extract insulator with horizontal line method 1
    Translated from ExtraInsWithLineH1.m
    
    Args:
        line_points: Power line points
        tower_points: Tower points
        fit_line: Fitted line parameters [slope, intercept]
        is_up_cross: Whether this is upper cross-arm
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    ins_pts = np.empty((0, 3))
    ins_len = 0.0
    
    if len(line_points) == 0 or len(tower_points) == 0:
        return ins_pts, ins_len
    
    # Cut power line points and tower points based on fitted line
    slope, intercept = fit_line
    
    if is_up_cross:
        # For upper cross-arm, use Y-Z plane separation
        yd = np.min(line_points[:, 1]) + (np.max(line_points[:, 1]) - np.min(line_points[:, 1])) / 2
        zd = np.min(line_points[:, 2]) + (np.max(line_points[:, 2]) - np.min(line_points[:, 2])) / 2
        
        # Line equation: slope * Y - Z + intercept = 0
        line_side_test = slope * yd - zd + intercept
        
        if line_side_test < 0:  # Power line is to the left of fitted line
            lp = line_points[slope * line_points[:, 1] - line_points[:, 2] + intercept < 0]
            tp = tower_points[slope * tower_points[:, 1] - tower_points[:, 2] + intercept < 0]
        else:
            lp = line_points[slope * line_points[:, 1] - line_points[:, 2] + intercept > 0]
            tp = tower_points[slope * tower_points[:, 1] - tower_points[:, 2] + intercept > 0]
    else:
        # For lower cross-arm, use X-Y plane separation
        xd = np.min(line_points[:, 0]) + (np.max(line_points[:, 0]) - np.min(line_points[:, 0])) / 2
        yd = np.min(line_points[:, 1]) + (np.max(line_points[:, 1]) - np.min(line_points[:, 1])) / 2
        
        # Line equation: slope * X - Y + intercept = 0
        line_side_test = slope * xd - yd + intercept
        
        if line_side_test < 0:  # Power line is to the left of fitted line
            lp = line_points[slope * line_points[:, 0] - line_points[:, 1] + intercept < 0]
            tp = tower_points[slope * tower_points[:, 0] - tower_points[:, 1] + intercept < 0]
        else:
            lp = line_points[slope * line_points[:, 0] - line_points[:, 1] + intercept > 0]
            tp = tower_points[slope * tower_points[:, 0] - tower_points[:, 1] + intercept > 0]
    
    if len(lp) > 0 and len(tp) > 0:
        # Extract insulator using horizontal line method
        ins_pts_r32, theta1, theta2 = ins_in_h_line3(tp, lp, grid_width)
        
        if len(ins_pts_r32) > 0:
            # Extract tower points based on insulator results
            from .math_utils import rotz, roty
            
            tp_r32 = tp @ rotz(np.degrees(theta1)).T @ roty(np.degrees(theta2)).T
            
            # Filter tower points within insulator bounding box
            ins_pts2_r32 = tp_r32[
                (tp_r32[:, 1] < np.max(ins_pts_r32[:, 1])) &
                (tp_r32[:, 1] > np.min(ins_pts_r32[:, 1])) &
                (tp_r32[:, 2] < np.max(ins_pts_r32[:, 2])) &
                (tp_r32[:, 2] > np.min(ins_pts_r32[:, 2]))
            ]
            
            # Combine results
            if len(ins_pts2_r32) > 0:
                ins_pts_combined = np.vstack([ins_pts_r32, ins_pts2_r32])
            else:
                ins_pts_combined = ins_pts_r32
            
            # Calculate length
            ins_len = np.max(ins_pts_combined[:, 0]) - np.min(ins_pts_combined[:, 0])
            
            # Rotate back to original coordinates
            ins_pts = ins_pts_combined @ roty(-np.degrees(theta2)).T @ rotz(-np.degrees(theta1)).T
    
    return ins_pts, ins_len

def ins_in_h_line3(tower_points, line_points, grid_width):
    """
    Extract insulator in horizontal line method 3
    Translated from InsInHLine3.m
    
    Args:
        tower_points: Tower points
        line_points: Line points
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, theta1, theta2) - extracted points and rotation angles
    """
    from .math_utils import rotate_with_axle, rotz, roty
    from .projection_utils import bin_projection
    from .histogram_utils import drow_z_barh, fill_bin
    
    ins_pts = np.empty((0, 3))
    theta1 = theta2 = 0.0
    
    if len(line_points) == 0:
        return ins_pts, theta1, theta2
    
    # Dual-axis rotation
    lr3, theta1 = rotate_with_axle(line_points, axis=3)
    lr32, theta2 = rotate_with_axle(lr3, axis=2)
    
    # Rotate tower points with same angles
    tr32 = tower_points @ rotz(np.degrees(theta1)).T @ roty(np.degrees(theta2)).T
    
    # Calculate tower X midpoint
    tx_mid = np.min(tr32[:, 0]) + (np.max(tr32[:, 0]) - np.min(tr32[:, 0])) / 2
    
    # XY plane extraction
    bin_xy, _ = bin_projection(lr32, grid_width, axis_x=0, axis_y=1)
    xy_wid = drow_z_barh(bin_xy, direction=-2, return_type='wid')
    
    # Fill missing bins and find maximum
    fxy_wid = fill_bin(xy_wid)
    max_ind = np.argmax(xy_wid)
    mid_pos = len(fxy_wid) // 2
    
    is_cut = False
    
    # Determine insulator position based on tower position
    if abs(tx_mid - np.min(lr32[:, 0])) > abs(tx_mid - np.max(lr32[:, 0])):
        # Insulator on top
        # Remove noise
        max_v = np.max(fxy_wid[:mid_pos])
        noise_mask = fxy_wid[:mid_pos] == max_v
        if np.sum(noise_mask) <= 3:
            fxy_wid[noise_mask] = max_v - 1
        
        # Calculate threshold
        thre = np.max(fxy_wid[:mid_pos]) + 1
        valid_indices = np.where(fxy_wid[mid_pos:] > thre)[0]
        
        if len(valid_indices) > 0:
            cut_pos = mid_pos + valid_indices[0]
            x_threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
            ins_pts = lr32[lr32[:, 0] > x_threshold]
            is_cut = True
    else:
        # Insulator below
        # Remove noise
        max_v = np.max(fxy_wid[mid_pos:])
        noise_mask = fxy_wid[mid_pos:] == max_v
        if np.sum(noise_mask) <= 3:
            fxy_wid[mid_pos + np.where(noise_mask)[0]] = max_v - 1
        
        thre = np.max(fxy_wid[mid_pos:]) + 1
        valid_indices = np.where(fxy_wid[:mid_pos] > thre)[0]
        
        if len(valid_indices) > 0:
            cut_pos = valid_indices[-1]
            x_threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
            ins_pts = lr32[lr32[:, 0] < x_threshold]
            is_cut = True
    
    # If XY plane cannot be divided, use XZ plane
    if not is_cut:
        bin_xz, _ = bin_projection(lr32, grid_width, axis_x=0, axis_y=2)
        xz_wid = drow_z_barh(bin_xz, direction=-2, return_type='wid')
        fxz_wid = fill_bin(xz_wid)
        
        thre = 4
        if abs(tx_mid - np.min(lr32[:, 0])) > abs(tx_mid - np.max(lr32[:, 0])):
            # Insulator on top
            valid_indices = np.where(fxz_wid > thre)[0]
            if len(valid_indices) > 0:
                cut_pos = valid_indices[-1]
                x_threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
                ins_pts = lr32[lr32[:, 0] > x_threshold]
        else:
            # Insulator below
            valid_indices = np.where(fxz_wid > thre)[0]
            if len(valid_indices) > 0:
                cut_pos = valid_indices[0]
                x_threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
                ins_pts = lr32[lr32[:, 0] < x_threshold]
    
    return ins_pts, theta1, theta2