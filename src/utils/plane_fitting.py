"""
Plane fitting utilities - exact 1:1 translation from MATLAB
Translated from fitPlane1.m, fitPlane2.m and related functions
"""
import numpy as np
from .math_utils import rotate_with_axle, rotz, roty
from .projection_utils import sine_projection

def fit_plane_1(points, tower_type):
    """
    Fit plane method 1 for boundary line extraction
    Exact 1:1 translation from MATLAB fitPlane1.m
    
    Args:
        points: Nx3 point cloud
        tower_type: Tower type (affects cutting behavior)
    Returns:
        tuple: (fleft, fright) - left and right line parameters [slope, intercept]
    """
    # Determine if should cut middle based on tower type
    if tower_type in [3, 5]:
        is_cut_mid = True
    elif tower_type in [1, 4]:
        is_cut_mid = False
    else:
        raise ValueError(f"Undefined tower type: {tower_type}")
    
    # Dual-axis rotation for alignment
    points_rz, theta3 = rotate_with_axle(points, axis=3)
    points_rzy, theta2 = rotate_with_axle(points_rz, axis=2)
    
    # Reorder coordinates: [2,1,3] -> [Y,X,Z]
    points_rzy = points_rzy[:, [1, 0, 2]]
    
    # Extract boundary points
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_rzy[:, :2])
        boundary_indices = hull.vertices
        points_b = points_rzy[boundary_indices]
    except:
        # Fallback if ConvexHull fails
        points_b = points_rzy
    
    # Translate to coordinate origin
    pts = np.zeros_like(points_b)
    pts[:, 0] = points_b[:, 0] - np.min(points_b[:, 0])
    pts[:, 1] = points_b[:, 1] - np.min(points_b[:, 1])
    pts[:, 2] = points_b[:, 2] - np.min(points_b[:, 2])
    
    # Fit left and right straight lines separately
    sam_gap = 0.1
    mid_x = np.min(pts[:, 0]) + (np.max(pts[:, 0]) - np.min(pts[:, 0])) / 2
    
    # Left fitting points
    pts_l_ind = np.where(pts[:, 0] < mid_x)[0]
    pts_left_r = extra_fit_pts(pts_l_ind, pts, points_b, sam_gap, is_cut_mid)
    
    # Transform back to original coordinates
    pts_left = pts_left_r[:, [1, 0, 2]] @ roty(-np.degrees(theta2)).T @ rotz(-np.degrees(theta3)).T
    fleft = ransac_fitline(pts_left[:, :2], 10000, sam_gap)
    
    # Right fitting points  
    pts_r_ind = np.where(pts[:, 0] >= mid_x)[0]
    pts_right_r = extra_fit_pts(pts_r_ind, pts, points_b, sam_gap, is_cut_mid)
    
    # Transform back to original coordinates
    pts_right = pts_right_r[:, [1, 0, 2]] @ roty(-np.degrees(theta2)).T @ rotz(-np.degrees(theta3)).T
    fright = ransac_fitline(pts_right[:, :2], 10000, sam_gap)
    
    # Set the line with smaller y coordinate as left fitting line
    if np.mean(pts_left[:, 1]) > np.mean(pts_right[:, 1]):
        fleft, fright = fright, fleft
    
    return fleft, fright

def fit_plane_2(points):
    """
    Fit plane method 2 for cross-arm boundary
    Exact 1:1 translation from MATLAB fitPlane2.m
    
    Args:
        points: Nx3 point cloud
    Returns:
        tuple: (fleft, fright) - left and right line parameters [slope, intercept]
    """
    # Extract cross-arm boundary points (Y-Z plane)
    plane_pts = points[:, [1, 2]]  # Y, Z coordinates
    
    # Get boundary points
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(plane_pts)
        boundary_indices = hull.vertices
        pts = plane_pts[boundary_indices]
    except:
        pts = plane_pts
    
    # Delete upper and lower boundary points
    sam_gap = 0.2
    pts_d, _ = sine_projection(pts, axis=1, grid_width=sam_gap)  # Project to Y axis
    
    half_len = len(pts_d) // 2
    
    # Find maximum in first half
    cut_ind1 = np.argmax(pts_d[:half_len])
    # Find maximum in second half  
    cut_ind2 = np.argmax(pts_d[half_len:]) + half_len
    
    cut_pos1 = np.min(pts[:, 1]) + cut_ind1 * sam_gap
    cut_pos2 = np.min(pts[:, 1]) + cut_ind2 * sam_gap
    
    # Filter out upper and lower boundary points
    pts_c = pts[(pts[:, 1] >= cut_pos1) & (pts[:, 1] <= cut_pos2)]
    
    # Fit left and right straight lines separately
    mid_x = np.min(pts_c[:, 0]) + (np.max(pts_c[:, 0]) - np.min(pts_c[:, 0])) / 2
    
    # Left fitting line
    left_pts = pts_c[pts_c[:, 0] < mid_x]
    fleft = ransac_fitline(left_pts, 10000, sam_gap)
    
    # Right fitting line
    right_pts = pts_c[pts_c[:, 0] > mid_x]
    fright = ransac_fitline(right_pts, 10000, sam_gap)
    
    return fleft, fright

def extra_fit_pts(half_ind, pts, points, sam_gap, is_cut_mid):
    """
    Extract points to be fitted
    Translated from ExtraFitPts function in fitPlane1.m
    
    Args:
        half_ind: Indices of half points
        pts: Processed points
        points: Original boundary points
        sam_gap: Sampling gap
        is_cut_mid: Whether to cut middle section
    Returns:
        Points to be fitted
    """
    pts_half = pts[half_ind]
    
    # Rotate point cloud once more
    pts_r = rota_pts(pts_half)
    pts_r_move = np.zeros_like(pts_r)
    pts_r_move[:, 0] = pts_r[:, 0] - np.min(pts_r[:, 0])
    pts_r_move[:, 1] = pts_r[:, 1] - np.min(pts_r[:, 1])
    pts_r_move[:, 2] = pts_r[:, 2] - np.min(pts_r[:, 2])
    
    # Cut off middle part if required
    out_pts_ind = np.arange(len(pts_r_move))
    if is_cut_mid:
        y_len = np.max(pts_r_move[:, 1]) - np.min(pts_r_move[:, 1])
        pts_r_move_ori = pts_r_move.copy()
        
        out_pts_ind = np.where(
            (pts_r_move[:, 1] <= np.min(pts_r_move[:, 1]) + y_len / 4) |
            (pts_r_move[:, 1] >= np.max(pts_r_move[:, 1]) - y_len / 4)
        )[0]
        pts_r_move = pts_r_move[out_pts_ind]
    
    # Project to X-axis and create width histogram
    wid_histo, pts_in_w_ind = sine_projection(pts_r_move, axis=0, grid_width=sam_gap)
    
    # Filter fitting points based on projection density
    max_ind = np.argmax(wid_histo)
    s_range = 1  # Take histograms of certain height as fitting points
    
    # Extract fitting points
    tep_ind = np.arange(max(0, max_ind - s_range), min(len(wid_histo), max_ind + s_range + 1))
    len_dgde_ind = []
    for i in tep_ind:
        len_dgde_ind.extend(np.where(pts_in_w_ind == i)[0])
    
    if is_cut_mid:
        dgde_ind = half_ind[out_pts_ind[len_dgde_ind]]
    else:
        dgde_ind = half_ind[len_dgde_ind]
    
    half_pts = points[dgde_ind]
    
    # Check if contains both upper and lower points, expand search if needed
    y_mid = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
    up_pts_len = np.sum(half_pts[:, 1] > y_mid)
    dw_pts_len = np.sum(half_pts[:, 1] < y_mid)
    is_ok = up_pts_len > 0 and dw_pts_len > 0
    
    # Expand scope if needed
    while not is_ok and s_range < len(wid_histo) // 3:
        s_range += 1
        
        # Expand to left
        tep_ind_l = np.arange(max(0, max_ind - s_range), max_ind + 1)
        len_dgde_ind_l = []
        for i in tep_ind_l:
            len_dgde_ind_l.extend(np.where(pts_in_w_ind == i)[0])
        
        if is_cut_mid:
            dgde_ind_l = half_ind[out_pts_ind[len_dgde_ind_l]]
        else:
            dgde_ind_l = half_ind[len_dgde_ind_l]
        
        half_pts_l = points[dgde_ind_l]
        up_pts_len_l = np.sum(half_pts_l[:, 1] > y_mid)
        dw_pts_len_l = np.sum(half_pts_l[:, 1] < y_mid)
        
        if up_pts_len_l > 1 and dw_pts_len_l > 1:
            is_ok = True
            half_pts = half_pts_l
            continue
        
        # Expand to right
        tep_ind_r = np.arange(max_ind, min(len(wid_histo), max_ind + s_range + 1))
        len_dgde_ind_r = []
        for i in tep_ind_r:
            len_dgde_ind_r.extend(np.where(pts_in_w_ind == i)[0])
        
        if is_cut_mid:
            dgde_ind_r = half_ind[out_pts_ind[len_dgde_ind_r]]
        else:
            dgde_ind_r = half_ind[len_dgde_ind_r]
        
        half_pts_r = points[dgde_ind_r]
        up_pts_len_r = np.sum(half_pts_r[:, 1] > y_mid)
        dw_pts_len_r = np.sum(half_pts_r[:, 1] < y_mid)
        
        if up_pts_len_r > 1 and dw_pts_len_r > 1:
            if is_ok:  # Can expand both sides, choose the one with more points
                if len(half_pts_r) >= len(half_pts_l):
                    half_pts = half_pts_r
                # else keep half_pts_l
            else:  # Can only expand to right
                is_ok = True
                half_pts = half_pts_r
    
    return half_pts

def rota_pts(pts):
    """
    Rotate point cloud function
    Translated from RotaPts function in fitPlane1.m
    
    Args:
        pts: Input point cloud
    Returns:
        tuple: (rotated_points, angle)
    """
    # Calculate direction vector using 2D points
    pts_2d = pts[:, :2]
    
    # Downsample for PCA (simplified)
    if len(pts_2d) > 100:
        indices = np.random.choice(len(pts_2d), 100, replace=False)
        pts_2d_down = pts_2d[indices]
    else:
        pts_2d_down = pts_2d
    
    # Calculate covariance matrix
    center = np.mean(pts_2d_down, axis=0)
    m = pts_2d_down - center
    mm = (m.T @ m) / len(pts_2d_down)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(mm)
    # Get principal direction (largest eigenvalue)
    principal_idx = np.argmax(eigenvalues)
    dire_vector = eigenvectors[:, principal_idx]
    
    # Calculate rotation angle
    angle = np.arccos(np.abs(dire_vector[0]) / np.linalg.norm(dire_vector))
    if dire_vector[0] * dire_vector[1] < 0:  # Clockwise from y to x
        angle = -angle
    
    # Rotate points
    pts_r = pts @ rotz(np.degrees(angle)).T
    
    return pts_r

def ransac_fitline(pts, max_iter, threshold):
    """
    RANSAC straight line fitting
    Translated from RanSaC_Fitline function in fitPlane1.m
    
    Args:
        pts: 2D points for line fitting
        max_iter: Maximum iterations
        threshold: Distance threshold
    Returns:
        Line parameters [slope, intercept]
    """
    if len(pts) < 2:
        return [0, 0]
    
    # Normalize points
    points = np.zeros_like(pts)
    points[:, 0] = pts[:, 0] - np.min(pts[:, 0])
    points[:, 1] = pts[:, 1] - np.min(pts[:, 1])
    
    max_num = 0
    best_line = [0, 0]
    fin_dist = np.zeros(len(points))
    
    for j in range(max_iter):
        mid_y = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
        
        # Randomly sample fitting points from upper and lower parts
        d_pts = points[points[:, 1] < mid_y]  # Lower half
        u_pts = points[points[:, 1] > mid_y]  # Upper half
        
        if len(d_pts) == 0 or len(u_pts) == 0:
            continue
        
        sample_num = min(max(1, len(d_pts) // 3), max(1, len(u_pts) // 3))
        
        if len(d_pts) < sample_num or len(u_pts) < sample_num:
            continue
        
        sample_idx_d = np.random.choice(len(d_pts), sample_num, replace=False)
        sample_idx_u = np.random.choice(len(u_pts), sample_num, replace=False)
        
        sam_pts = np.vstack([d_pts[sample_idx_d], u_pts[sample_idx_u]])
        
        # Preliminary line fitting
        if len(sam_pts) >= 2:
            try:
                f = np.polyfit(sam_pts[:, 0], sam_pts[:, 1], 1)
                
                # Calculate distance from each point to line
                distance = np.abs(f[0] * points[:, 0] - points[:, 1] + f[1]) / np.sqrt(f[0]**2 + 1)
                
                # Count points near the line
                near_sum = np.sum(distance < threshold)
                
                # Combine nearby points to fit line
                if near_sum > sample_num:
                    near_pts = points[distance < threshold]
                    f = np.polyfit(near_pts[:, 0], near_pts[:, 1], 1)
                    distance = np.abs(f[0] * points[:, 0] - points[:, 1] + f[1]) / np.sqrt(f[0]**2 + 1)
                    near_sum = np.sum(distance < threshold)
                
                # Update optimal line
                if near_sum > max_num:
                    fin_dist = distance.copy()
                    max_num = near_sum
                    best_line = f.copy()
            except:
                continue
    
    # Final fitting with best inliers
    if max_num > 0:
        best_pts = pts[fin_dist < threshold]
        if len(best_pts) >= 2:
            try:
                best_line = np.polyfit(best_pts[:, 0], best_pts[:, 1], 1)
                
                # Additional refinement for nearly horizontal lines
                if abs(best_line[0]) < 20:
                    mid_y = np.min(best_pts[:, 1]) + (np.max(best_pts[:, 1]) - np.min(best_pts[:, 1])) / 2
                    d_pts = best_pts[best_pts[:, 1] < mid_y]
                    u_pts = best_pts[best_pts[:, 1] > mid_y]
                    
                    if len(d_pts) > 0 and len(u_pts) > 0:
                        sample_num = min(max(1, len(d_pts) // 3), max(1, len(u_pts) // 3))
                        if len(d_pts) >= sample_num and len(u_pts) >= sample_num:
                            sample_idx_d = np.random.choice(len(d_pts), sample_num, replace=False)
                            sample_idx_u = np.random.choice(len(u_pts), sample_num, replace=False)
                            refined_pts = np.vstack([d_pts[sample_idx_d], u_pts[sample_idx_u]])
                            best_line = np.polyfit(refined_pts[:, 0], refined_pts[:, 1], 1)
            except:
                pass
    
    return best_line