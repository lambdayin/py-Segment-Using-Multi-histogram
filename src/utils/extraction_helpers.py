"""
Additional extraction helper functions
Translated from MATLAB helper functions
"""
import numpy as np
from .projection_utils import bin_projection
from .histogram_utils import drow_z_barh, fill_bin
from .pointcloud_utils import cluster_points_dbscan
from .math_utils import rotate_with_axle, rotz

def split_overline_4_mid1(line_points, grid_width):
    """
    Split overline into middle section
    Exact 1:1 translation from MATLAB SplitOverline4Mid1.m
    
    Args:
        line_points: Nx3 line point cloud
        grid_width: Grid cell size
    Returns:
        Split line points
    """
    # Import the complete implementation
    from ..utils.missing_algorithms import split_overline_4_mid1_complete
    return split_overline_4_mid1_complete(line_points, grid_width)

def extra_ins_with_line_h1(cross_line, tower_points, fit_line, direction, grid_width):
    """
    Extract horizontal insulator with line method 1
    Translated from ExtraInsWithLineH1.m
    
    Args:
        cross_line: Line points for this cross arm
        tower_points: Tower points
        fit_line: Fitted line parameters [slope, intercept]
        direction: Direction parameter
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    insulator_points = np.empty((0, 3))
    insulator_length = 0.0
    
    if len(cross_line) < 5 or len(tower_points) == 0:
        return insulator_points, insulator_length
    
    # Rotate line points to align with transmission direction
    line_r, theta = rotate_with_axle(cross_line, axis=3)
    
    # Apply same rotation to tower points
    tower_r = tower_points @ rotz(np.degrees(theta)).T
    
    # Project line to find cut position
    bin_xz, _ = bin_projection(line_r, grid_width, axis_x=0, axis_y=2)
    
    # Get width histogram to find insulator boundary
    width_hist = drow_z_barh(bin_xz, direction=1, return_type='wid')
    
    # Find cut position based on width changes
    if len(width_hist) > 0:
        # Look for significant width increases (indicating insulator start)
        width_changes = np.abs(np.diff(width_hist))
        if len(width_changes) > 0:
            max_change_idx = np.argmax(width_changes)
            
            # Convert to Z coordinate
            z_min, z_max = np.min(line_r[:, 2]), np.max(line_r[:, 2])
            z_range = z_max - z_min
            
            if z_range > 0:
                z_cut = z_min + (max_change_idx / len(width_hist)) * z_range
                
                # Extract line points above cut position
                line_ins_mask = line_r[:, 2] >= z_cut
                line_ins_points = line_r[line_ins_mask]
                
                if len(line_ins_points) > 0:
                    # Extract corresponding tower points
                    x_min, x_max = np.min(line_ins_points[:, 0]), np.max(line_ins_points[:, 0])
                    y_min, y_max = np.min(line_ins_points[:, 1]), np.max(line_ins_points[:, 1])
                    z_min_ins = np.min(line_ins_points[:, 2])
                    
                    tower_ins_mask = (
                        (tower_r[:, 0] >= x_min) & 
                        (tower_r[:, 0] <= x_max) &
                        (tower_r[:, 1] >= y_min) & 
                        (tower_r[:, 1] <= y_max) &
                        (tower_r[:, 2] >= z_min_ins)
                    )
                    
                    tower_ins_points = tower_r[tower_ins_mask]
                    
                    # Combine line and tower insulator points
                    if len(tower_ins_points) > 0:
                        all_ins_points = np.vstack([line_ins_points, tower_ins_points])
                    else:
                        all_ins_points = line_ins_points
                    
                    # Reverse rotation to original coordinates
                    insulator_points = all_ins_points @ rotz(-np.degrees(theta)).T
                    
                    # Calculate length
                    if len(insulator_points) > 1:
                        insulator_length = (np.max(insulator_points[:, 2]) - 
                                         np.min(insulator_points[:, 2]))
    
    return insulator_points, insulator_length

def ins_in_h_line3(tower_points, line_points, fit_line, grid_width):
    """
    Extract insulator in horizontal line method 3
    Complete 1:1 translation from MATLAB InsInHLine3.m
    
    Args:
        tower_points: Tower point cloud
        line_points: Line point cloud
        fit_line: Fitted line parameters (unused in this implementation)
        grid_width: Grid cell size
    Returns:
        tuple: (insulator_points, insulator_length)
    """
    # Import the complete implementation
    from ..utils.missing_algorithms import ins_in_h_line3_complete
    
    insulator_points, theta1, theta2 = ins_in_h_line3_complete(
        tower_points, line_points, grid_width
    )
    
    # Calculate length
    if len(insulator_points) > 1:
        insulator_length = np.max(insulator_points[:, 2]) - np.min(insulator_points[:, 2])
    else:
        insulator_length = 0.0
    
    return insulator_points, insulator_length

def append_loc(binary_image, locations):
    """
    Append location information based on binary image analysis
    Translated from appendLoc.m
    
    Args:
        binary_image: 2D binary image
        locations: Existing cross-arm locations
    Returns:
        Extended locations with additional position info
    """
    if len(locations) == 0:
        return locations
    
    # Get density histogram
    density = drow_z_barh(binary_image, direction=1, return_type='sum')
    
    # Calculate density changes
    density_changes = drow_z_barh(binary_image, direction=1, return_type='Dsum')
    
    # Extend each location with additional boundary information
    extended_locations = []
    
    for loc_start, loc_end in locations:
        # Find density peaks around this location
        search_start = max(0, loc_start - 5)
        search_end = min(len(density), loc_end + 5)
        
        if search_start < search_end:
            local_density = density[search_start:search_end]
            
            # Find local maxima
            if len(local_density) > 2:
                # Simple peak detection
                peaks = []
                for i in range(1, len(local_density) - 1):
                    if (local_density[i] > local_density[i-1] and 
                        local_density[i] > local_density[i+1]):
                        peaks.append(search_start + i)
                
                if peaks:
                    # Use the most significant peak
                    peak_values = [density[p] for p in peaks]
                    main_peak = peaks[np.argmax(peak_values)]
                    
                    # Extend location based on peak position
                    extended_start = min(loc_start, main_peak - 2)
                    extended_end = max(loc_end, main_peak + 2)
                    
                    extended_locations.append([extended_start, extended_end])
                else:
                    extended_locations.append([loc_start, loc_end])
            else:
                extended_locations.append([loc_start, loc_end])
        else:
            extended_locations.append([loc_start, loc_end])
    
    return extended_locations