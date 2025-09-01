"""
Point cloud projection utilities
Translated from MATLAB BinProjection and related functions
"""
import numpy as np
from typing import Tuple, List

def bin_projection(points, grid_width, axis_x=0, axis_y=2):
    """
    Project 3D points to 2D binary image with grid structure
    Exact 1:1 translation from MATLAB BinProjection.m
    
    Args:
        points: Nx3 point cloud
        grid_width: Size of each grid cell
        axis_x: Index of axis for X projection (0, 1, or 2)  
        axis_y: Index of axis for Y projection (0, 1, or 2)
    Returns:
        tuple: (binary_image, grid_cells)
            binary_image: 2D binary array (matches MATLAB orientation)
            grid_cells: Grid structure with point indices
    """
    if len(points) == 0:
        return np.array([[False]]), {}
    
    # Extract coordinates for specified axes
    x_coords = points[:, axis_x]  # Width axis
    y_coords = points[:, axis_y]  # Height axis
    
    # Calculate grid dimensions exactly as in MATLAB
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Grid dimensions (matching MATLAB ceil calculation)
    grid_w = int(np.ceil((x_max - x_min) / grid_width))
    grid_h = int(np.ceil((y_max - y_min) / grid_width))
    
    # Ensure minimum size
    grid_w = max(1, grid_w)
    grid_h = max(1, grid_h)
    
    # Initialize binary image and grid structure
    binary_image = np.zeros((grid_h, grid_w), dtype=bool)
    grid_cells = {}
    
    # Process each point (matching MATLAB indexing logic)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Calculate grid indices (MATLAB uses floor + 1 for 1-based indexing)
        if grid_w == 1:
            grid_x = 0
        else:
            grid_x = int(np.floor((x - x_min) / grid_width))
            grid_x = max(0, min(grid_x, grid_w - 1))
            
        if grid_h == 1:
            grid_y = 0  
        else:
            grid_y = int(np.floor((y - y_min) / grid_width))
            grid_y = max(0, min(grid_y, grid_h - 1))
        
        # Mark cell as occupied
        binary_image[grid_y, grid_x] = True
        
        # Store point indices in grid structure (matching MATLAB Grid{i,j})
        key = (grid_y, grid_x)
        if key not in grid_cells:
            grid_cells[key] = []
        grid_cells[key].append(i)
    
    return binary_image, grid_cells

def simple_bin_projection(points, grid_width, axis_x=1, axis_y=2):
    """
    Simple binary projection without grid structure
    Translated from binPro.m
    
    Args:
        points: Nx3 point cloud
        grid_width: Size of each grid cell
        axis_x: Index of axis for X projection
        axis_y: Index of axis for Y projection  
    Returns:
        2D binary image
    """
    if len(points) == 0:
        return np.array([[False]])
    
    # Extract coordinates
    x_coords = points[:, axis_x]
    y_coords = points[:, axis_y]
    
    # Calculate grid dimensions
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Handle case where all points are the same
    if x_max == x_min:
        x_max += grid_width
    if y_max == y_min:
        y_max += grid_width
    
    # Calculate grid size
    grid_width_x = max(1, int(np.ceil((x_max - x_min) / grid_width)))
    grid_height_y = max(1, int(np.ceil((y_max - y_min) / grid_width)))
    
    # Initialize binary image
    binary_image = np.zeros((grid_height_y, grid_width_x), dtype=bool)
    
    # Assign points to grid
    for x, y in zip(x_coords, y_coords):
        grid_x = int((x - x_min) / grid_width)
        grid_y = int((y - y_min) / grid_width)
        
        # Handle edge cases
        grid_x = max(0, min(grid_x, grid_width_x - 1))
        grid_y = max(0, min(grid_y, grid_height_y - 1))
        
        binary_image[grid_y, grid_x] = True
    
    return binary_image

def sine_projection(points, axis=0, grid_width=0.1):
    """
    Sine-based projection for width histogram
    Translated from sinPro.m
    
    Args:
        points: Nx3 point cloud
        axis: Axis to project onto (0, 1, or 2)
        grid_width: Width of each histogram bin
    Returns:
        tuple: (histogram, point_indices)
            histogram: 1D array of point counts per bin
            point_indices: Array indicating which bin each point belongs to
    """
    if len(points) == 0:
        return np.array([]), np.array([])
    
    # Extract coordinates for specified axis
    coords = points[:, axis]
    
    # Calculate bin boundaries
    coord_min = np.min(coords)
    coord_max = np.max(coords)
    
    # Calculate number of bins
    n_bins = int(np.ceil((coord_max - coord_min) / grid_width))
    if n_bins == 0:
        n_bins = 1
    
    # Calculate bin indices for each point
    point_indices = ((coords - coord_min) / grid_width).astype(int)
    point_indices = np.clip(point_indices, 0, n_bins - 1)
    
    # Create histogram
    histogram = np.zeros(n_bins)
    for idx in point_indices:
        histogram[idx] += 1
    
    return histogram, point_indices

def create_depth_histogram(binary_image, direction='vertical'):
    """
    Create depth/density histogram from binary image
    Translated from drowZbarh.m functionality
    
    Args:
        binary_image: 2D binary image
        direction: 'vertical' or 'horizontal'
    Returns:
        1D histogram array
    """
    if binary_image.size == 0:
        return np.array([])
    
    if direction == 'vertical':
        # Sum along horizontal axis to get vertical density
        histogram = np.sum(binary_image, axis=1)
    else:
        # Sum along vertical axis to get horizontal density  
        histogram = np.sum(binary_image, axis=0)
    
    return histogram

def cross_location_detection(binary_image, ratio=3):
    """
    Detect cross-arm locations from binary image histogram
    Translated from CrossLocation.m
    
    Args:
        binary_image: 2D binary image of tower projection
        ratio: Density threshold ratio for cross-arm detection
    Returns:
        List of [start, end] positions for each cross-arm
    """
    if binary_image.size == 0:
        return []
    
    # Calculate vertical density histogram
    density = np.sum(binary_image, axis=1)  # Sum along width (axis=1)
    
    # Take upper half of the tower
    half_len = len(density) // 2
    upper_density = density[len(density) - half_len:]
    
    # Apply threshold based on ratio
    max_density = np.max(upper_density)
    threshold = max_density / ratio
    thresholded_density = upper_density - threshold
    thresholded_density[thresholded_density < 0] = 0
    
    # Create shifted version for dislocation analysis
    shifted_density = np.zeros_like(thresholded_density)
    if len(thresholded_density) > 1:
        shifted_density[1:] = thresholded_density[:-1]
    
    # Dislocation addition
    sum_density = thresholded_density + shifted_density
    
    # Find start and end positions of cross-arms
    begin_indices = []
    end_indices = []
    
    for i in range(len(sum_density)):
        if (thresholded_density[i] == sum_density[i] and 
            sum_density[i] != 0 and i > 0):
            begin_indices.append(i + half_len)
        
        if (shifted_density[i] == sum_density[i] and 
            sum_density[i] != 0):
            end_indices.append(i + half_len)
    
    # Remove noise based on density comparison
    max_original_density = np.max(density) if len(density) > 0 else 0
    filtered_begin = []
    filtered_end = []
    
    for i, (begin, end) in enumerate(zip(begin_indices, end_indices)):
        if i < len(end_indices):
            actual_end = min(end, len(density) - 1)
            if begin < len(density):
                segment_max = np.max(density[begin:actual_end+1]) if begin <= actual_end else density[begin]
                if segment_max * 1.8 >= max_original_density:  # Noise filtering
                    filtered_begin.append(begin)
                    filtered_end.append(actual_end)
    
    # Handle length mismatch
    if len(filtered_begin) != len(filtered_end):
        if len(filtered_end) < len(filtered_begin):
            filtered_end.append(len(density) - 1)
        else:
            filtered_begin = filtered_begin[:len(filtered_end)]
    
    # Create location pairs (flip order to match MATLAB convention)
    locations = []
    for begin, end in zip(reversed(filtered_begin), reversed(filtered_end)):
        locations.append([begin, end])
    
    return locations

def append_location(binary_image, locations):
    """
    Append additional location information
    Translated from appendLoc.m
    
    Args:
        binary_image: 2D binary image
        locations: List of cross-arm locations
    Returns:
        Updated locations list
    """
    # This is a placeholder - the actual implementation would depend
    # on the specific requirements of the appendLoc.m function
    # For now, just return the original locations
    return locations