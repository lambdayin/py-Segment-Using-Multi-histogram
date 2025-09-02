"""
Histogram analysis utilities - exact 1:1 translation from MATLAB
Translated from drowZbarh.m and related functions
"""
import numpy as np

def drow_z_barh(binary_image, direction=1, return_type='sum'):
    """
    Comprehensive histogram analysis function
    Exact 1:1 translation from MATLAB drowZbarh.m
    
    Args:
        binary_image: 2D binary image (numpy array)
        direction: Direction for analysis
                  1: Left to right (default)
                  -1: Right to left
                  2: Bottom to top
                  -2: Top to bottom
        return_type: Type of analysis to return
                    'sum': Pixel density per row/column
                    'Dsum': Absolute differences of density
                    'wid': Width (last - first + 1) per row
                    'Dwid': Absolute differences of width
                    'epy': Empty pixels between first and last
                    'fir': First non-zero pixel position per row
                    'end': Last non-zero pixel position per row
                    'Dfir': Absolute differences of first positions
                    'Dend': Absolute differences of last positions
    Returns:
        Corresponding analysis array
    """
    # Handle empty input
    if binary_image.size == 0:
        return np.array([])
    
    # Apply direction transformations (MATLAB indexing: 1-based, we use 0-based)
    img = binary_image.copy()
    
    if direction == -1:  # Right to left
        img = np.fliplr(img)
    elif direction == 2:  # Bottom to top
        img = np.flipud(img.T)
    elif direction == -2:  # Top to bottom
        img = np.fliplr(img.T)
    # direction == 1: Left to right (no transformation needed)
    
    # Get dimensions
    height, width = img.shape
    
    # Initialize result arrays
    img_s = np.zeros(height)    # Sum (density)
    img_f = np.zeros(height)    # First non-zero position
    img_e = np.zeros(height)    # Last non-zero position  
    img_w = np.zeros(height)    # Width
    img_ey = np.zeros(height)   # Empty pixels between first and last
    
    # Process each row
    for i in range(height):
        row = img[i, :]
        
        # Calculate sum (density)
        img_s[i] = np.sum(row)
        
        # Find non-zero positions
        non_zero_indices = np.where(row > 0)[0]
        
        if len(non_zero_indices) > 0:
            # Convert to 1-based indexing to match MATLAB
            img_f[i] = non_zero_indices[0] + 1
            img_e[i] = non_zero_indices[-1] + 1
            
            # Calculate width
            img_w[i] = img_e[i] - img_f[i] + 1
            
            # Count empty pixels between first and last (inclusive)
            first_idx = int(img_f[i] - 1)  # Convert back to 0-based
            last_idx = int(img_e[i] - 1)   # Convert back to 0-based
            
            if first_idx <= last_idx:
                segment = row[first_idx:last_idx + 1]
                img_ey[i] = np.sum(segment == 0)
        else:
            # No non-zero pixels in this row
            img_f[i] = 0
            img_e[i] = 0
            img_w[i] = 0
            img_ey[i] = 0
    
    # Return based on requested type
    if return_type == 'sum':
        return img_s
    elif return_type == 'Dsum':
        return np.abs(np.diff(img_s))
    elif return_type == 'wid':
        return img_w
    elif return_type == 'Dwid':
        return np.abs(np.diff(img_w))
    elif return_type == 'epy':
        return img_ey
    elif return_type == 'fir':
        return img_f
    elif return_type == 'end':
        return img_e
    elif return_type == 'Dfir':
        return np.abs(np.diff(img_f))
    elif return_type == 'Dend':
        return np.abs(np.diff(img_e))
    else:
        # Default to sum if invalid type
        return img_s

def get_cut_pos1(binary_image):
    """
    Hole detection algorithm for finding cut positions
    Exact 1:1 translation from MATLAB getCutpos1.m
    
    Args:
        binary_image: 2D binary image
    Returns:
        Cut position index (1-based to match MATLAB, or 0 if not found)
    """
    if binary_image.size == 0:
        return 0
    
    # Get empty pixel count per row - exactly matching MATLAB logic
    epy = drow_z_barh(binary_image, direction=1, return_type='epy')
    
    # Create binary mask for empty/non-empty rows (MATLAB logic)
    epy0 = epy.copy()
    epy0[epy0 != 0] = -1  # Mark non-empty as -1
    epy0[epy0 == 0] = 1   # Mark empty as 1
    epy0[epy0 == -1] = 0  # Convert non-empty back to 0
    
    # Find hole interval boundaries
    if len(epy0) < 2:
        return 0
        
    d_epy0 = np.diff(epy0)
    beg_indices = np.where(d_epy0 == 1)[0]  # Start of hole intervals
    
    if len(beg_indices) == 0:
        return 0
    
    # Get density changes
    ds = drow_z_barh(binary_image, direction=1, return_type='Dsum')
    
    if len(ds) == 0:
        return 0
    
    # Find maximum density change position
    dwid_max_ind = np.argmax(ds)
    
    # Find first hole interval that starts after maximum density change
    valid_holes = beg_indices - dwid_max_ind >= 0
    
    if np.any(valid_holes):
        cut_pos_idx = np.where(valid_holes)[0][0]  # First valid hole
        # Convert to 1-based indexing to match MATLAB
        cut_pos = beg_indices[cut_pos_idx] + 1
        return int(cut_pos)
    else:
        return 0

def fill_bin(histogram, fill_value=None):
    """
    Fill missing values in histogram
    Translated from fillBin.m
    
    Args:
        histogram: 1D array with potential zero/missing values
        fill_value: Value to use for filling (if None, uses neighbor average)
    Returns:
        Filled histogram
    """
    if len(histogram) == 0:
        return histogram
    
    filled = histogram.copy().astype(float)
    zero_indices = np.where(filled == 0)[0]
    
    for idx in zero_indices:
        if fill_value is not None:
            filled[idx] = fill_value
        else:
            # Find nearest non-zero neighbors
            left_val = 0
            right_val = 0
            
            # Look left for non-zero value
            for left_idx in range(idx - 1, -1, -1):
                if filled[left_idx] != 0:
                    left_val = filled[left_idx]
                    break
            
            # Look right for non-zero value
            for right_idx in range(idx + 1, len(filled)):
                if filled[right_idx] != 0:
                    right_val = filled[right_idx]
                    break
            
            # Use average of neighbors, or single neighbor if only one exists
            if left_val != 0 and right_val != 0:
                filled[idx] = (left_val + right_val) / 2
            elif left_val != 0:
                filled[idx] = left_val
            elif right_val != 0:
                filled[idx] = right_val
            # If no neighbors found, leave as 0
    
    return filled

def detect_muta_in_bin(binary_image, direction=1):
    """
    Detect mutations/changes in binary image
    Translated from DetectMutaInBin.m
    
    Args:
        binary_image: 2D binary image
        direction: Analysis direction (1=horizontal, 2=vertical)
    Returns:
        Array indicating mutation positions
    """
    if binary_image.size == 0:
        return np.array([])
    
    if direction == 2:
        # Analyze columns instead of rows
        binary_image = binary_image.T
    
    height, width = binary_image.shape
    mutations = np.zeros(height)
    
    for i in range(height):
        row = binary_image[i, :]
        
        # Count transitions from 0 to 1 and 1 to 0
        transitions = np.sum(np.abs(np.diff(row.astype(int))))
        mutations[i] = transitions
    
    return mutations

def create_histogram_features(binary_image):
    """
    Create comprehensive histogram feature set
    Based on the 5 histogram types: HD/HV/HW/VW/VV
    
    Args:
        binary_image: 2D binary image
    Returns:
        Dictionary of histogram features
    """
    features = {}
    
    # HD - Height/Density histogram (vertical density)
    features['HD'] = drow_z_barh(binary_image, direction=1, return_type='sum')
    
    # HV - Height/Vertical histogram (row-wise analysis)
    features['HV'] = drow_z_barh(binary_image, direction=2, return_type='sum')
    
    # HW - Width histogram
    features['HW'] = drow_z_barh(binary_image, direction=1, return_type='wid')
    
    # VW - Volume/Width relationship (empty space analysis)
    features['VW'] = drow_z_barh(binary_image, direction=1, return_type='epy')
    
    # VV - Vertical variance (changes in density)
    features['VV'] = drow_z_barh(binary_image, direction=1, return_type='Dsum')
    
    return features

def merge_location(locations, target_cross_num):
    """
    Merge locations to match target number of cross arms
    Exact translation from MATLAB mergeLoc.m
    
    Args:
        locations: List of [start, end] location pairs
        target_cross_num: Target number of cross arms
    Returns:
        Merged locations list
    """
    if len(locations) <= target_cross_num:
        return locations
    
    # Convert to numpy array for easier manipulation
    loc_array = np.array(locations)
    
    while loc_array.shape[0] > target_cross_num:
        # Calculate gaps between adjacent cross arms
        gaps = loc_array[:-1, 1] - loc_array[1:, 0]  # end[i] - start[i+1]
        
        # Find minimum gap
        min_gap_idx = np.argmin(gaps)
        
        # Merge the cross arms with minimum gap
        loc_array[min_gap_idx, 1] = loc_array[min_gap_idx + 1, 1]  # Extend end
        
        # Remove the merged cross arm
        loc_array = np.delete(loc_array, min_gap_idx + 1, axis=0)
    
    return loc_array.tolist()

def detect_muta_in_bin(bin_array):
    """
    Detect mutation patterns in binary array
    Complete 1:1 translation from MATLAB DetectMutaInBin.m
    
    Args:
        bin_array: 1D array, usually histogram differences
    Returns:
        Detected mutation pattern array
    """
    from ..utils.missing_algorithms import detect_muta_in_bin as detect_func
    return detect_func(bin_array)