"""
Tower type classification module
Translated from MATLAB TypeDetect.m and OTowerDetect.m
"""
import numpy as np
from ..core.constants import TowerType
from ..utils.math_utils import rotate_with_axle, rotz, max_label
from ..utils.projection_utils import bin_projection, simple_bin_projection, cross_location_detection, create_depth_histogram
from ..utils.pointcloud_utils import cluster_points_dbscan, extract_largest_cluster
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening

def o_tower_detect(binary_image, cut_location=0.5, auto_threshold=None):
    """
    Detect O-type tower based on cavity area analysis
    Translated from OTowerDetect.m
    
    Args:
        binary_image: 2D binary image of tower projection
        cut_location: Fraction of tower to analyze (from top)
        auto_threshold: Custom threshold for cavity area
    Returns:
        bool: True if O-type tower detected
    """
    if binary_image.size == 0:
        return False
    
    # Only analyze upper portion of tower
    cut_row = int(np.ceil(binary_image.shape[0] * cut_location))
    img_section = binary_image[cut_row:, :]
    
    if img_section.size == 0:
        return False
    
    # Set top row to True to ensure proper boundary detection
    img_section = img_section.copy()
    img_section[0, :] = True
    
    # Find connected components (cavities are False regions surrounded by True)
    # We need to invert the image to find holes
    inverted_img = ~img_section
    
    # Label connected components in inverted image
    labeled_image = label(inverted_img, connectivity=2)
    
    # Get region properties
    regions = regionprops(labeled_image)
    
    # Calculate areas of all regions
    areas = [region.area for region in regions]
    
    if not areas:
        return False
    
    max_area = max(areas)
    
    # Calculate automatic threshold if not provided
    if auto_threshold is None:
        auto_threshold = img_section.shape[0] * img_section.shape[1] * 0.07
    
    # Check if largest cavity meets criteria for O-type tower
    if max_area > auto_threshold and max_area > 300:
        return True
    else:
        return False

def detect_tower_type(tower_points, line_points):
    """
    Main tower type detection function
    Translated from TypeDetect.m
    
    Args:
        tower_points: Nx3 array of tower point cloud
        line_points: Mx3 array of line point cloud
    Returns:
        TowerType enum value
    """
    if len(tower_points) == 0:
        return TowerType.WINE_GLASS  # Default type
    
    # Remove noise points above tower (similar to preprocessing)
    z_values = tower_points[:, 2]
    top_z, top_indices = np.sort(z_values)[-10:], np.argsort(z_values)[-10:]
    
    # Find significant drops in height
    if len(top_z) > 1:
        z_diffs = np.diff(top_z)
        significant_drops = np.where(z_diffs < -1.0)[0]
        
        if len(significant_drops) > 0:
            cut_idx = top_indices[significant_drops[-1] + 1]
            cut_height = tower_points[cut_idx, 2]
            tower_points = tower_points[tower_points[:, 2] <= cut_height]
    
    # Rotate tower to standard orientation using top 3m
    max_z = np.max(tower_points[:, 2])
    top_points = tower_points[tower_points[:, 2] > max_z - 3.0]
    
    if len(top_points) >= 3:
        rotated_tower, theta = rotate_with_axle(top_points, axis=3)
        rotation_matrix = rotz(np.degrees(theta))
        tower_rotated = tower_points @ rotation_matrix.T
        
        # Rotate line points with same transformation
        if len(line_points) > 0:
            line_rotated = line_points @ rotation_matrix.T
        else:
            line_rotated = line_points
    else:
        tower_rotated = tower_points
        line_rotated = line_points
    
    # Create binary projections for analysis
    grid_width_coarse = 0.2  # For cavity detection
    grid_width_fine = 0.1    # For cross-arm detection
    
    # YZ projection for cavity detection (axis 1=Y, axis 2=Z)
    bin_image_coarse, _ = bin_projection(tower_rotated, grid_width_coarse, axis_x=1, axis_y=2)
    
    # YZ projection for cross-arm detection
    bin_image_fine, _ = bin_projection(tower_rotated, grid_width_fine, axis_x=1, axis_y=2)
    
    # Detect cross-arm positions
    cross_locations = cross_location_detection(bin_image_fine, ratio=4)
    num_cross_arms = len(cross_locations)
    
    # Tower type classification logic
    if o_tower_detect(bin_image_coarse, 0.5):  # Upper half cavity detection
        if num_cross_arms == 1:
            return TowerType.WINE_GLASS  # O-shaped tower, 1 cross arm
        else:
            return TowerType.CAT_HEAD    # O-shaped tower, multiple cross arms
    
    elif num_cross_arms == 1 and o_tower_detect(bin_image_coarse, 1/3):
        return TowerType.WINE_GLASS  # Expanded range detection for wine glass
    
    elif num_cross_arms == 2:
        if o_tower_detect(bin_image_coarse, 0.5, 500):  # Cat head with missing data
            return TowerType.CAT_HEAD
        else:
            return TowerType.TENSION_DRY  # Dry font tower
    
    elif num_cross_arms == 1:
        return TowerType.SINGLE_CROSS
    
    else:  # Multiple cross arms - drum-shaped tower
        # Further classification between tension and DC types
        if len(line_rotated) > 0:
            # Cluster power lines to find main group
            labels = cluster_points_dbscan(line_rotated, eps=1.0, min_samples=1)
            
            if len(labels) > 0:
                main_label = max_label(labels)
                main_line_points = line_rotated[labels == main_label]
                
                # Create XZ projection to analyze vertical gaps
                bin_image_xz = simple_bin_projection(main_line_points, grid_width_fine, axis_x=1, axis_y=2)
                
                # Analyze histogram for gap detection
                histogram = create_depth_histogram(bin_image_xz, direction='vertical')
                
                # Calculate mean gap size (simplified version of drowZbarh analysis)
                if len(histogram) > 0:
                    mean_gap = np.mean(histogram)
                    if mean_gap > 4:  # Threshold from MATLAB (4 pixels ~ 1m for 0.1m grid)
                        return TowerType.TENSION_DRUM  # Tension type
                    else:
                        return TowerType.DC_DRUM       # DC type
    
    # Default fallback
    return TowerType.WINE_GLASS

def classify_tower_by_features(cross_arms_count, has_cavity, cavity_area=0, line_gap_metric=0):
    """
    Helper function for tower classification based on extracted features
    
    Args:
        cross_arms_count: Number of detected cross arms
        has_cavity: Boolean indicating presence of significant cavity
        cavity_area: Size of largest cavity
        line_gap_metric: Metric indicating power line spacing
    Returns:
        TowerType enum value
    """
    if has_cavity:
        if cross_arms_count == 1:
            return TowerType.WINE_GLASS
        else:
            return TowerType.CAT_HEAD
    
    elif cross_arms_count == 1:
        if has_cavity:  # Extended cavity detection
            return TowerType.WINE_GLASS
        else:
            return TowerType.SINGLE_CROSS
    
    elif cross_arms_count == 2:
        if has_cavity:
            return TowerType.CAT_HEAD
        else:
            return TowerType.TENSION_DRY
    
    else:  # Multiple cross arms
        if line_gap_metric > 4:
            return TowerType.TENSION_DRUM
        else:
            return TowerType.DC_DRUM
            
    return TowerType.WINE_GLASS  # Default