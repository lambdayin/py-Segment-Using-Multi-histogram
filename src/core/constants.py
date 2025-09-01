"""
Constants for insulator segmentation algorithm
Translated from MATLAB implementation
"""
from enum import IntEnum

class TowerType(IntEnum):
    """Tower types as defined in original MATLAB code"""
    WINE_GLASS = 1      # 酒杯塔 - Wine glass tower  
    CAT_HEAD = 2        # 猫头塔 - Cat head tower
    SINGLE_CROSS = 3    # 单横担 - Single cross-arm
    TENSION_DRY = 4     # 耐张干字塔 - Tension resistant dry tower
    TENSION_DRUM = 5    # 耐张鼓型塔 - Tension drum tower
    DC_DRUM = 6         # 直流鼓型塔 - DC drum tower
    PORTAL = 8          # 门型塔 - Portal tower

# Grid configuration
GRID_WIDTH_MIN = 0.05    # Minimum grid width in meters
GRID_WIDTH_MAX = 0.15    # Maximum grid width in meters
GRID_WIDTH_STEP = 0.01   # Grid width step size

# Algorithm thresholds
VERTICALITY_THRESHOLD = 0.8     # Threshold to distinguish horizontal vs vertical insulators
MIN_INSULATOR_LENGTH = 1.0      # Minimum insulator length in meters
MAX_INSULATOR_LENGTH = 2.5      # Maximum insulator length in meters

# Clustering parameters
DBSCAN_MIN_SAMPLES = 1
CLUSTERING_DISTANCES = {
    'power_line_2d': 0.1,
    'power_line_c4': [0.5, 0.6, 0.8],
    'power_line_c5': 0.5,
    'general': 1.0
}

# Cross-arm detection ratios
CROSS_ARM_DETECTION_RATIOS = [1/2, 1/3, 1/4]

# Visualization colors
COLORS = {
    'tower': 'blue',
    'line': 'green', 
    'insulator': 'red'
}