#!/usr/bin/env python3
"""
Test Type4 horizontal insulator extraction specifically
"""
import sys
import os
sys.path.insert(0, '.')

from src.core.main_algorithm import InsulatorSegmentationPipeline

# Test with single tower 
pipeline = InsulatorSegmentationPipeline('./Data/')

# Load data
tower_points, line_points = pipeline.load_tower_data('001')
print(f"Loaded data: {len(tower_points)} tower points, {len(line_points)} line points")

# Test preprocessing
from src.preprocessing.tower_alignment import preprocess_point_clouds
tower_rotated, line_rotated, rotation_angle = preprocess_point_clouds(tower_points, line_points)

# Force tower type to single cross (type 3) to test Type4 extraction
from src.core.constants import TowerType
from src.insulator_extraction.extraction_methods import extract_insulators_type4
from src.utils.projection_utils import bin_projection, cross_location_detection
from src.utils.pointcloud_utils import split_point_cloud_c4

print("\n=== Testing Type4 Horizontal Insulator Extraction ===")

# Create binary projection for cross-arm detection
bin_image_yz, _ = bin_projection(tower_rotated, 0.1, axis_x=1, axis_y=2)
print(f"Binary projection size: {bin_image_yz.shape}")

# Detect cross-arm locations
cross_locations = cross_location_detection(bin_image_yz, ratio=3)
print(f"Cross-arm locations: {cross_locations}")

# Split line points into clusters
cross_line_clusters = split_point_cloud_c4(line_rotated, n_clusters=2, eps=0.5)
print(f"Line clusters: {len(cross_line_clusters)} clusters")

# Test Type4 extraction
if cross_line_clusters and cross_locations:
    insulator_points, length = extract_insulators_type4(
        tower_rotated, cross_line_clusters, cross_locations, 0.1, TowerType.SINGLE_CROSS
    )
    print(f"Type4 result: {len(insulator_points)} insulator points, length={length:.2f}")
else:
    print("No clusters or cross-locations found for Type4 test")