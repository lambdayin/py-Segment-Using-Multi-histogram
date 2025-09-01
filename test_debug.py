#!/usr/bin/env python3
"""
Quick debug test to verify algorithm steps
"""
import sys
import os
sys.path.insert(0, '.')

from src.core.main_algorithm import InsulatorSegmentationPipeline

# Test with single tower and single grid width
pipeline = InsulatorSegmentationPipeline('./Data/')

# Load data
tower_points, line_points = pipeline.load_tower_data('001')

print(f"Loaded data: {len(tower_points)} tower points, {len(line_points)} line points")

# Test preprocessing
from src.preprocessing.tower_alignment import preprocess_point_clouds
tower_rotated, line_rotated, rotation_angle = preprocess_point_clouds(tower_points, line_points)
print(f"After preprocessing: rotation angle = {rotation_angle:.4f} rad")

# Test core algorithm with single grid width
from src.insulator_extraction.core_algorithm import type_inside_tree

print("\n=== Testing with grid width 0.1m ===")
insulator_points, is_cable, insulator_length = type_inside_tree(tower_rotated, line_rotated, 0.1)
print(f"Final result: {len(insulator_points)} insulator points, length={insulator_length}")