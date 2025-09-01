#!/usr/bin/env python3
"""
Basic functionality tests for the insulator segmentation system
"""

import sys
import os
import numpy as np
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.math_utils import rotz, rotate_with_axle, calc_verticality
from src.utils.pointcloud_utils import load_point_cloud, cluster_points_dbscan
from src.utils.projection_utils import bin_projection, cross_location_detection
from src.tower_detection.tower_classifier import detect_tower_type
from src.core.constants import TowerType

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core modules"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample point clouds for testing
        self.sample_tower = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [0, 0, 2]
        ])
        
        self.sample_line = np.array([
            [2, 2, 3],
            [2, -2, 3],
            [-2, 2, 3],
            [-2, -2, 3]
        ])
        
        self.vertical_points = np.array([
            [0, 0, 0],
            [0, 0, 1], 
            [0, 0, 2],
            [0, 0, 3]
        ])
        
        self.horizontal_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0]
        ])
    
    def test_rotation_matrices(self):
        """Test rotation matrix functions"""
        # Test Z-axis rotation
        rot_matrix = rotz(90)  # 90 degrees
        self.assertEqual(rot_matrix.shape, (3, 3))
        
        # Test that rotation matrix is orthogonal
        identity = rot_matrix @ rot_matrix.T
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)
    
    def test_point_cloud_rotation(self):
        """Test point cloud rotation functionality"""
        rotated_points, angle = rotate_with_axle(self.sample_tower, axis=3)
        
        # Check that we get the same number of points
        self.assertEqual(len(rotated_points), len(self.sample_tower))
        self.assertEqual(rotated_points.shape, self.sample_tower.shape)
        
        # Check that angle is a scalar
        self.assertIsInstance(angle, (float, np.floating))
    
    def test_verticality_calculation(self):
        """Test verticality calculation"""
        # Vertical points should have high verticality
        vertical_score = calc_verticality(self.vertical_points)
        self.assertGreater(vertical_score, 0.5, "Vertical points should have high verticality")
        
        # Horizontal points should have low verticality
        horizontal_score = calc_verticality(self.horizontal_points)
        self.assertLess(horizontal_score, 0.5, "Horizontal points should have low verticality")
    
    def test_clustering(self):
        """Test point cloud clustering"""
        # Create two separate clusters
        cluster1 = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        cluster2 = np.array([[5, 5, 0], [5.1, 5, 0], [5, 5.1, 0]])
        
        combined = np.vstack([cluster1, cluster2])
        labels = cluster_points_dbscan(combined, eps=0.5, min_samples=1)
        
        # Should have 2 clusters
        unique_labels = len(np.unique(labels[labels >= 0]))
        self.assertEqual(unique_labels, 2, "Should detect 2 separate clusters")
    
    def test_binary_projection(self):
        """Test binary projection functionality"""
        binary_image, grid_cells = bin_projection(
            self.sample_tower, grid_width=0.5, axis_x=0, axis_y=2
        )
        
        # Check that binary image is created
        self.assertGreater(binary_image.size, 0)
        self.assertEqual(len(binary_image.shape), 2)
        
        # Check that grid cells are created
        self.assertIsInstance(grid_cells, list)
    
    def test_cross_location_detection(self):
        """Test cross-arm location detection"""
        # Create a simple binary image with some structure
        binary_image = np.zeros((20, 10), dtype=bool)
        binary_image[5:8, :] = True   # Cross-arm 1
        binary_image[12:15, :] = True  # Cross-arm 2
        
        locations = cross_location_detection(binary_image, ratio=3)
        
        # Should detect some locations
        self.assertIsInstance(locations, list)
        # Note: Exact number depends on the algorithm implementation
    
    def test_tower_type_detection(self):
        """Test tower type detection"""
        tower_type = detect_tower_type(self.sample_tower, self.sample_line)
        
        # Should return a valid tower type
        self.assertIsInstance(tower_type, TowerType)
        self.assertIn(tower_type, list(TowerType))
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        empty_points = np.empty((0, 3))
        
        # Rotation with empty points
        rotated, angle = rotate_with_axle(empty_points)
        self.assertEqual(len(rotated), 0)
        self.assertEqual(angle, 0.0)
        
        # Verticality with empty points
        verticality = calc_verticality(empty_points)
        self.assertEqual(verticality, 0.0)
        
        # Clustering with empty points
        labels = cluster_points_dbscan(empty_points)
        self.assertEqual(len(labels), 0)
    
    def test_data_format_consistency(self):
        """Test that data formats are consistent throughout pipeline"""
        # All point clouds should be Nx3 arrays
        self.assertEqual(self.sample_tower.shape[1], 3)
        self.assertEqual(self.sample_line.shape[1], 3)
        
        # Binary projections should return 2D arrays
        binary_image, _ = bin_projection(self.sample_tower, 0.5)
        self.assertEqual(len(binary_image.shape), 2)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and file operations"""
    
    def test_sample_data_loading(self):
        """Test loading sample data if available"""
        # Try to load a sample data file if it exists
        sample_file = "../Data/001Tower.txt"
        if os.path.exists(sample_file):
            points = load_point_cloud(sample_file)
            
            # Basic validity checks
            self.assertGreater(len(points), 0, "Sample data should not be empty")
            self.assertEqual(points.shape[1], 3, "Points should have 3 coordinates")
            self.assertTrue(np.all(np.isfinite(points)), "All coordinates should be finite")
        else:
            self.skipTest("Sample data file not found")

def run_basic_tests():
    """Run basic functionality tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running basic functionality tests...")
    success = run_basic_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)