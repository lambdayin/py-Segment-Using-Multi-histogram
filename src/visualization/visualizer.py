"""
Visualization utilities for point clouds and results
Translated from MATLAB drowPts functionality
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from ..core.constants import COLORS

class PointCloudVisualizer:
    """Point cloud visualization class"""
    
    def __init__(self, use_open3d=True):
        self.use_open3d = use_open3d
        self.fig = None
        self.ax = None
        
    def setup_matplotlib_3d(self):
        """Setup matplotlib 3D plot"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
    def plot_points_matplotlib(self, points, color='blue', marker='.', markersize=1, label=None):
        """Plot points using matplotlib"""
        self.setup_matplotlib_3d()
        
        if len(points) > 0:
            if isinstance(color, str):
                color_map = {
                    'blue': 'b', 'green': 'g', 'red': 'r', 
                    'yellow': 'y', 'cyan': 'c', 'magenta': 'm'
                }
                color = color_map.get(color, color)
            
            self.ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=color, marker=marker, s=markersize, label=label, alpha=0.6
            )
    
    def plot_points_open3d(self, tower_points, line_points=None, insulator_points=None):
        """Plot points using Open3D"""
        geometries = []
        
        # Tower points (blue)
        if len(tower_points) > 0:
            tower_pcd = o3d.geometry.PointCloud()
            tower_pcd.points = o3d.utility.Vector3dVector(tower_points)
            tower_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
            geometries.append(tower_pcd)
        
        # Line points (green)  
        if line_points is not None and len(line_points) > 0:
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line_points)
            line_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green
            geometries.append(line_pcd)
        
        # Insulator points (red)
        if insulator_points is not None and len(insulator_points) > 0:
            ins_pcd = o3d.geometry.PointCloud()
            if insulator_points.shape[1] >= 3:
                ins_pcd.points = o3d.utility.Vector3dVector(insulator_points[:, :3])
            else:
                ins_pcd.points = o3d.utility.Vector3dVector(insulator_points)
            ins_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            geometries.append(ins_pcd)
        
        if geometries:
            o3d.visualization.draw_geometries(geometries)
    
    def drow_pts(self, *args):
        """
        Main visualization function mimicking MATLAB drowPts
        Usage: drow_pts(tower_pts, '.b', line_pts, '.g', ins_pts, '.r')
        """
        if self.use_open3d:
            self._drow_pts_open3d(*args)
        else:
            self._drow_pts_matplotlib(*args)
    
    def _drow_pts_matplotlib(self, *args):
        """Matplotlib implementation of drowPts"""
        self.setup_matplotlib_3d()
        
        i = 0
        while i < len(args):
            if i + 1 < len(args) and isinstance(args[i+1], str):
                # Format: points, style_string
                points = args[i]
                style = args[i+1]
                
                if len(points) > 0:
                    color = self._parse_style_string(style)
                    self.plot_points_matplotlib(points, color=color)
                
                i += 2
            else:
                # Just points without style
                points = args[i]
                if len(points) > 0:
                    self.plot_points_matplotlib(points)
                i += 1
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')  
        self.ax.set_zlabel('Z')
        self.ax.legend()
        plt.show()
    
    def _drow_pts_open3d(self, *args):
        """Open3D implementation of drowPts"""
        geometries = []
        
        i = 0
        while i < len(args):
            if i + 1 < len(args) and isinstance(args[i+1], str):
                points = args[i]
                style = args[i+1]
                
                if len(points) > 0:
                    color = self._parse_style_to_rgb(style)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    pcd.paint_uniform_color(color)
                    geometries.append(pcd)
                
                i += 2
            else:
                points = args[i]
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Default gray
                    geometries.append(pcd)
                i += 1
        
        if geometries:
            o3d.visualization.draw_geometries(geometries)
    
    def _parse_style_string(self, style):
        """Parse MATLAB-style string like '.r', '.b', '.g'"""
        color_map = {
            'r': 'red', 'g': 'green', 'b': 'blue',
            'y': 'yellow', 'c': 'cyan', 'm': 'magenta',
            'k': 'black', 'w': 'white'
        }
        
        color_char = style.replace('.', '')
        return color_map.get(color_char, 'blue')
    
    def _parse_style_to_rgb(self, style):
        """Parse style string to RGB values for Open3D"""
        color_map = {
            'r': [1.0, 0.0, 0.0], 'g': [0.0, 1.0, 0.0], 'b': [0.0, 0.0, 1.0],
            'y': [1.0, 1.0, 0.0], 'c': [0.0, 1.0, 1.0], 'm': [1.0, 0.0, 1.0],
            'k': [0.0, 0.0, 0.0], 'w': [1.0, 1.0, 1.0]
        }
        
        color_char = style.replace('.', '')
        return color_map.get(color_char, [0.0, 0.0, 1.0])
    
    def plot_histogram(self, histogram, title="Histogram"):
        """Plot 1D histogram"""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(histogram)), histogram)
        plt.title(title)
        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.show()
    
    def plot_binary_image(self, binary_image, title="Binary Image"):
        """Plot 2D binary image"""
        plt.figure(figsize=(8, 6))
        plt.imshow(binary_image, cmap='gray', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()

# Global visualizer instance
_visualizer = PointCloudVisualizer()

def drow_pts(*args):
    """
    Global function mimicking MATLAB drowPts
    """
    _visualizer.drow_pts(*args)

def set_visualization_backend(use_open3d=True):
    """
    Set visualization backend
    
    Args:
        use_open3d: If True, use Open3D; if False, use matplotlib
    """
    global _visualizer
    _visualizer.use_open3d = use_open3d