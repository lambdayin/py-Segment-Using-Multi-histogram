"""
Main insulator segmentation algorithm
Translated from MATLAB main.m
"""
import os
import numpy as np
from glob import glob
from ..utils.pointcloud_utils import load_point_cloud
from ..preprocessing.tower_alignment import preprocess_point_clouds
from ..insulator_extraction.core_algorithm import process_multi_grid_extraction
from ..insulator_extraction.adaptive_grid import adaptive_grid_tension, merge_insulator_results, merge_cell3
from ..visualization.visualizer import drow_pts
from ..core.constants import GRID_WIDTH_MIN, GRID_WIDTH_MAX, GRID_WIDTH_STEP

class InsulatorSegmentationPipeline:
    """Main pipeline for insulator segmentation using multi-histogram"""
    
    def __init__(self, data_path="./Data/"):
        self.data_path = data_path
        self.grid_widths = np.arange(GRID_WIDTH_MIN, GRID_WIDTH_MAX + GRID_WIDTH_STEP, GRID_WIDTH_STEP)
        
    def get_tower_ids(self):
        """
        Get list of available tower IDs from data directory
        Translated from getTowerID.m
        
        Returns:
            List of tower ID strings
        """
        if not os.path.exists(self.data_path):
            print(f"Data path {self.data_path} does not exist!")
            return []
        
        # Find all tower files (pattern: XXXTower.txt)
        tower_files = glob(os.path.join(self.data_path, "*Tower.txt"))
        
        tower_ids = []
        for file_path in tower_files:
            filename = os.path.basename(file_path)
            # Extract tower ID (remove 'Tower.txt')
            tower_id = filename.replace('Tower.txt', '')
            tower_ids.append(tower_id)
        
        return sorted(tower_ids)
    
    def load_tower_data(self, tower_id):
        """
        Load tower and line point cloud data
        
        Args:
            tower_id: Tower identifier string
        Returns:
            tuple: (tower_points, line_points)
        """
        tower_file = os.path.join(self.data_path, f"{tower_id}Tower.txt")
        line_file = os.path.join(self.data_path, f"{tower_id}Line.txt")
        
        tower_points = load_point_cloud(tower_file)
        line_points = load_point_cloud(line_file)
        
        return tower_points, line_points
    
    def process_single_tower(self, tower_id, visualize=True):
        """
        Process a single tower for insulator extraction
        
        Args:
            tower_id: Tower identifier
            visualize: Whether to show visualization
        Returns:
            dict: Processing results containing insulator points, lengths, etc.
        """
        print(f"Processing Tower {tower_id}...")
        
        # Load data
        tower_points, line_points = self.load_tower_data(tower_id)
        
        if len(tower_points) == 0:
            print(f"No tower data found for {tower_id}")
            return None
        
        # Preprocess point clouds
        tower_rotated, line_rotated, rotation_angle = preprocess_point_clouds(
            tower_points, line_points
        )
        
        print(f"Loaded {len(tower_rotated)} tower points, {len(line_rotated)} line points")
        print(f"Rotation angle: {np.degrees(rotation_angle):.2f} degrees")
        
        # Multi-scale processing
        print("Processing multiple grid scales...")
        all_insulator_results = []
        all_length_results = []
        
        for i, grid_width in enumerate(self.grid_widths):
            print(f"  Grid width {grid_width:.3f}m ({i+1}/{len(self.grid_widths)})")
            
            insulator_results, length_results = process_multi_grid_extraction(
                tower_rotated, line_rotated, [grid_width]
            )
            
            if insulator_results:
                all_insulator_results.append(insulator_results[0])
                all_length_results.append(length_results[0])
            else:
                all_insulator_results.append(np.empty((0, 3)))
                all_length_results.append(0.0)
        
        # Adaptive grid selection
        print("Applying adaptive grid optimization...")
        
        # Reshape results for adaptive_grid_tension function
        # MATLAB expects (n_insulators, n_grid_widths) format
        if all_insulator_results:
            print(f"DEBUG: Reshaping results - {len(all_insulator_results)} grid results")
            
            # For single-insulator-per-grid case, create proper matrix format
            # This matches the MATLAB InsPtsInGrids and LenPtsInGrids structure
            reshaped_insulators = [all_insulator_results]  # Single row of all grid results
            reshaped_lengths = np.array([all_length_results])  # Single row of lengths
            
            print(f"DEBUG: Calling adaptive_grid_tension with shapes: insulators={len(reshaped_insulators)}x{len(reshaped_insulators[0])}, lengths={reshaped_lengths.shape}")
            
            final_insulators, final_lengths, final_indices, final_verticalities = adaptive_grid_tension(
                reshaped_insulators, reshaped_lengths
            )
            
            print(f"DEBUG: Adaptive grid result: {len(final_insulators)} final points")
        else:
            print("DEBUG: No insulator results to process")
            final_insulators = np.empty((0, 4))
            final_lengths = np.array([])
            final_indices = np.array([])
            final_verticalities = np.array([])
        
        # Prepare results
        results = {
            'tower_id': tower_id,
            'tower_points': tower_rotated,
            'line_points': line_rotated,
            'insulator_points': final_insulators,
            'insulator_lengths': final_lengths,
            'grid_indices': final_indices,
            'verticalities': final_verticalities,
            'rotation_angle': rotation_angle,
            'grid_widths': self.grid_widths
        }
        
        print(f"Final extraction results:")
        print(f"  - Insulator points: {len(final_insulators)}")
        print(f"  - Insulator lengths: {final_lengths}")
        print(f"  - Grid indices: {final_indices}")
        print(f"  - Verticalities: {final_verticalities}")
        
        # Visualization (matching MATLAB drowPts call)
        if visualize and len(final_insulators) > 0:
            print("Displaying results...")
            try:
                drow_pts(
                    tower_rotated, '.b',
                    line_rotated, '.g', 
                    final_insulators[:, :3], '.r'
                )
            except Exception as e:
                print(f"Visualization error: {e}")
        elif visualize:
            print("No insulator points to visualize")
        
        return results
    
    def process_all_towers(self, visualize=False):
        """
        Process all available towers
        
        Args:
            visualize: Whether to show visualizations for each tower
        Returns:
            List of processing results
        """
        tower_ids = self.get_tower_ids()
        
        if not tower_ids:
            print("No tower data found!")
            return []
        
        print(f"Found {len(tower_ids)} towers to process: {tower_ids}")
        
        all_results = []
        
        for i, tower_id in enumerate(tower_ids):
            print(f"\n=== Processing Tower {i+1}/{len(tower_ids)}: {tower_id} ===")
            
            try:
                results = self.process_single_tower(tower_id, visualize=visualize)
                if results:
                    all_results.append(results)
                    
                    # Print summary
                    n_insulators = len(results['insulator_points'])
                    avg_length = np.mean(results['insulator_lengths']) if len(results['insulator_lengths']) > 0 else 0
                    print(f"Summary: {n_insulators} insulators, avg length: {avg_length:.2f}m")
                
            except Exception as e:
                print(f"Error processing tower {tower_id}: {e}")
                continue
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed {len(all_results)}/{len(tower_ids)} towers")
        
        return all_results
    
    def print_summary_statistics(self, results_list):
        """Print summary statistics for all processed towers"""
        if not results_list:
            print("No results to summarize")
            return
        
        total_towers = len(results_list)
        total_insulators = sum(len(r['insulator_points']) for r in results_list)
        
        all_lengths = []
        all_verticalities = []
        
        for results in results_list:
            if len(results['insulator_lengths']) > 0:
                all_lengths.extend(results['insulator_lengths'])
            if len(results['verticalities']) > 0:
                all_verticalities.extend(results['verticalities'])
        
        print("\n=== Summary Statistics ===")
        print(f"Total towers processed: {total_towers}")
        print(f"Total insulators extracted: {total_insulators}")
        print(f"Average insulators per tower: {total_insulators/total_towers:.1f}")
        
        if all_lengths:
            print(f"Insulator lengths - Mean: {np.mean(all_lengths):.2f}m, "
                  f"Std: {np.std(all_lengths):.2f}m, "
                  f"Range: [{np.min(all_lengths):.2f}, {np.max(all_lengths):.2f}]m")
        
        if all_verticalities:
            print(f"Verticalities - Mean: {np.mean(all_verticalities):.3f}, "
                  f"Std: {np.std(all_verticalities):.3f}")

def main(data_path="./Data/", visualize_all=False):
    """
    Main entry point matching MATLAB main.m functionality
    
    Args:
        data_path: Path to data directory
        visualize_all: Whether to show visualizations for all towers
    """
    # Initialize pipeline
    pipeline = InsulatorSegmentationPipeline(data_path)
    
    # Process all towers
    results = pipeline.process_all_towers(visualize=visualize_all)
    
    # Print summary
    pipeline.print_summary_statistics(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    results = main(data_path="../Data/", visualize_all=True)