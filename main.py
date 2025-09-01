#!/usr/bin/env python3
"""
Main entry point for insulator segmentation using multi-histogram
Python translation of MATLAB main.m

Usage:
    python main.py [--data-path DATA_PATH] [--visualize] [--tower-id TOWER_ID]
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.main_algorithm import InsulatorSegmentationPipeline
from src.visualization.visualizer import set_visualization_backend

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Insulator Segmentation using Multi-Type and Multi-Scale Feature Histogram'
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='./Data/',
        help='Path to data directory containing tower point clouds (default: ./Data/)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Enable visualization of results'
    )
    
    parser.add_argument(
        '--tower-id', '-t',
        type=str,
        default=None,
        help='Process specific tower ID only (e.g., "001"). If not specified, processes all towers.'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        choices=['matplotlib', 'open3d'],
        default='open3d',
        help='Visualization backend (default: open3d)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    print("=== Insulator Segmentation using Multi-Histogram ===")
    print(f"Data path: {args.data_path}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    print(f"Backend: {args.backend}")
    
    # Set visualization backend
    set_visualization_backend(use_open3d=(args.backend == 'open3d'))
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        print("Please check the path or use --data-path to specify the correct location.")
        return 1
    
    # Initialize pipeline
    pipeline = InsulatorSegmentationPipeline(args.data_path)
    
    try:
        if args.tower_id:
            # Process single tower
            print(f"Processing single tower: {args.tower_id}")
            results = pipeline.process_single_tower(args.tower_id, visualize=args.visualize)
            
            if results:
                print("\n=== Single Tower Results ===")
                print(f"Tower ID: {results['tower_id']}")
                print(f"Insulators extracted: {len(results['insulator_points'])}")
                print(f"Insulator lengths: {results['insulator_lengths']}")
                print(f"Average verticality: {results['verticalities'].mean():.3f}" if len(results['verticalities']) > 0 else "No verticality data")
            else:
                print(f"Failed to process tower {args.tower_id}")
                return 1
        else:
            # Process all towers
            print("Processing all available towers...")
            results = pipeline.process_all_towers(visualize=args.visualize)
            
            if results:
                pipeline.print_summary_statistics(results)
            else:
                print("No towers were successfully processed!")
                return 1
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nProcessing completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)