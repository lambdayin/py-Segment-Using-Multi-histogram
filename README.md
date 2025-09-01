# Insulator Segmentation using Multi-Histogram (Python Version)

This is a Python implementation for **"Insulator Extraction from UAV LiDAR Point Cloud Based on Multi-Type and Multi-Scale Feature Histogram"** by Chen, M.; Li, J.; Pan, J.; Ji, C.; Ma, W. (Drones 2024, 8, 241).

## Overview

A Python implementation of the automated insulator extraction system from UAV LiDAR point clouds using:

- **5 Feature Histograms**: HD/HV/HW/VW/VV (Height/Density/Width/Volume/Verticality)
- **Multi-Scale Adaptive Grids**: 0.05m - 0.15m with 0.01m steps
- **Tower-Type Specific Processing**: 6 different tower types supported
- **2-Stage Workflow**: Pylon/PL refinement + Insulator extraction

## Installation

### Requirements

- Python 3.7+
- See `requirements.txt` for detailed dependencies

### Setup

```bash
# Clone or extract the project
cd py-Segment-Using-Multi-histogram

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

### Dependencies

- **NumPy**: Numerical computations
- **SciPy**: Scientific computing and rotations
- **scikit-learn**: Machine learning algorithms (DBSCAN, RANSAC)
- **scikit-image**: Image processing
- **Open3D**: Point cloud processing and visualization
- **OpenCV**: Image operations
- **Matplotlib**: Plotting and visualization

## Usage

### Basic Usage

```bash
# Process all towers in default data directory
python main.py

# Process with visualization enabled
python main.py --visualize

# Specify custom data path
python main.py --data-path /path/to/your/data --visualize

# Process specific tower
python main.py --tower-id 001 --visualize
```

### Command Line Options

- `--data-path, -d`: Path to data directory (default: `./Data/`)
- `--visualize, -v`: Enable result visualization
- `--tower-id, -t`: Process specific tower ID (e.g., "001")
- `--backend`: Visualization backend (`open3d` or `matplotlib`)

### Data Format

The system expects data files in the following format:

```
Data/
├── 001Tower.txt    # Tower point cloud (X Y Z format)
├── 001Line.txt     # Power line point cloud (X Y Z format)
├── 002Tower.txt
├── 002Line.txt
└── ...
```

Each `.txt` file should contain point cloud data with space-separated X, Y, Z coordinates:

```
x1 y1 z1
x2 y2 z2
...
```

## Algorithm Architecture

### Core Modules

1. **Preprocessing** (`src/preprocessing/`)
   - Tower alignment and rotation
   - Noise filtering and cleanup

2. **Tower Detection** (`src/tower_detection/`)
   - 6 tower type classification
   - Cross-arm detection
   - Cavity analysis for O-type towers

3. **Insulator Extraction** (`src/insulator_extraction/`)
   - Type-specific extraction algorithms
   - Multi-scale processing
   - Adaptive grid optimization

4. **Visualization** (`src/visualization/`)
   - 3D point cloud display
   - Result visualization
   - Multiple backend support

### Supported Tower Types

1. **Wine Glass Tower** (酒杯塔)
2. **Cat Head Tower** (猫头塔)  
3. **Single Cross-arm** (单横担)
4. **Tension Dry Tower** (耐张干字塔)
5. **Tension Drum Tower** (耐张鼓型塔)
6. **DC Drum Tower** (直流鼓型塔)
7. **Portal Tower** (门型塔)

### Key Features

- **Multi-Scale Processing**: Tests grid widths from 0.05m to 0.15m
- **Adaptive Grid Selection**: Automatically selects optimal grid size per insulator
- **Type-Specific Algorithms**: Different extraction methods for each tower type
- **Verticality Analysis**: Distinguishes horizontal vs vertical insulators
- **Quality Metrics**: Length, completeness, and verticality scoring

## API Usage

```python
from src.core.main_algorithm import InsulatorSegmentationPipeline
from src.visualization.visualizer import set_visualization_backend

# Initialize pipeline
pipeline = InsulatorSegmentationPipeline(data_path="./Data/")

# Process single tower
results = pipeline.process_single_tower("001", visualize=True)

# Process all towers
all_results = pipeline.process_all_towers(visualize=False)

# Print statistics
pipeline.print_summary_statistics(all_results)
```

## Algorithm Flow

1. **Data Loading**: Read tower and line point clouds
2. **Preprocessing**: Align towers, remove noise
3. **Tower Classification**: Detect tower type (1-6, 8)
4. **Multi-Scale Processing**: Test multiple grid widths
5. **Type-Specific Extraction**: Apply appropriate algorithm
6. **Adaptive Optimization**: Select optimal results
7. **Visualization**: Display results

## Performance

The Python implementation aims to maintain the performance characteristics:

- **Processing Speed**: <2 seconds per tower
- **Suspension Tower Accuracy**: Target 100%
- **Tension Tower Accuracy**: Target 97.3%

## Troubleshooting

### Common Issues

**"No tower data found"**

- Check data path and file naming convention
- Ensure files follow `XXXTower.txt` and `XXXLine.txt` format

**Visualization not working**  

- Try switching backends: `--backend matplotlib`
- Check Open3D installation: `python -c "import open3d; print('OK')"`

**Import errors**

- Install requirements: `pip install -r requirements.txt`
- Check Python version: requires Python 3.7+

### Performance Tips

- Use `--backend matplotlib` for faster processing without 3D visualization
- Process specific towers for debugging: `--tower-id 001`
- Disable visualization for batch processing