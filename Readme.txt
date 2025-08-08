# AI-Powered Automated SPM Scanning System

## Overview

The system performs grid-based scanning, analyzes scan data using Detectron2-based object detection, and executes targeted high-resolution scans on detected features.

## Prerequisites

### Software Requirements
```
Python >= 3.8
PyTorch >= 1.9.0
Detectron2 >= 0.6
OpenCV >= 4.5.0
NumPy >= 1.21.0
Matplotlib >= 3.3.0
scikit-learn >= 1.0.0
```

### System Setup
1. **SPM Connection**: Ensure SmartRemote server
2. **AI Model**: Pre-trained model at `output/model_final.pth`

## Quick Start Guide

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

```

### 2. Basic Operation
```python
from Smart_remote import SmartScanningSystem, ScanParams

# Configure scan parameters
scan_params = ScanParams(
    x_range=60.0,           # Scan width (μm)
    y_range=60.0,           # Scan height (μm) 
    x_iterations=5,         # Grid columns
    y_iterations=5,         # Grid rows
    approximate_resolution=256,    # Initial scan resolution
    precision_resolution=256,      # High-res scan resolution
    score_threshold=0.7     # AI detection confidence threshold
)

# Execute automated scanning
scanner = SmartScanningSystem()
success = scanner.run_grid_scan(scan_params)
```

### 3. Workflow Execution
The system automatically executes the following sequence:
1. **Grid Navigation**: Moves SPM stage to each grid position
2. **Approximate Scanning**: Performs initial scan at each position
3. **AI Analysis**: Detects objects using trained CNN model
4. **Precision Scanning**: Executes high-resolution scans on detected features
5. **Statistical Analysis**: Applies GMM/KDE analysis to scan results

## Configuration

### Scan Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `x_range` | Scan area width (μm) | 60.0 | 1-500 |
| `y_range` | Scan area height (μm) | 60.0 | 1-500 |
| `x_iterations` | Grid columns | 5 | 1-20 |
| `y_iterations` | Grid rows | 5 | 1-20 |
| `score_threshold` | AI confidence threshold | 0.7 | 0.1-1.0 |
| `size_multiplier` | Precision scan size factor | 4.0 | 1.0-10.0 |

### Data Storage Structure
```
Data/
├── Approximate/          # Initial grid scans
│   └── YYMMDD_HHMMSS_ZeroScan/
├── Precision/            # High-resolution targeted scans
│   └── YYMMDD_HHMMSS_Precision_N/
└── Analysis/             # Statistical analysis results
    ├── spatial_data.csv
    ├── gmm_results.csv
    └── visualization.png
```

### Statistical Analysis
The system provides comprehensive GMM analysis:
- **Model Selection**: Automatic component number optimization using BIC/AIC
- **Statistical Validation**: Dip test for multimodality assessment  
- **Visualization**: Multi-panel analysis plots and individual component visualization


---
