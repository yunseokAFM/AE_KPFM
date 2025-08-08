# AI-Powered Automated SPM Scanning System

## Overview

This system provides automated scanning capabilities for Scanning Probe Microscopy (SPM) instruments, integrating artificial intelligence for intelligent object detection and precision scanning. The system performs grid-based scanning, analyzes scan data using Detectron2-based object detection, and executes targeted high-resolution scans on detected features.

## System Architecture

```
Grid Scanning → AI Object Detection → Precision Scanning → Statistical Analysis
     ↓                    ↓                     ↓                    ↓
 Approximate        Detectron2 CNN        Targeted Scans     GMM Analysis
   Scans            Segmentation         on Objects         & Validation
```

## Prerequisites

### Hardware Requirements
- SPM instrument with SmartRemote interface capability
- Computer with sufficient processing power for AI inference
- Stable network connection to SPM control system

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
1. **SPM Connection**: Ensure SmartRemote server is running on `localhost:5581`
2. **AI Model**: Pre-trained model or custom trained model at `output/model_final.pth`
3. **Data Storage**: Configure data directories with appropriate write permissions

## Quick Start Guide

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify SPM connection
python test_connection.py
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

## Advanced Features

### AI Model Training
For custom object detection:
```python
# Place training images in object/train/
# Add COCO format annotations
python Segmentation_Training.py
```

### Statistical Analysis
The system provides comprehensive GMM analysis:
- **Model Selection**: Automatic component number optimization using BIC/AIC
- **Statistical Validation**: Dip test for multimodality assessment  
- **Quality Metrics**: Assignment confidence and component validation
- **Visualization**: Multi-panel analysis plots and individual component visualization

### Performance Optimization
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Memory Management**: Dynamic memory cleanup and optimization
- **Batch Processing**: Configurable batch sizes for different hardware capabilities

## Output Files

### Scan Data
- **TIFF Images**: Raw scan data in standard TIFF format
- **Metadata**: Scan parameters and stage positions
- **Timestamps**: Automatic timestamping for data organization

### Analysis Results
- **Detection Results**: Object coordinates and confidence scores
- **Statistical Analysis**: GMM fitting parameters and validation metrics
- **Visualizations**: Comprehensive plots for data interpretation

## Troubleshooting

### Connection Issues
- **Symptom**: "SmartRemote connection failed"
- **Solution**: Verify SmartRemote server status and port 5581 availability

### Detection Problems  
- **Symptom**: No objects detected consistently
- **Solution**: Lower `score_threshold` or retrain detection model

### Performance Issues
- **Symptom**: Slow processing or memory errors
- **Solution**: Enable CPU mode or reduce batch size in configuration

### Hardware Limitations
- **Symptom**: Stage movement failures
- **Solution**: Verify scan parameters within instrument limits

## System Monitoring

The system provides real-time feedback:
- **Connection Status**: Continuous SPM connection monitoring
- **Scan Progress**: Grid position and completion tracking  
- **Detection Results**: Object count and confidence reporting
- **Performance Metrics**: Processing time and success rates

## Safety Features

- **Automatic Limits**: Stage movement boundary validation
- **Error Recovery**: Automatic retry mechanisms for failed operations
- **Data Integrity**: Checksum validation for critical data files
- **Graceful Shutdown**: Safe system termination protocols

## Technical Support

### Log Files
Check system logs for detailed error information:
- Connection logs: Monitor SPM communication
- Detection logs: AI model performance tracking
- Scan logs: Stage movement and scan execution

### Performance Monitoring
- **GPU Memory**: Automatic monitoring and optimization
- **Processing Time**: Per-scan timing analysis
- **Success Rates**: Statistical tracking of operation success

---

**Compatibility**: SPM systems with SmartRemote interface
