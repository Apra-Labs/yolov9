# YOLOv9 SAHI Integration

This document describes the integration of SAHI (Slicing Aided Hyper Inference) with YOLOv9 for improved small object detection.

## Overview

Two new scripts have been added to enhance the YOLOv9 detection capabilities using SAHI:
1. `val_dual_sahi.py` - For validation with SAHI support
2. `detect_sahi.py` - For inference with SAHI support

## Key Additions and Modifications

### Common Features in Both Scripts

1. **SAHI Integration**
   - Added SAHI model initialization using `AutoDetectionModel`
   - Support for sliced prediction using `get_sliced_prediction`
   - Configurable SAHI parameters:
     - `sahi_slice_height` (default: 512)
     - `sahi_slice_width` (default: 512)
     - `sahi_overlap_height_ratio` (default: 0.2)
     - `sahi_overlap_width_ratio` (default: 0.2)
     - `sahi_blob_area_threshold` (default: 0)

2. **Bounding Box Handling**
   - Added `resize_with_letterbox_bbox` function for proper bbox coordinate transformation
   - Enhanced letterboxing support with the `letterbox` function
   - Support for custom ROIs with `use_custom_rois` parameter

3. **Blob Defects Support**
   - Added support for blob defects information through JSON files
   - Integration with `blob_defects_info` parameter

### val_dual_sahi.py Specific Features

1. **Validation Enhancements**
   - Added SAHI-specific validation metrics
   - Support for dual validation (with and without SAHI)
   - Enhanced batch processing with SAHI support
   - Added class-specific confidence thresholds

2. **Additional Parameters**
   - `min_items`: Experimental parameter for minimum items
   - `top_k`: Parameter for top-k predictions
   - Enhanced support for COCO format output

### detect_sahi.py Specific Features

1. **Inference Enhancements**
   - Real-time SAHI-based detection
   - Support for various input sources (images, videos, webcam)
   - Enhanced visualization options
   - Support for saving results in multiple formats

2. **Additional Parameters**
   - `vid_stride`: Video frame-rate stride
   - Enhanced visualization parameters
   - Support for class-specific confidence thresholds

## Usage Examples

### Validation with SAHI
```bash
python val_dual_sahi.py --weights yolov9.pt --data data.yaml --use-sahi --sahi-slice-height 512 --sahi-slice-width 512
```

### Detection with SAHI
```bash
python detect_sahi.py --weights yolov9.pt --source path/to/images --use-sahi --sahi-slice-height 512 --sahi-slice-width 512
```

## Key Differences from Original Scripts

1. **SAHI Integration**
   - Original scripts: No SAHI support
   - New scripts: Full SAHI integration for improved small object detection

2. **Bounding Box Handling**
   - Original scripts: Basic bbox handling
   - New scripts: Enhanced bbox handling with letterboxing support

3. **Performance Optimization**
   - Original scripts: Standard inference
   - New scripts: Optimized for small object detection through SAHI

4. **Additional Features**
   - Added support for blob defects
   - Enhanced visualization options
   - More flexible configuration options

## Dependencies

The new scripts require additional dependencies:
```bash
pip install sahi
pip install opencv-python
pip install torch
pip install numpy
```

## Notes

- SAHI is particularly useful for detecting small objects in large images
- The scripts maintain backward compatibility with original YOLOv9 functionality
- Performance may vary based on the SAHI parameters chosen
- Memory usage may increase when using SAHI due to image slicing

## SAHI Parameters Explanation

### Slice Size
- `sahi_slice_height` and `sahi_slice_width`: Define the size of each slice
- Default: 512x512 pixels
- Smaller slices may improve small object detection but increase processing time

### Overlap Ratio
- `sahi_overlap_height_ratio` and `sahi_overlap_width_ratio`: Define the overlap between slices
- Default: 0.2 (20% overlap)
- Higher overlap may improve detection at slice boundaries but increase processing time

### Blob Area Threshold
- `sahi_blob_area_threshold`: Minimum area for blob detection
- Default: 0
- Useful for filtering out noise in blob detection

## Best Practices

1. **Parameter Tuning**
   - Adjust slice size based on your object sizes
   - Increase overlap ratio for better boundary detection
   - Use blob area threshold to reduce false positives

2. **Memory Management**
   - Monitor memory usage when processing large images
   - Adjust batch size if needed
   - Consider using smaller slice sizes for memory-constrained systems

3. **Performance Optimization**
   - Use GPU acceleration when available
   - Adjust overlap ratio based on your specific use case
   - Consider using custom ROIs for specific areas of interest

## Troubleshooting

1. **Memory Issues**
   - Reduce slice size
   - Decrease batch size
   - Use smaller overlap ratio

2. **Performance Issues**
   - Enable GPU acceleration
   - Optimize SAHI parameters
   - Use custom ROIs to focus on specific areas

3. **Detection Quality**
   - Increase overlap ratio
   - Adjust confidence thresholds
   - Fine-tune slice size based on object sizes

## Contributing

Feel free to contribute to the SAHI integration by:
1. Reporting issues
2. Suggesting improvements
3. Submitting pull requests

## License

This SAHI integration follows the same license as the original YOLOv9 project. 