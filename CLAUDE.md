# Claude Code Instructions for Logo Removal Repository

This document provides guidance for Claude Code when working on this repository.

## Repository Overview

This is an AI-powered logo/watermark removal tool designed specifically for batch processing anime and stylized images. The tool uses:
- **YOLOv11** for automatic logo detection
- **OWLv2** for watermark presence classification
- **IOPaint** with MAT/LaMa models for inpainting

## Key Architecture Decisions

### Detection Pipeline
1. **OWLv2 Classifier First**: Binary classifier determines if watermark exists (faster, prevents false negatives)
2. **YOLO Localization**: Only runs if OWLv2 detects a watermark, provides bounding box coordinates
3. **Mask Expansion**: Expands bounding boxes by 15px (default) to ensure complete logo coverage

### Inpainting Strategy
- **Default Model**: MAT (Mask-Aware Transformer) - optimized for anime/stylized images and large masks
- **Alternative**: LaMa (Large Mask Inpainting) - better for photographic/realistic images
- **Implementation**: Subprocess calls to IOPaint CLI (not in-memory) for model flexibility

### Performance Considerations
- YOLO parameters tuned for anime images: `imgsz=1024`, `augment=True`, `iou=0.5`
- GPU-first approach with CPU fallback
- Batch processing with per-image error handling (one failure doesn't stop the batch)

## Models and Dependencies

### Required Models
1. **YOLO Detection Model** (`models/best.pt`, 110MB)
   - Source: https://huggingface.co/corzent/yolo11x_watermark_detection
   - Downloaded via `setup_models.py`

2. **OWLv2 Classifier** (`models/far5y1y5-8000.pt`, 342MB)
   - Source: https://huggingface.co/fancyfeast/joycaption-watermark-detection
   - Downloaded automatically on first run

3. **IOPaint Models** (auto-downloaded on first use)
   - MAT model (~1GB, cached in `~/.cache`)
   - LaMa model (~200MB, cached in `~/.cache`)

### Key Dependencies
- `ultralytics>=8.0.0` - YOLO framework
- `iopaint` - Inpainting models (MAT, LaMa)
- `transformers>=4.30.0` - OWLv2 model
- `torch>=2.0.0` + `torchvision>=0.15.0` - Deep learning backend
- `opencv-python>=4.8.0` - Image processing

## Code Structure

### Main Script: `remove_logos.py`

**Class: `DetectorModelOwl`** (lines 25-80)
- OWLv2-based binary watermark classifier
- Frozen vision model + trainable classification head
- Returns confidence score (0-1) for watermark presence

**Class: `LogoRemover`** (lines 82-455)
- `__init__`: Loads YOLO, OWLv2, sets up device
- `detect_logos()`: Runs YOLO detection with optimized parameters
- `create_mask()`: Generates binary mask from bounding boxes with expansion
- `inpaint_image()`: Calls IOPaint via subprocess
- `process_image()`: Orchestrates OWLv2 → YOLO → Mask → Inpaint pipeline
- `process_directory()`: Batch processing with error handling

### Important Code Patterns

**Detection Parameters** (line 142-147):
```python
results = self.model(
    image_path,
    conf=self.confidence_threshold,
    imgsz=1024,      # Critical for anime images
    augment=True,    # Better detection quality
    iou=0.5,         # IoU threshold for NMS
    verbose=False
)
```

**IOPaint Subprocess Call** (lines 274-286):
```python
cmd = [
    'iopaint', 'run',
    '--model', self.inpaint_model,  # 'mat' or 'lama'
    '--device', device_param,        # 'cuda' or 'cpu'
    '--image', tmp_image_path,
    '--mask', tmp_mask_path,
    '--output', tmp_output_dir
]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
```

**Output Path Handling** (line 289):
```python
# IOPaint creates subdirectory and uses input filename
result_image_path = os.path.join(tmp_output_dir, 'input.png')
```

## Development Guidelines

### Testing
- Always test on the `pictures/` folder (5 anime test images included)
- Expected detection rate: 4/5 images (IMG_0968 is difficult)
- Compare outputs visually - inpainting quality is subjective

### Adding Features
1. **Detection improvements**: Modify YOLO parameters in `detect_logos()` method
2. **New inpainting models**: Add to IOPaint `--model` choices and update CLI args
3. **Post-processing**: Add between inpainting and final save in `process_image()`

### Common Modifications

**Adjusting Detection Sensitivity**:
- Lower `confidence_threshold` (default 0.25) for more detections
- Increase `mask_expansion` (default 15px) if logo edges remain visible
- Modify `iou` parameter for overlapping logo handling

**Changing Default Model**:
- Edit line 94: `inpaint_model: str = 'mat'` to `'lama'`
- Or update CLI default at line 512: `default='mat'`

**Adding Logging**:
- Use existing logger: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`
- Configured at line 20-21 with file handler

### Known Limitations

1. **IMG_0968 Detection Failure**:
   - OWLv2 detects watermark (confidence 0.52)
   - YOLO cannot localize it (too subtle/small)
   - Solution: Needs better YOLO model or lower confidence threshold

2. **Large Images**:
   - YOLO `imgsz=1024` may downsample very large images
   - Consider adding dynamic `imgsz` based on input dimensions

3. **Pillow Version Conflict**:
   - IOPaint requires Pillow 9.5.0
   - Non-critical warning can be ignored

## Testing Checklist

When making changes, verify:
- [ ] All 5 test images process without errors
- [ ] At least 4/5 logos detected (IMG_0968 may fail)
- [ ] Inpainting quality is clean (no blur/artifacts)
- [ ] Both `--inpaint-model mat` and `--inpaint-model lama` work
- [ ] CPU fallback works when CUDA unavailable
- [ ] Output images match input dimensions and format
- [ ] `log/logo_removal.log` shows no unexpected errors

## Useful Commands

```bash
# Clean test run
rm -rf pictures/no_logos && python remove_logos.py -i pictures -v

# Test with LaMa model
python remove_logos.py -i pictures --inpaint-model lama -v

# Test CPU-only mode
python remove_logos.py -i pictures -d cpu -v

# Check model files
ls -lh models/

# View recent logs
tail -50 log/logo_removal.log
```

## Git Workflow

This repository is synced to Google Drive via rclone bisync (every 10 minutes). Changes will automatically sync bidirectionally.

### Committing Changes
- Avoid committing `venv/`, `pictures/no_logos/`, `*.log`, `__pycache__/` (already in `.gitignore`)
- Model files in `models/` should be committed (tracked for reproducibility)
- Always test before committing changes to `remove_logos.py`

## Future Improvements

Potential enhancements to consider:
1. Add support for more YOLO models via config file
2. Implement progress bars for batch processing (using `tqdm`)
3. Add dry-run mode to preview detections without inpainting
4. Support video file processing (frame extraction + batch processing)
5. Create web UI using Gradio for non-technical users
6. Add automatic mask refinement using edge detection
7. Implement ensemble detection (multiple YOLO models voting)
8. Add support for manual mask editing (interactive mode)

## References

- **JoyCaption Watermark Detection**: https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection
  - Source of OWLv2 classifier approach and YOLO parameters
- **IOPaint Documentation**: https://github.com/Sanster/IOPaint
  - MAT and LaMa model integration guide
- **Ultralytics YOLO**: https://docs.ultralytics.com/
  - YOLO detection parameter reference
