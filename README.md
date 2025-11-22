<p align="center">
  <img src="https://i.imgur.com/vVl8U0q.png" alt="Hierarchy Tree Logo" width="300" height="300"/>
</p>

Automatically detect and remove artist watermarks/logos from images using AI-based object detection and inpainting. This tool provides a completely hands-free, scalable solution for batch processing images.

## ✨ Features

- **Automatic Detection**: Uses YOLOv11 to automatically locate logos/watermarks on images
- **AI-Powered Removal**: Uses IOPaint with MAT or LaMa models for seamless logo removal
- **Batch Processing**: Process entire folders of images automatically
- **GPU Acceleration**: Supports CUDA for faster processing (falls back to CPU if unavailable)
- **Preserves Quality**: Maintains original image dimensions, format, and quality
- **Offline & Free**: Uses open-source models, no API keys or internet required (after setup)
- **Smart Masking**: Automatically expands detection regions for complete logo coverage

## 👣 Quick Start

Get up and running in 5 minutes!

### Step 1: Install Python Dependencies

```rb
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS

# Install packages
pip install -r requirements.txt
```

### Step 2: Download the Model

```rb
python setup_models.py
```

This downloads the YOLOv11 model (~250MB) for logo detection.

### Step 3: Process Your Images

```rb
python remove_logos.py -i /path/to/your/images
```

That's it! Your processed images will be in `/path/to/your/images/cleaned/`

**Common options:**
- Use verbose mode: `python remove_logos.py -i /path/to/images -v`
- Use LaMa model instead of MAT: `python remove_logos.py -i /path/to/images --inpaint-model lama`
- Lower confidence for more detections: `python remove_logos.py -i /path/to/images -c 0.15`
- Increase mask expansion: `python remove_logos.py -i /path/to/images -e 25`

For detailed instructions, see the sections below.

---

## 🛠️ Prerequisites

- **Operating System**: Linux (tested on Linux Mint), macOS, or Windows
- **Python**: Version 3.8 or higher
- **Storage**: ~500MB for models
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing

## 🎛️ Setup

### 1. Clone or Download This Repository

```rb
cd /path/to/your/projects
git clone <repository-url>
cd logo-removal
```

Or simply download and extract the files to a folder.

### 2. Create a Virtual Environment (Recommended)

```rb
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```rb
# Install required packages
pip install -r requirements.txt
```

**For GPU Support (NVIDIA CUDA):**

If you have an NVIDIA GPU and want faster processing:

```rb
# Install PyTorch with CUDA support (adjust cu118 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download the YOLO Model

Run the setup script to download the pre-trained logo detection model:

```rb
python setup_models.py
```

This will download the YOLOv11 model (~250MB) to the `models/` directory.

**Alternative Manual Download:**

If the script fails, you can manually download the model:

1. Visit: https://huggingface.co/corzent/yolo11x_watermark_detection
2. Download `best.pt`
3. Create a `models/` directory in the project folder
4. Place the downloaded file in `models/best.pt`

## 🚀 Usage

### Basic Usage

Process all images in a folder:

```rb
python remove_logos.py -i /path/to/images/with/logos
```

This will:
- Scan `/path/to/images/with/logos` for images
- Detect and remove logos from each image
- Save results to `/path/to/images/with/logos/cleaned/`

### Advanced Options

```rb
# Specify custom output directory
python remove_logos.py -i /path/to/input -o /path/to/output

# Use a custom YOLO model
python remove_logos.py -i /path/to/images -m /path/to/custom_model.pt

# Adjust detection confidence threshold (0-1)
# Lower = more detections (may include false positives)
# Higher = fewer detections (may miss some logos)
python remove_logos.py -i /path/to/images -c 0.5

# Adjust mask expansion (pixels to expand around detected logo)
# Increase if logo edges are still visible after removal
python remove_logos.py -i /path/to/images -e 20

# Choose inpainting model (mat or lama)
# MAT: Better for large masks and anime images (default)
# LaMa: Better for photographic images
python remove_logos.py -i /path/to/images --inpaint-model lama

# Force CPU usage (disable GPU)
python remove_logos.py -i /path/to/images -d cpu

# Enable verbose logging
python remove_logos.py -i /path/to/images -v
```

### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input directory with images (required) | - |
| `--output` | `-o` | Output directory for processed images | `INPUT_DIR/cleaned` |
| `--model` | `-m` | Path to YOLO model weights | `models/best.pt` |
| `--confidence` | `-c` | Detection confidence threshold (0-1) | `0.25` |
| `--expansion` | `-e` | Mask expansion in pixels | `15` |
| `--device` | `-d` | Device to use (auto/cpu/cuda) | `auto` |
| `--inpaint-model` | - | Inpainting model (mat/lama) | `mat` |
| `--verbose` | `-v` | Enable verbose logging | `False` |

### Example Workflow

```rb
# 1. Activate virtual environment
source venv/bin/activate

# 2. Process images
python remove_logos.py -i ~/Pictures/anime_images

# 3. Check the results
ls ~/Pictures/anime_images/cleaned/

# 4. View the log for any issues
cat log/logo_removal.log
```

## 📁 Project Structure

```
logo-removal/
├── remove_logos.py          # Main script for logo removal
├── setup_models.py          # Script to download YOLO model
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── models/                  # Model files (created by setup_models.py)
│   └── best.pt
└── log/logo_removal.log     # Processing log (created when script runs)
```

## 🔧 How It Works

1. **Detection Phase**
   - YOLOv11 scans each image to locate logos/watermarks
   - Returns bounding box coordinates for each detected logo

2. **Masking Phase**
   - Creates a binary mask for each detected logo region
   - Expands the mask by 15 pixels (configurable) to ensure complete coverage
   - White pixels (255) indicate areas to remove, black (0) is preserved

3. **Inpainting Phase**
   - IOPaint inpainting model (MAT or LaMa) analyzes the surrounding context
   - Intelligently fills in the masked region
   - Seamlessly blends the inpainted area with the original image

4. **Output**
   - Saves the processed image with the same filename
   - Maintains original format, dimensions, and quality
   - Skips already processed files

## 🎨 Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP

## ⚡ Performance Tips

1. **Use GPU**: Processing is 5-10x faster with CUDA-enabled GPU
2. **Batch Size**: Process large folders in chunks if memory is limited
3. **Mask Expansion**: Start with default (15px), increase if logo remnants remain
4. **Confidence Threshold**: Lower for subtle logos, raise to reduce false detections

## 🐛 Troubleshooting

### No logos detected

- **Increase confidence threshold**: Try `-c 0.15` for more sensitive detection
- **Check image quality**: Very low resolution images may not detect well
- **Logo type**: The model is trained on common watermark styles; unusual logos may not be detected

### Logo not fully removed

- **Increase mask expansion**: Try `-e 20` or `-e 25`
- **Check detection**: Use verbose mode `-v` to see if logo was detected correctly

### Out of memory errors

- **Force CPU**: Use `-d cpu` (slower but uses less memory)
- **Process fewer images**: Split large folders into smaller batches
- **Close other applications**: Free up RAM before processing

### Model download fails

- **Manual download**: Follow the alternative download instructions above
- **Check internet**: Ensure stable connection to Hugging Face
- **Firewall**: Check if downloads from huggingface.co are blocked

### Poor inpainting quality

- Try switching inpainting models:
  - **MAT** (default): Better for anime/stylized images and large masks
  - **LaMa**: Better for photographic/realistic images
  - Example: `python remove_logos.py -i /path/to/images --inpaint-model lama`
- Inpainting works best when:
  - The logo is on a relatively uniform background
  - The logo doesn't cover critical details
  - The surrounding context provides clear patterns to continue

## 📊 Example Results

**Before Processing:**
- Image with logo in corner
- Watermark text overlay

**After Processing:**
- Clean image with logo seamlessly removed
- Background/artwork continued naturally

## 🔒 Privacy & Security

- **Fully Offline**: After initial model download, no internet required
- **No Telemetry**: No data is sent anywhere
- **Local Processing**: All processing happens on your machine

## 📝 Logging

The script creates `log/logo_removal.log` with detailed information:
- Which images were processed
- Number of logos detected per image
- Any errors or warnings
- Processing statistics

View the log:
```rb
cat log/logo_removal.log
# or
tail -f log/logo_removal.log  # Follow in real-time
```

## 🎭 Contributing

Contributions are welcome! Here are ways you can help:

### Reporting Issues

If you encounter bugs or have suggestions:

1. Check existing issues first
2. Provide detailed information:
   - Operating system and Python version
   - Full error message and log output
   - Steps to reproduce
   - Sample image (if possible)

### Feature Requests

Have an idea? Open an issue with:
- Clear description of the feature
- Use case and benefits
- Any implementation ideas

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Improving Documentation

- Fix typos or unclear instructions
- Add examples or use cases
- Translate to other languages

### Testing

- Test on different operating systems
- Try various image types and sizes
- Report compatibility issues

## 📄 License

This project uses open-source models and libraries:

- **YOLOv11**: AGPL-3.0 (Ultralytics)
- **LaMa**: Apache-2.0 (Samsung Research)
- **OpenCV**: Apache-2.0
- **PyTorch**: BSD-style license

Please ensure compliance with respective licenses when using this tool.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv11 framework
- **Samsung Research** for the LaMa inpainting model
- **foduucom** for the watermark detection model on Hugging Face
- The open-source community for the amazing tools and libraries

## 📧 Support

For questions, issues, or suggestions:

1. Check this README first
2. Review the troubleshooting section
3. Check the log file for errors
4. Open an issue with detailed information

---

**Happy Logo Removing!** 🎨✨

---

## ⚖️ Legal Disclaimer

This tool is provided for educational and research purposes only. The user assumes all responsibility for its use. This project is not intended to support, encourage, or facilitate the removal of official trademarks, copyrights, or other intellectual property from images for purposes of piracy, infringement, or any other illegal activities.

By using this software, you acknowledge and agree that:
- You will comply with all applicable local, national, and international laws and regulations.
- You will not use this tool to infringe upon the rights of any third party, including intellectual property rights.
- You are solely responsible for ensuring that your use of this tool is lawful and does not violate any terms of service or intellectual property rights.

The creators and contributors of this repository disclaim any and all liability for any misuse of this software and for any direct, indirect, incidental, special, exemplary, or consequential damages arising out of or in connection with the use of this tool.

---

