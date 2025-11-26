# Soccer AI ⚽

AI-powered soccer video analysis using YOLOv8 for player detection, tracking, and team classification.

**⚡ Quick Start:** [Jump to Quick Commands](#-quick-commands)

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Commands](#-quick-commands)
- [GPU Setup](#-gpu-setup--recommended)
- [Usage](#️-usage)
- [Batch Processing](#-batch-processing-all-modes)
- [Performance](#-performance)
- [Troubleshooting](#️-troubleshooting)
- [Project Structure](#-project-structure)

---

## 💻 Installation

### Prerequisites
- **Python 3.8+** (Python 3.10 recommended)
- **Windows 10/11** (64-bit) or Linux/macOS
- **NVIDIA GPU** (optional, but **highly recommended** for 60-120x faster processing)

### Step-by-Step Setup

```bash
# 1. Install from source
pip install git+https://github.com/roboflow/sports.git
cd examples/soccer

# 2. Install dependencies
pip install -r requirements.txt gdown

# 3. Fix OpenCV (Windows only)
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84

# 4. Download models & videos
# Windows PowerShell:
.\setup.ps1
# Windows CMD:
setup.bat
# Linux/macOS:
./setup.sh
```

---

## ⚡ Quick Commands

### Run Single Mode (GPU)
```cmd
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\result.mp4 --device cuda --mode PLAYER_DETECTION --no-display
```

### Run All Modes Automatically
```powershell
# PowerShell (recommended)
.\run_all_gpu.ps1

# CMD
run_all_gpu.bat
```

**Time:** ~4-5 minutes (GPU) vs ~8-10 hours (CPU)

---

## 🚀 GPU Setup (Recommended)

### 1. Check GPU
```cmd
nvidia-smi
```

### 2. Install PyTorch with CUDA
```cmd
# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify CUDA
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:** `CUDA: True`

---

## 🛠️ Usage

### Available Modes

| Mode | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| `PITCH_DETECTION` | Detect field boundaries | ~30s | ~60min |
| `PLAYER_DETECTION` | Detect players, ball, referees | ~30s | ~60min |
| `BALL_DETECTION` | Track ball movement | ~40s | ~90min |
| `PLAYER_TRACKING` | Track players with IDs | ~35s | ~70min |
| `TEAM_CLASSIFICATION` | Classify players by team | ~50s | ~120min |
| `RADAR` | All features + radar view | ~60s | ~150min |

### Command Examples

#### PLAYER_DETECTION (Most Common)
```cmd
python main.py ^
  --source_video_path data\0bfacc_0.mp4 ^
  --target_video_path output\player_detection.mp4 ^
  --device cuda ^
  --mode PLAYER_DETECTION ^
  --no-display
```

#### TEAM_CLASSIFICATION
```cmd
python main.py ^
  --source_video_path data\0bfacc_0.mp4 ^
  --target_video_path output\team_classification.mp4 ^
  --device cuda ^
  --mode TEAM_CLASSIFICATION ^
  --no-display
```

#### RADAR (All Features)
```cmd
python main.py ^
  --source_video_path data\0bfacc_0.mp4 ^
  --target_video_path output\radar.mp4 ^
  --device cuda ^
  --mode RADAR ^
  --no-display
```

### Command Options

| Option | Values | Description |
|--------|--------|-------------|
| `--source_video_path` | Path | Input video file |
| `--target_video_path` | Path | Output video file |
| `--device` | `cpu` or `cuda` | Processing device |
| `--mode` | See modes above | Analysis mode |
| `--no-display` | Flag | Disable GUI (recommended) |

---

## 🔥 Batch Processing (All Modes)

### Automatic Script

Run all 6 modes with a single command:

**PowerShell:**
```powershell
.\run_all_gpu.ps1
```

**CMD:**
```cmd
run_all_gpu.bat
```

### What It Does

1. ✅ Checks CUDA availability
2. ✅ Processes all 6 modes sequentially
3. ✅ Shows progress for each mode
4. ✅ Generates 6 output videos:
   - `pitch_detection.mp4`
   - `player_detection.mp4`
   - `ball_detection.mp4`
   - `player_tracking.mp4`
   - `team_classification.mp4`
   - `radar.mp4`
5. ✅ Displays summary with timing

**Total Time:** ~4-5 minutes (GPU) vs ~8-10 hours (CPU)

---

## ⚡ Performance

### Speed Comparison (30s video, 1920x1080, 30fps)

| Device | Speed | Total Time (6 modes) | Speedup |
|--------|-------|---------------------|---------|
| 🐌 CPU | ~0.5 fps | ~8-10 hours | 1x |
| 🚀 GPU (GTX 1650) | ~30-60 fps | ~4-5 minutes | **120x** |
| 🚀 GPU (RTX 3080) | ~100+ fps | ~2 minutes | **240x** |

**Recommendation:** Always use `--device cuda` if you have NVIDIA GPU!

---

## ⚠️ Troubleshooting

### OpenCV Error
```
AttributeError: module 'cv2' has no attribute 'FONT_HERSHEY_SIMPLEX'
```

**Solution:**
```cmd
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python==4.10.0.84
```

---

### CUDA Not Available
```
CUDA available: False
```

**Solution:**
1. Check GPU: `nvidia-smi`
2. Reinstall PyTorch:
   ```cmd
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Close other GPU applications
- Use CPU: `--device cpu`

---

### Output Only 1 Frame

**Cause:** Pressed Q key during processing

**Solution:** Use `--no-display` flag

---

### Slow Processing Despite GPU

**Check if GPU is being used:**
```cmd
# Run this while processing:
nvidia-smi
```

If GPU Usage = 0%, ensure you're using `--device cuda`

---

## 📁 Project Structure

```
soccer/
├── 📄 README.md              # This file
├── 📄 main.py                # Main script
├── 📄 setup.bat              # Windows CMD setup
├── 📄 setup.ps1              # Windows PowerShell setup
├── 📄 setup.sh               # Linux/macOS setup
├── 📄 run_all_gpu.bat        # Batch processing (CMD)
├── 📄 run_all_gpu.ps1        # Batch processing (PowerShell)
├── 📄 requirements.txt       # Python dependencies
├── 📁 data/                  # Models & videos (created by setup)
│   ├── *.pt                 # YOLO models
│   └── *.mp4                # Sample videos
├── 📁 output/                # Output videos (auto-created)
└── 📁 notebooks/             # Training notebooks
```

---

## 🎯 Complete Workflow Example

```cmd
# 1. First-time setup (one time only)
pip install -r requirements.txt gdown
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84
.\setup.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Run all modes
.\run_all_gpu.ps1

# 4. View results
# Open output folder and play the 6 generated videos
```

---

## 💡 Tips & Best Practices

1. **Always use GPU** - 60-120x faster than CPU
2. **Use `--no-display`** - Faster and prevents accidental interruption
3. **Batch processing** - Use `run_all_gpu.ps1` to process all modes at once
4. **Close other apps** - Free up GPU memory for better performance
5. **Check progress** - Progress bar shows FPS, elapsed time, and ETA

---

## 📊 Features

- ✅ **6 Analysis Modes** - From basic detection to full radar view
- ✅ **GPU Acceleration** - CUDA support for NVIDIA GPUs
- ✅ **Progress Bar** - Real-time progress with ETA
- ✅ **Batch Processing** - Run all modes with one command
- ✅ **No-Display Mode** - Process without GUI
- ✅ **Auto Output Directory** - Creates folders automatically
- ✅ **Windows Compatible** - Full Windows 10/11 support
- ✅ **Cross-Platform** - Works on Windows, Linux, macOS

---

## 🗺️ Roadmap

- [ ] Add smoothing to eliminate flickering in RADAR mode
- [ ] Add notebook for offline data analysis
- [x] Windows compatibility
- [x] GPU acceleration support
- [x] Progress bar with ETA
- [x] Batch processing scripts
- [x] No-display mode

---

## ⚽ Datasets

Original data from [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout).

| Use Case | Dataset | Train Model |
|:---------|:--------|:------------|
| Soccer player detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb) |
| Soccer ball detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb) |
| Soccer pitch detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_pitch_keypoint_detector.ipynb) |

---

## 🤖 Models

- **[YOLOv8](https://docs.ultralytics.com/models/yolov8/)** - Player & ball detection
- **[SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip)** - Feature extraction
- **[UMAP](https://umap-learn.readthedocs.io/en/latest/)** - Dimensionality reduction
- **[KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)** - Team classification

---

## © License

- **ultralytics**: [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **sports**: [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md)

---

## 🙏 Acknowledgments

- Data: [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
- Built with: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Powered by: [Supervision](https://github.com/roboflow/supervision)

---

**Made with ⚽ by [Roboflow](https://roboflow.com)**
