<div align="center">

  <h1>⚽ Sports AI Analytics</h1>
  
  <p>
    <strong>Advanced Computer Vision Tools for Sports Analytics</strong>
  </p>

  <p>
    <a href="https://github.com/roboflow/notebooks">notebooks</a> •
    <a href="https://github.com/roboflow/inference">inference</a> •
    <a href="https://github.com/autodistill/autodistill">autodistill</a> •
    <a href="https://github.com/roboflow/multimodal-maestro">maestro</a>
  </p>

</div>

---

## 📖 Overview

In sports, every centimeter and every second matter. This repository provides state-of-the-art computer vision tools specifically designed for sports analytics, pushing the boundaries of object detection, image segmentation, keypoint detection, and foundational models to their limits.

**Sports AI Analytics** offers reusable, production-ready components that can be applied across various sports and beyond, enabling coaches, analysts, and enthusiasts to extract meaningful insights from video footage.

## ✨ Features

- **🎯 Player Detection & Tracking**: Real-time detection and tracking of players with team classification
- **⚽ Ball Trajectory Analysis**: Advanced ball tracking with trajectory visualization and prediction
- **🗺️ Tactical Radar View**: Top-down tactical view with player positioning and movement analysis
- **📊 Advanced Statistics**: Speed, distance, possession, and heatmap analytics
- **🎥 Multi-Camera Support**: Calibration and synchronization for multiple camera angles
- **🔄 Real-time Processing**: Optimized for live game analysis and instant replay

## 🎬 Demo Videos

### Player Radar & Tactical View
Real-time tactical analysis with player positioning on a mini-map:

https://github.com/user-attachments/assets/radar_clean.mp4

### Ball Trajectory Tracking
Advanced ball tracking with trajectory visualization:

https://github.com/user-attachments/assets/ball_trajectory.mp4

> **Note**: For full demo videos, check the `examples/soccer/output/` directory after running the examples.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time processing)
- 8GB+ RAM

### Installation

#### Standard Installation

```bash
# Clone the repository
git clone https://github.com/roboflow/sports.git
cd sports

# Install the package
pip install git+https://github.com/roboflow/sports.git
```

#### Windows Installation

This project is **fully compatible with Windows 10/11** with GPU acceleration support!

```cmd
# Install the package
pip install git+https://github.com/roboflow/sports.git

# Navigate to soccer example
cd examples\soccer

# Install dependencies
pip install -r requirements.txt gdown

# Fix OpenCV conflicts
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84

# Run setup script
.\setup.ps1  # PowerShell
# or
setup.bat    # Command Prompt
```

For detailed Windows setup instructions including GPU acceleration, OpenCV configuration, and troubleshooting, see the [Soccer AI README](examples/soccer/README.md).

### Running Your First Analysis

```bash
cd examples/soccer
python main.py --mode PLAYER_DETECTION --input your_video.mp4
```

## 🥵 Technical Challenges

We're tackling some of the most difficult problems in sports computer vision:

- **⚽ Ball Tracking**: Tracking small, fast-moving objects in high-resolution video with motion blur and occlusions
- **🔢 Jersey Number Recognition**: OCR on blurry, rotated, or partially obscured jersey numbers
- **👥 Player Tracking**: Maintaining consistent player IDs through occlusions and similar appearances
- **🔄 Player Re-identification**: Re-identifying players who exit and re-enter the frame
- **📐 Camera Calibration**: Accurate calibration for extracting real-world metrics (speed, distance, positioning)
- **⚡ Real-time Processing**: Optimizing complex pipelines for live game analysis

## 📊 Datasets

High-quality, annotated datasets for training and evaluation:

| Sport | Use Case | Dataset |
|:------|:---------|:--------|
| ⚽ Soccer | Player Detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| ⚽ Soccer | Ball Detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) |
| ⚽ Soccer | Pitch Keypoint Detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) |
| 🏀 Basketball | Court Keypoint Detection | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) |
| 🏀 Basketball | Jersey Number OCR | [![Download](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr) |

Explore more sport-related datasets on [Roboflow Universe](https://universe.roboflow.com/).

## 📚 Examples

### Soccer Analysis

Comprehensive soccer analysis with multiple modes:

```bash
cd examples/soccer

# Player detection and tracking
python main.py --mode PLAYER_DETECTION --input game.mp4

# Tactical radar view
python main.py --mode RADAR --input game.mp4

# Ball trajectory analysis
python main.py --mode BALL_TRAJECTORY --input game.mp4

# Full analysis with all features
python main.py --mode FULL --input game.mp4
```

See [examples/soccer/README.md](examples/soccer/README.md) for detailed documentation.

## 🛠️ Development

### Project Structure

```
sports/
├── examples/
│   └── soccer/          # Soccer analysis example
│       ├── main.py      # Main entry point
│       ├── config/      # Configuration files
│       └── output/      # Generated videos and data
├── sports/              # Core library
│   ├── annotators/      # Visualization tools
│   ├── common/          # Shared utilities
│   └── configs/         # Configuration schemas
└── tests/               # Unit tests
```

### Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs**: Open an issue with detailed reproduction steps
2. **💡 Suggest Features**: Share your ideas for new capabilities
3. **🔧 Submit PRs**: Fix bugs, add features, or improve documentation
4. **📝 Improve Docs**: Help make our documentation clearer
5. **🎓 Share Examples**: Contribute new use cases and examples

See our [contribution guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ by [Roboflow](https://roboflow.com/) and the open-source community.

Special thanks to all contributors who have helped push the boundaries of sports analytics!

## 📞 Support

- **📧 Issues**: [GitHub Issues](https://github.com/roboflow/sports/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/roboflow/sports/discussions)
- **🌐 Website**: [roboflow.com](https://roboflow.com/)
- **📖 Documentation**: [Roboflow Docs](https://docs.roboflow.com/)

---

<div align="center">
  <p>Made with ⚽ by the Roboflow team</p>
  <p>
    <a href="https://github.com/roboflow/sports/stargazers">⭐ Star us on GitHub</a> •
    <a href="https://twitter.com/roboflow">🐦 Follow on Twitter</a>
  </p>
</div>
