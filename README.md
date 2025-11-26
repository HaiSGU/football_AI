<div align="center">

  <h1>sports</h1>

[notebooks](https://github.com/roboflow/notebooks) | [inference](https://github.com/roboflow/inference) | [autodistill](https://github.com/autodistill/autodistill) | [maestro](https://github.com/roboflow/multimodal-maestro)

</div>

## 👋 hello

In sports, every centimeter and every second matter. That's why Roboflow decided to use sports as a testing ground to push our object detection, image segmentation, keypoint detection, and foundational models to their limits. This repository contains reusable tools that can be applied in sports and beyond.

## 🥵 challenges

Are you also a fan of computer vision and sports?  We welcome contributions from anyone who shares our passion! Together, we can build powerful open-source tools for sports analytics. Here are the main challenges we're looking to tackle:

- **Ball tracking:** Tracking the ball is extremely difficult due to its small size and rapid movements, especially in high-resolution videos.
- **Reading jersey numbers:** Accurately reading player jersey numbers is often hampered by blurry videos, players turning away, or other objects obscuring the numbers.
- **Player tracking:** Maintaining consistent player identification throughout a game is a challenge due to frequent occlusions caused by other players or objects on the field.
- **Player re-identification:** Re-identifying players who have left and re-entered the frame is tricky, especially with moving cameras or when players are visually similar.
- **Camera calibration:** Accurately calibrating camera views is crucial for extracting advanced statistics like player speed and distance traveled. This is a complex task due to the dynamic nature of sports and varying camera angles.

## 💻 install

We don't have a Python package yet. Install from source in a
[**Python>=3.8**](https://www.python.org/) environment.

### Quick Install

```bash
pip install git+https://github.com/roboflow/sports.git
```

### Windows Users

This project is **fully compatible with Windows 10/11**! For detailed Windows setup instructions including:
- GPU acceleration with CUDA
- OpenCV configuration
- Troubleshooting common issues

See the [Soccer AI README](examples/soccer/README.md) for complete Windows installation guide.

**Quick Windows Setup:**
```cmd
pip install git+https://github.com/roboflow/sports.git
cd examples\soccer
pip install -r requirements.txt gdown
pip uninstall opencv-python-headless -y
pip install opencv-python==4.10.0.84
.\setup.ps1  # or setup.bat
```

## ⚽ datasets

| use case                               | dataset                                                                                                                                                           |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ⚽ soccer player detection              | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)  |
| ⚽ soccer ball detection                | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)     |
| ⚽ soccer pitch keypoint detection      | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)    |
| 🏀 basketball court keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2)      |
| 🏀 basketball jersey numbers ocr       | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr)     |


Visit [Roboflow Universe](https://universe.roboflow.com/) and explore other sport-related datasets.

## 🔥 demos

https://github.com/roboflow/sports/assets/26109316/7ad414dd-cc4e-476d-9af3-02dfdf029205

## 🏆 contribution

We love your input! [Let us know](https://github.com/roboflow/sports/issues) what else we should build!
