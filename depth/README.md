# 📏 RailAI-Vision: Depth Fusion Module

This module performs real-time or offline fusion of **object detection**, **railway track segmentation**, and **depth estimation** using **ZED2 stereo camera** or **MiDaS monocular model**. It is a core component of the `RailAI-Vision` framework for enhancing railway environment perception and safety.

---

## 📁 Folder Structure
```
depth/
├── scripts/
│ ├── run_depth_fusion_zed.py       # Real-time fusion using ZED2
│ └── run_depth_fusion_midas.py     # Offline fusion using MiDaS
├── engine.py                       # Fusion pipeline logic
├── models/
│ ├── loader.py                     # Model loaders (YOLOv11, DeepLabV3+, MiDaS)
│ └── utils.py                      # Depth utility functions
├── utils/
│ └── visualization.py              # Drawing, overlaying, and display utilities
├── results/                        # Evaluation outputs: confusion matrices, prediction samples, metrics
├── requirements.txt                # Dependencies for the detection module
└── README.md                       # This documentation file                     # Documentation and usage guide
```

---

## 🚀 Functionality Overview

| Feature                   | ZED2 Mode (`run_depth_fusion_zed.py`) | MiDaS Mode (`run_depth_fusion_midas.py`) |
|---------------------------|---------------------------------------|------------------------------------------|
| Real-time Inference       | ✅                                     | ❌ (offline only)                         |
| Monocular Depth Support   | ❌                                     | ✅                                        |
| Stereo Depth Support      | ✅                                     | ❌                                        |
| Detection (YOLOv11)       | ✅                                     | ✅                                        |
| Segmentation (DeepLabV3+) | ✅                                     | ✅                                        |
| Distance Calibration      | ✅                                     | ✅ (based on track gauge)                 |
| Track-aware Object Fusion | ✅                                     | ✅                                        |
| Output Visualization      | ✅ Live Display                        | ✅ Saved PNGs                             |

---

## 📦 Dependencies

Make sure the following libraries are installed:

```bash
pip install opencv-python torch torchvision ultralytics
pip install pyzed==4.0  # Only for ZED2 mode
```

Also ensure the ZED SDK is properly installed for real-time camera usage.

---

## 🧠 Models Used

| Model      | Purpose                 | Source                            |
|------------|-------------------------|-----------------------------------|
| YOLOv11    | Object detection        | `weights/yolo11n.pt`              |
| DeepLabV3+ | Rail track segmentation | Custom `.pth` file                |
| MiDaS      | Monocular depth         | `intel-isl/MiDaS` via `torch.hub` |

All models should be stored in the `weights/` directory.

---

## 🧪 Usage
**1.** 📷 Real-Time Depth Fusion (ZED2 Camera)
- Requires ZED2 device connected

```bash
cd depth
python scripts/run_depth_fusion_zed.py
```
- Press `q` to exit.
- Live detection, segmentation, and depth measurement will be displayed.

**2.** 🖼️ Offline Depth Fusion (MiDaS)
- Uses MiDaS for monocular depth estimation.

```bash
cd depth
python scripts/run_depth_fusion_midas.py
```
- Input folder: `images/`
- Output folder: `outputs/`
- Results saved as `.png` files with visualized fusion.

---

## 📚 Configuration
- Track Class ID: Set in both scripts (default: 12 for RailSem19)
- Detection Threshold: Default is 0.4
- Depth Calibration: Uses rail width (1.435m) to scale MiDaS depth
- Visual Overlay:
  - Track mask → Yellow
  - Detected objects:
    - Red box: off-track
    - Green box: intersects track (with distance annotation)

---

## 🛠️ Developer Notes
- `engine.py`: Implements `fuse_outputs()` — the main fusion logic reused across scripts.
- `models/loader.py`: Clean modular loaders for YOLO, DeepLabV3+, and MiDaS.
- `models/utils.py`: Core helpers for bounding box depth and calibration.
- `utils/visualization.py`: Draw boxes, labels, overlays with flexibility.

---

## 📚 Acknowledgments

- [Torchvision Semantic Segmentation](https://github.com/qubvel-org/segmentation_models.pytorch)
- [Intel ISL MiDaS](https://github.com/isl-org/MiDaS)
- [ZED SDK](https://github.com/stereolabs/zed-sdk)
- [RailSem19 Dataset](https://www.wilddash.cc/railsem19)


---


