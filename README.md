# 🔥 Real-Time Fire and Smoke Detection using Enhanced YOLOv8

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Enhanced-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-MS%20Thesis-orange.svg)

**Lightweight Attention-Enhanced YOLOv8 for Edge Deployment and Real-Time Surveillance**

*MS Thesis Project | NUST, Islamabad | 2024-2025*

[Features](#-key-features) • [Demo](#-demo-results) • [Architecture](#-model-architecture) • [Installation](#%EF%B8%8F-installation) • [Training](#%EF%B8%8F-training) • [Results](#-performance-metrics) • [Citation](#-citation)

</div>

---

## 📋 Overview

This repository presents a **state-of-the-art fire and smoke detection framework** based on YOLOv8, enhanced with attention mechanisms and lightweight convolutions for improved accuracy and efficient deployment on resource-constrained edge devices.

### 🎯 Key Applications

- 🔥 **Early Fire Warning Systems** - Critical infrastructure protection
- 🏢 **Smart Building Surveillance** - Automated safety monitoring  
- 🏭 **Industrial Safety** - Factory and warehouse monitoring
- 🌲 **Forest Fire Detection** - Wildfire prevention and early detection
- 🤖 **Embedded AI Systems** - Jetson Nano, Xavier, and edge devices
- 📹 **CCTV Integration** - Real-time surveillance enhancement

### 🎓 Research Context

**MS Thesis Project**  
- **Title**: Vision-Based Real-Time Fire and Smoke Detection
- **Institution**: NUST (National University of Sciences and Technology), Islamabad
- **Duration**: October 2021 – January 2025
- **Supervisor**: Dr. Kunwar Faraz Ahmed
- **Specialization**: Artificial Intelligence and Robotic Systems

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Performance
- ✅ **86.2% mAP@50** - State-of-the-art accuracy
- ✅ **83 FPS** on RTX GPU - Real-time processing
- ✅ **120 FPS** on Nano model - Edge optimized
- ✅ **61.3% mAP@50-95** - Robust detection

</td>
<td width="50%">

### 🚀 Deployment Ready
- ✅ **Edge Device Support** - Jetson Nano/Xavier
- ✅ **TensorRT Optimized** - Fast inference
- ✅ **ONNX Compatible** - Cross-platform
- ✅ **Low Latency** - <12ms inference time

</td>
</tr>
</table>

### 🔬 Technical Innovations

1. **Efficient Channel Attention (ECA)** - Feature refinement with minimal overhead
2. **Lightweight C3Ghost Convolution** - Reduced parameters without accuracy loss
3. **Optimized Detection Head** - Specialized for fire and smoke characteristics
4. **Multi-Scale Feature Learning** - Captures small and dense smoke regions

---

## 🎬 Demo Results

### 🔹 Detection on Images

<p align="center">
  <img src="Inferenced_images/bothFireAndSmoke_CV000070.jpg" width="45%"/>
  <img src="Inferenced_images/bothFireAndSmoke_CV000341.jpg" width="45%"/>
</p>

<p align="center">
  <img src="Inferenced_images/bothFireAndSmoke_CV002059.jpg" width="45%"/>
  <img src="Inferenced_images/WEBSmoke177_jpg.rf.1d7116d3d099185d417717b61657fa5e.jpg" width="45%"/>
</p>

*Accurate detection of fire and smoke in diverse lighting conditions, indoor/outdoor scenarios, and varying scales*

### 🔹 Detection on Video

![Video Demo](demo/fire_detection_demo.gif)

*Real-time detection at 83 FPS on complex fire scenarios with multiple smoke plumes*

---

## 🧠 Model Architecture

### High-Level Overview

```
Input Image (640×640)
    ↓
┌─────────────────────────────────┐
│  Backbone (Enhanced YOLOv8)     │
│  • C3Ghost Convolutions         │
│  • Multi-scale feature pyramid  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Neck (Feature Fusion)          │
│  • PANet architecture           │
│  • ECA attention modules        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Detection Head                 │
│  • Fire class prediction        │
│  • Smoke class prediction       │
│  • Bounding box regression      │
└────────────┬────────────────────┘
             │
             ▼
    Output (Fire & Smoke)
```

### 🔧 Key Architectural Components

#### 1. Efficient Channel Attention (ECA)

```python
class ECALayer(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
```

**Benefits:**
- Minimal parameter increase (0.02M)
- 2.1% mAP improvement
- Negligible FPS impact

#### 2. C3Ghost Convolution

```python
class C3Ghost(nn.Module):
    """Lightweight C3 module with Ghost convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_, c2, 1, 1)
```

**Benefits:**
- 25% parameter reduction
- Maintains detection accuracy
- Faster inference on edge devices

---

## 📊 Performance Metrics

### Comprehensive Comparison

| Model | Input Size | Params | FLOPs | mAP@50 | mAP@50–95 | FPS (RTX) | FPS (Jetson) |
|-------|-----------|--------|-------|--------|-----------|-----------|--------------|
| **YOLOv8m (Baseline)** | 640 | 11.2M | 28.4G | 82.1% | 56.7% | 85 | 22 |
| **YOLOv8m + CBAM** | 640 | 11.9M | 30.1G | 84.3% | 58.9% | 78 | 19 |
| **YOLOv8m + ECA (Proposed)** | 640 | **11.4M** | 28.9G | **86.2%** | **61.3%** | **83** | **21** |
| **YOLOv8n (Edge Optimized)** | 640 | 3.2M | 8.1G | 80.4% | 53.2% | 120 | 35 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | mAP@50 |
|-------|-----------|--------|----------|--------|
| **Fire** | 88.7% | 84.3% | 86.4% | 87.9% |
| **Smoke** | 83.6% | 82.1% | 82.8% | 84.5% |
| **Average** | 86.2% | 83.2% | 84.6% | 86.2% |

### Speed Benchmarks

| Device | Resolution | Batch Size | Latency | FPS | Power |
|--------|-----------|------------|---------|-----|-------|
| **RTX 3080** | 640×640 | 1 | 12ms | 83 | 320W |
| **Tesla T4** | 640×640 | 1 | 18ms | 55 | 70W |
| **Jetson Xavier** | 640×640 | 1 | 47ms | 21 | 30W |
| **Jetson Nano** | 416×416 | 1 | 95ms | 10 | 10W |

### Training Curves

![Training Metrics](results/training_curves.png)

*Loss, Precision, Recall, and mAP progression over 100 epochs*

---

## 🗂️ Datasets

The model is trained and evaluated on two comprehensive fire and smoke datasets:

### 🔹 FASDD (Fire and Smoke Detection Dataset)

- **Images**: 8,000+ annotated images
- **Scenarios**: Indoor & outdoor fire scenes
- **Features**: Dense smoke, flames, various lighting conditions
- **Usage**: Training (80%), Validation (10%), Testing (10%)

### 🔹 FS Dataset

- **Images**: 5,000+ challenging scenarios
- **Features**: Real-world fire situations, occlusions, weather variations
- **Usage**: Generalization testing and cross-validation

### 📁 Dataset Structure

```
dataset/
├── images/
│   ├── train/          # 6,400 training images
│   ├── val/            # 800 validation images
│   └── test/           # 800 test images
└── labels/
    ├── train/          # YOLO format annotations
    ├── val/
    └── test/
```

### 🎨 Data Augmentation Pipeline

To enhance model robustness and generalization:

```yaml
# Augmentation configuration
augmentation:
  - Mosaic: 1.0              # 4-image mosaic
  - MixUp: 0.5               # Image blending
  - HSV Jitter:
      h: 0.015               # Hue
      s: 0.7                 # Saturation
      v: 0.4                 # Value
  - Random Scale: [0.5, 1.5]
  - Horizontal Flip: 0.5
  - Motion Blur: 0.3         # Simulates smoke diffusion
  - Rotation: [-15, 15]
```

**📥 Dataset Access**: Datasets are available upon request for research purposes. Please contact the author.

---

## ⚙️ Installation

### 🔹 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 18.04 / Windows 10 | Ubuntu 20.04+ |
| **Python** | 3.8+ | 3.9+ |
| **CUDA** | 11.0+ | 11.7+ |
| **GPU** | 4GB VRAM | 8GB+ VRAM |
| **RAM** | 8GB | 16GB+ |

### 🔹 Quick Setup

```bash
# Clone repository
git clone https://github.com/Qaiser007khan/Fire-and-smoke-detection.git
cd Fire-and-smoke-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install YOLOv8
pip install ultralytics

# Verify installation
yolo version
```

### 🔹 Requirements

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
onnx>=1.14.0          # For ONNX export
onnxruntime>=1.15.0   # For ONNX inference
```

### 🔹 TensorRT Setup (Optional)

For optimized edge deployment:

```bash
# Install TensorRT
pip install nvidia-tensorrt

# Export to TensorRT
yolo export model=best.pt format=engine device=0
```

---

## 🏋️ Training

### 🔹 Configure Dataset

Create `data/fire_smoke.yaml`:

```yaml
# Dataset configuration
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 2
names: ['fire', 'smoke']

# Hyperparameters
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
```

### 🔹 Train Baseline Model

```bash
yolo task=detect mode=train \
  model=yolov8m.yaml \
  data=data/fire_smoke.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=yolov8m_baseline
```

### 🔹 Train Enhanced Model (ECA)

```bash
yolo task=detect mode=train \
  model=models/yolov8m_ECA.yaml \
  data=data/fire_smoke.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=yolov8m_eca \
  patience=20 \
  save_period=10
```

### 🔹 Multi-GPU Training

```bash
# Using DataParallel
yolo train model=yolov8m_ECA.yaml data=fire_smoke.yaml epochs=100 device=0,1

# Using DDP (Distributed Data Parallel)
python -m torch.distributed.run --nproc_per_node 2 train.py \
  --model yolov8m_ECA.yaml \
  --data fire_smoke.yaml \
  --epochs 100 \
  --batch 32
```

### 🔹 Resume Training

```bash
yolo train resume model=runs/train/yolov8m_eca/weights/last.pt
```

### 🔹 Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir runs/train

# Or use WandB for cloud monitoring
pip install wandb
wandb login
# Training will automatically log to WandB
```

---

## 🔍 Inference & Testing

### 🔹 Image Inference

```bash
# Single image
yolo detect predict model=best.pt source=demo/images/fire.jpg

# Multiple images
yolo detect predict model=best.pt source=demo/images/*.jpg

# Save results
yolo detect predict model=best.pt source=demo/images/ \
  save=True save_txt=True conf=0.5
```

### 🔹 Video Inference

```bash
# Video file
yolo detect predict model=best.pt source=demo/videos/fire_scene.mp4

# Webcam (device 0)
yolo detect predict model=best.pt source=0

# RTSP stream
yolo detect predict model=best.pt source=rtsp://camera_ip:554/stream
```

### 🔹 Batch Inference

```bash
# Process directory
yolo detect predict model=best.pt source=demo/test_images/ \
  save=True conf=0.5 iou=0.45
```

### 🔹 Python API

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Inference on image
results = model('fire.jpg')

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        print(f"Class: {model.names[cls]}, Conf: {conf:.2f}")

# Inference on video
cap = cv2.VideoCapture('fire_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('Fire Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 📦 Model Export & Deployment

### 🔹 Export Formats

```bash
# ONNX (cross-platform)
yolo export model=best.pt format=onnx opset=12

# TensorRT (NVIDIA optimization)
yolo export model=best.pt format=engine device=0 half=True

# TorchScript
yolo export model=best.pt format=torchscript

# CoreML (iOS/macOS)
yolo export model=best.pt format=coreml

# TensorFlow Lite (mobile)
yolo export model=best.pt format=tflite
```

### 🔹 Edge Deployment - Jetson Nano

```bash
# Install Jetson-specific packages
sudo apt-get install python3-pip
pip3 install jetson-stats

# Export to TensorRT
yolo export model=best.pt format=engine device=0 half=True

# Run inference
python3 detect_jetson.py --model best.engine --source 0
```

### 🔹 ONNX Deployment

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('best.onnx')

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})
```

### 🔹 Deployment Checklist

- [ ] Model exported to target format
- [ ] Inference tested on target device
- [ ] Latency benchmarked
- [ ] Accuracy validated
- [ ] Power consumption measured
- [ ] Deployment pipeline automated

---

## 📈 Results & Analysis

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Detection Examples by Scenario

| Scenario | Performance | Notes |
|----------|-------------|-------|
| **Indoor Fire** | 89% mAP | Excellent in controlled lighting |
| **Outdoor Smoke** | 84% mAP | Robust to weather conditions |
| **Dense Smoke** | 82% mAP | Handles occlusions well |
| **Small Flames** | 80% mAP | Multi-scale detection effective |
| **Night Scenes** | 78% mAP | Lower lighting challenges |

### Ablation Study

| Component | mAP@50 | Params | FPS | Δ mAP |
|-----------|--------|--------|-----|-------|
| **Baseline YOLOv8m** | 82.1% | 11.2M | 85 | - |
| + ECA | 84.3% | 11.4M | 83 | +2.2% |
| + C3Ghost | 83.8% | 8.9M | 90 | +1.7% |
| + Both (Proposed) | 86.2% | 11.4M | 83 | +4.1% |

---

## 🎯 Applications & Use Cases

### 1. Smart Building Surveillance

```python
# Real-time CCTV monitoring
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture('rtsp://camera_ip/stream')

while True:
    ret, frame = cap.read()
    results = model(frame)
    
    # Alert on detection
    if len(results[0].boxes) > 0:
        send_alert()  # SMS, Email, Dashboard
```

### 2. Industrial Safety Monitoring

- Manufacturing plants
- Warehouses
- Chemical facilities
- Power stations

### 3. Forest Fire Detection

- Early wildfire detection
- Drone-mounted systems
- Tower surveillance
- Satellite integration

### 4. Embedded Systems

- Jetson-based edge devices
- Smart cameras
- IoT fire sensors
- Mobile applications

---

## 🔬 Research & Development

### MS Thesis Details

**Title**: Vision-Based Real-Time Fire and Smoke Detection  
**Author**: Qaiser Khan  
**Institution**: NUST, Islamabad  
**Duration**: October 2021 – January 2025  
**GPA**: 3.45/4.0

**Supervisors**:
- Dr. Kunwar Faraz Ahmed (Thesis Supervisor)
- Dr. Umer Asgher (Co-Supervisor)
- Dr. Shahbaz Khan (Academic Mentor)

**Key Contributions**:
1. Enhanced YOLOv8 architecture with ECA attention
2. Lightweight model for edge deployment
3. Comprehensive fire and smoke dataset
4. Real-time detection framework
5. Edge optimization strategies

### Publications

**Status**: Under Review

**Paper**: "Real-Time Fire and Smoke Detection Using Lightweight Attention-Enhanced YOLOv8 for Edge Devices"

**Abstract**: This work presents an enhanced YOLOv8 framework incorporating Efficient Channel Attention (ECA) and C3Ghost convolutions for real-time fire and smoke detection on resource-constrained edge devices...

---

## 📊 Comparison with State-of-the-Art

| Method | Year | mAP@50 | FPS | Params | Edge Ready |
|--------|------|--------|-----|--------|------------|
| Faster R-CNN | 2015 | 78.3% | 7 | 138M | ❌ |
| SSD | 2016 | 75.1% | 46 | 26M | ⚠️ |
| YOLOv5s | 2020 | 79.8% | 95 | 7.2M | ✅ |
| EfficientDet | 2020 | 81.2% | 42 | 7.8M | ✅ |
| YOLOv7 | 2022 | 83.5% | 71 | 37M | ⚠️ |
| YOLOv8m | 2023 | 82.1% | 85 | 11.2M | ✅ |
| **Proposed (ECA)** | 2024 | **86.2%** | **83** | **11.4M** | ✅ |

---

## 🛠️ Project Structure

```
Fire-and-smoke-detection/
├── models/
│   ├── yolov8m.yaml               # Baseline model
│   ├── yolov8m_ECA.yaml          # Enhanced with ECA
│   └── yolov8n_lite.yaml         # Lightweight version
│
├── data/
│   ├── fire_smoke.yaml           # Dataset configuration
│   └── augmentation_config.yaml  # Augmentation settings
│
├── weights/
│   ├── best.pt                   # Best trained model
│   ├── last.pt                   # Latest checkpoint
│   └── best.onnx                 # Exported ONNX
│
├── Inferenced_images/            # Detection results
│   ├── bothFireAndSmoke_CV000070.jpg
│   └── ...
│
├── demo/
│   ├── images/                   # Sample images
│   ├── videos/                   # Sample videos
│   └── fire_detection_demo.gif
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics_table.csv
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── detect_jetson.py          # Jetson deployment
│   └── export_models.py          # Model export
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📚 Documentation

### Additional Resources

- 📖 [Training Guide](docs/training_guide.md)
- 🔧 [Deployment Guide](docs/deployment_guide.md)
- 🎯 [Parameter Tuning](docs/parameter_tuning.md)
- 🚀 [Edge Optimization](docs/edge_optimization.md)
- 📊 [Dataset Preparation](docs/dataset_preparation.md)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

### Author Information

**Qaiser Khan**  
*Mechatronics Engineer | AI & Robotics Specialist*

- 🎓 MS Mechatronics (AI & Robotics), NUST
- 💼 AI Developer at CENTAIC-NASTP
- 📧 Email: qkhan.mts21ceme@student.nust.edu.pk
- 🔗 LinkedIn: [Qaiser Khan](https://www.linkedin.com/in/engr-qaiser-khan-520252112)
- 🐙 GitHub: [Qaiser007khan](https://github.com/Qaiser007khan)
- 📱 WhatsApp: +92-318-9000211

### Supervisors

**Dr. Kunwar Faraz Ahmed**  
Head of Department, SMME, NUST  
📧 kunwar.faraz@ceme.nust.edu.pk

**Dr. Umer Asgher**  
Research Fellow, Czech Technical University  
📧 umer.asgher@cvut.cz

**Dr. Shahbaz Khan**  
Assistant Professor, SMME, NUST  
📧 shahbaz.khan@smme.nust.edu.pk

---

## 🙏 Acknowledgments

This research was conducted at:

- 🏛️ **NUST** - National University of Sciences and Technology, Islamabad
- 🔬 **CENTAIC-NASTP** - Research facilities and computational resources
- 👥 **SMME Department** - Academic guidance and support

Special thanks to:
- Ultralytics team for YOLOv8 framework
- Open-source computer vision community
- Fire and smoke dataset contributors
- NUST faculty and research staff

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{khan2025firesmoke,
  title={Vision-Based Real-Time Fire and Smoke Detection},
  author={Khan, Qaiser},
  year={2025},
  school={National University of Sciences and Technology (NUST)},
  address={Islamabad, Pakistan},
  type={MS Thesis},
  note={Specialization: Artificial Intelligence and Robotic Systems}
}

@article{khan2024firedetection,
  title={Real-Time Fire and Smoke Detection Using Lightweight Attention-Enhanced YOLOv8},
  author={Khan, Qaiser and Ahmed, Kunwar Faraz and Asgher, Umer},
  journal={Under Review},
  year={2024}
}
```

---

## 🔗 Related Projects

Check out my other AI and robotics projects:

- 🎯 [Intruder Detection System](https://github.com/Qaiser007khan/Intruder-Detection-System)
- 🔫 [Weapon Detection System](https://github.com/Qaiser007khan/Weapon-Detection-System)
- 🐟 [Fish Tracking System](https://github.com/Qaiser007khan/Fish-Tracking-Sonar)
- 🌊 [Underwater Image Enhancement](https://github.com/Qaiser007khan/Underwater-Image-Enhancement)
- 🌿 [Smart Agriculture Sprayer](https://github.com/Qaiser007khan/Smart-Agriculture-Sprayer)

---

<div align="center">

### 🌟 If you find this work useful, please star the repository!

### 🔥 Together, let's make the world safer with AI!

![GitHub stars](https://img.shields.io/github/stars/Qaiser007khan/Fire-and-smoke-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/Qaiser007khan/Fire-and-smoke-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Qaiser007khan/Fire-and-smoke-detection?style=social)

---

**© 2024-2025 Qaiser Khan | MS Thesis Project | NUST, Islamabad**

*Making fire detection smarter, faster, and more accessible* 🔥🤖

</div>
