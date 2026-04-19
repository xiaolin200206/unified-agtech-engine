# Unified AgTech Edge AI Engine

**Real-time agricultural disease and pest detection on resource-constrained edge devices.**

A production-grade Edge AI platform developed during an internship at Urban Farm Tech (Jan–Apr 2026). Designed for commercial greenhouse deployments with zero cloud dependency, built around YOLOv8 object detection running on Raspberry Pi 5.

> **Research Note:** The engineering challenges documented in this system — thermal management, domain shift, small object detection, and Docker containerization on ARM64 — are the subject of a peer-reviewed manuscript currently under review at *Engineering Applications of Artificial Intelligence* (Elsevier, Q1, IF: 8.0).

---

## What This System Does

- Captures live video from a Raspberry Pi Camera Module 3
- Runs YOLOv8 ONNX inference at ~5.6 FPS on CPU (no GPU required)
- Detects 5 basil disease/pest classes in real-time: Fungal, Leaf Damage, Mealybugs, Miner, Mite
- Sends instant Telegram alerts with annotated images when disease is detected
- Manages CPU thermal load via configurable duty-cycle scheduling
- Logs all detections, latency, temperature, and CPU metrics to CSV

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Edge Device | Raspberry Pi 5 (8GB RAM) |
| Camera | Raspberry Pi Camera Module 3 (12MP, autofocus) |
| OS | Raspberry Pi OS (64-bit) |
| Storage | 32GB+ microSD |
| Power | Official 27W USB-C PSU recommended |

---

## Software Requirements

```bash
Python 3.9+
ultralytics>=8.0
opencv-python-headless
picamera2
numpy
requests
psutil
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
unified-agtech-engine/
├── basil.py              # Main inference script
├── basil.onnx            # YOLOv8 ONNX model (not included in repo)
├── requirements.txt
├── basil_logs/           # Auto-generated session logs
│   └── session_YYYYMMDD_HHMMSS/
│       ├── basil_data.csv
│       └── images/
└── README.md
```

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/xiaolin200206/unified-agtech-engine.git
cd unified-agtech-engine
```

**2. Add your model**
```bash
# Place your basil.onnx in the project root
# Or download via:
wget "YOUR_MODEL_URL" -O basil.onnx
```

**3. Configure Telegram (optional)**

Edit `basil.py` and set your bot token and chat ID:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

**4. Run**
```bash
python3 basil.py
```

Press `q` to quit.

---

## Configuration

All parameters are at the top of `basil.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.40 | Minimum detection confidence |
| `INFERENCE_SIZE` | 640 | Input resolution for YOLOv8 |
| `CYCLE_ACTIVE_SEC` | 180 | Active inference duration per cycle |
| `CYCLE_SLEEP_SEC` | 45 | Sleep duration per cycle |
| `MAX_TEMP_LIMIT` | 82.0°C | CPU overheat protection threshold |
| `TELEGRAM_COOLDOWN_SEC` | 60.0 | Minimum interval between alerts |
| `SAVE_IMG_INTERVAL` | 2.0s | Minimum interval between saved images |

---

## Detection Classes

| Class | Description |
|-------|-------------|
| Fungal | Fungal leaf infections |
| Leaf Damage | Physical or environmental leaf damage |
| Mealybugs | Mealybug pest infestation |
| Miner | Leaf miner pest damage |
| Mite | Spider mite damage (small white spots) |

---

## System Architecture

```
Camera (1080p) → Frame Capture → YOLOv8s ONNX Inference
                                        ↓
                              Confidence Thresholding (≥0.40)
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                                       ↓
            CSV Logging                          Telegram Alert
        (latency, FPS, temp)                 (annotated image + metadata)
```

**Duty Cycle Thermal Management:**
```
[Active 180s] → [Sleep 45s] → [Active 180s] → ...
     ↑ inference runs              ↑ CPU cools down
```

This prevents thermal throttling (observed at >82°C) which causes inference latency to become unpredictable. Without scheduling, 66.2% of frames were affected by throttling in field tests.

---

## Architecture Evolution: From Classification to Detection

This system went through a critical architectural pivot during development. Understanding this evolution explains why the current design is built the way it is.

### v1: MobileNetV2 Classification

The initial approach used MobileNetV2 as an image classifier — the entire 640x640 frame was resized to 224x224 and passed through the network to produce a single disease label.

**What went wrong:**
- The classifier had no spatial awareness. It could say "there is a miner" but not *where* on the leaf.
- Resizing a full HD frame to 224x224 caused severe **pixel evaporation** — tiny features like mite white spots and mealybug colonies were compressed into just a few pixels and became unrecognizable.
- False positives spiked when background elements (soil, reflections, clothing) dominated the frame.
- The system was fundamentally unsuitable for a fixed overhead camera watching multiple plants simultaneously.

**Conclusion:** Classification is appropriate for single-leaf, close-up mobile capture (like PlantVillage-style apps). It is not suitable for continuous overhead greenhouse monitoring.

### v2: YOLOv8s Object Detection (Current)

Switching to YOLOv8s object detection solved the core problems:

- **Spatial localization:** Each detection produces a bounding box, so the system knows exactly which leaf region is affected.
- **Multi-instance detection:** Multiple disease instances across multiple plants are detected in a single inference pass.
- **Better small object handling:** YOLOv8's multi-scale feature pyramid (FPN) retains more spatial detail for small targets than a flat classifier.
- **Practical FPS:** YOLOv8s achieves ~5.6 FPS on Raspberry Pi 5 CPU — sufficient for greenhouse monitoring where disease progression is measured in hours, not seconds.

**Trade-off acknowledged:** Mite damage (small white spots) and mealybugs remain challenging at standard operating distances due to their physical size — this is a hardware constraint, not a model failure. Close-range inspection or higher resolution cameras are recommended for these classes.

### Architecture Design: Model-Agnostic Core

The inference pipeline is designed to support both modes, allowing future switching without rewriting the deployment stack:

| Mode | Model | Input | Output | Best For |
|------|-------|-------|--------|----------|
| Classification | MobileNetV2 | 224x224 patch | Class label + confidence | Single leaf, close-up |
| Detection | YOLOv8s | 640x640 frame | Bounding boxes + labels | Multi-plant, overhead camera |

---

## Key Engineering Findings

**Docker reduces CPU load by 49.1%**
Containerized execution on ARM64 using `python:3.9-slim-bullseye` eliminated background OS processes, reducing CPU utilization from ~85% to ~43% and peak temperature from 76.2°C to 60.9°C.

**Real data outperforms proxy-augmented data**
A model trained exclusively on 1,710 real greenhouse images achieved 99.13% validation accuracy, versus 97.82% for a model augmented with web-scraped proxy images — demonstrating the risk of domain shift in practical deployments.

**Small object limitations**
Mite damage (small white spots) and mealybugs remain challenging at standard operating distances. Recommend close-range inspection or higher-resolution capture for these classes. This is a known hardware constraint, not a model failure.

---

## Known Limitations

- **Mite & Mealybug detection**: Performance degrades at distances >30cm due to small object size. Camera Module 3 autofocus helps but does not fully resolve this.
- **OOD rejection**: The system has limited rejection for out-of-distribution inputs (e.g., reflective clothing, dramatic lighting changes). Fixed camera positioning minimizes this.
- **Single crop**: Model is trained on basil only. Not suitable for other crops without retraining.

---

## Training Pipeline (Offline)

The model was trained using the following pipeline:

```
Field Data Collection (Greenhouse, 3 weeks)
        ↓
Manual Annotation (CVAT platform)
        ↓
Augmentation (Albumentations)
        ↓
YOLOv8s Transfer Learning (PyTorch)
        ↓
ONNX Export (ARM64-optimized)
        ↓
Edge Deployment (Raspberry Pi 5)
```

**Dataset:** 1,710 images, 6 classes, collected on-site at Urban Farm Tech commercial greenhouse, Kuala Lumpur, Malaysia.

---

## Roadmap

- [ ] Active Learning pipeline (auto-save ambiguous detections for HITL relabeling)
- [ ] OTA model update via scheduled download script
- [ ] LoRa-based offline alerting for connectivity-limited farms
- [ ] Multi-crop support (Durian, Tomato)
- [ ] Hardware ID binding for commercial deployment (DRM)

---

## Author

**Lin Ding Shan**
AI Development Intern, Urban Farm Tech (Jan–Apr 2026)
GitHub: [xiaolin200206](https://github.com/xiaolin200206)
Email: l1055505011@gmail.com

---

## License

This project was developed during an internship at Urban Farm Tech, Malaysia. The model weights (`basil.onnx`) and training dataset are proprietary and not included in this repository.
