# CrossSafe — Pedestrian Intention Prediction

> Real-time deep learning framework for pedestrian crossing intention prediction in Indian traffic conditions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

CrossSafe is a real-time computer vision pipeline that analyzes dashcam video from a moving vehicle and predicts — **before any physical movement begins** — whether each visible pedestrian intends to cross the road.

The system outputs one of three labels per pedestrian per frame:

| Label | Color | Meaning |
|---|---|---|
| **CROSSING** | 🔴 Red | High probability of crossing |
| **MAYBE** | 🟠 Orange | Ambiguous — monitoring |
| **NOT CROSSING** | 🟢 Green | Stationary / not intending to cross |

This addresses a critical gap in current ADAS systems: existing solutions are **reactive** (they respond after a pedestrian steps onto the road). CrossSafe is **proactive** — it predicts intent from body language before movement begins.

---

## Demo Output

```
CrossSafe - Perception Layer
FPS: 8.6  Frame: 273/21210  Peds: 2  gp_set_0001_vid_0008
[G]GT  [K]Pose  [I]Intent  [X]Seg  [P]Pause  [S]Step  [Q]Quit
```

Each detected pedestrian shows:
- Colored bounding box (red/orange/green)
- 33-point pose skeleton (color matches intent)
- Label tag: `#ID LABEL SCORE` e.g. `#1 CRS 0.65`
- Body orientation: SIDE / FRONT / BACK
- Vertical score bar on right edge of box
- Ground truth annotation overlay (from IDD-PeD dataset)

---

## Architecture

The perception pipeline has 4 modules:

```
Dashcam Video Frame
        │
        ▼
┌───────────────────┐
│   Module 1        │  YOLOv8n
│   Detection       │  Detect persons + vehicles
└────────┬──────────┘
         │  3-Layer Rider Suppression
         │  (geometry + proximity + pose)
         ▼
┌───────────────────┐
│   Module 2        │  MediaPipe BlazePose
│   Pose Estimation │  33 body landmarks per person
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Module 3        │  Rule-Based Weighted Classifier
│   Intent Scoring  │  Stride + Head turn + Body lean
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Module 4        │  SegFormer-b0 (Cityscapes)
│   Segmentation    │  Road / Footpath / Crosswalk zones
└────────┬──────────┘
         │
         ▼
   CROSSING / MAYBE / NOT CROSSING + Score
```

---

## Module Details

### Module 1 — Pedestrian Detection (YOLOv8n)

- Detects persons (class 0) at `conf=0.45`
- Detects vehicles (classes 1,2,3,5,7) at `conf=0.28` — used for suppression only, never displayed
- **3-Layer Rider Suppression:**
  - **Layer 1 (Geometry):** Rejects boxes with `height/width < 1.3` or `width > 200px` — scooter riders create wide boxes
  - **Layer 2 (Proximity):** Suppresses person if centroid falls inside an expanded vehicle bounding box — catches riders even when pose is noisy
  - **Layer 3 (Pose):** Suppresses if `torso_height < 0.20` in normalised coordinates — seated riding posture

### Module 2 — Pose Estimation (MediaPipe BlazePose)

- Extracts 33 body landmarks per pedestrian
- Runs on individual person crops (not full frame) for accuracy
- 3-attempt padding fallback: 20px → 40px → 60px padding for edge/occluded persons
- `model_complexity=0` for real-time CPU performance
- `min_detection_confidence=0.30` (lowered for partial crops)

### Module 3 — Crossing Intent Classifier (Rule-Based)

Computes a crossing probability score (0.0 – 1.0) from pose landmarks:

| Feature | Signal | Max Score |
|---|---|---|
| **Stride spread** | `abs(left_ankle.x - right_ankle.x)` — feet together = 0, walking stride = 0.28 | 0.28 |
| **Head rotation** | Ear-to-ear pixel distance in crop — small distance = head turned checking traffic | 0.18 |
| **Forward lean** | Shoulder midpoint vs hip midpoint offset — body leaning toward road | 0.12 |
| **Zone context** | Segmentation zone at feet — in road = +0.20, near crosswalk = +0.10 | 0.20 |
| **Size proxy** | Box height > 30% frame height — person is very close | 0.05 |
| **Maximum total** | | **0.83** |

**Thresholds:**
- Score ≥ 0.55 → **CROSSING** (red)
- Score 0.28–0.55 → **MAYBE** (orange)
- Score < 0.28 → **NOT CROSSING** (green)

> **Note:** Body orientation (SIDE/FRONT/BACK) is displayed but does NOT contribute to the score. In dashcam footage almost all pedestrians appear side-on regardless of intent, so orientation alone is not a reliable signal.

### Module 4 — Scene Segmentation (SegFormer-b0)

- Model: `nvidia/segformer-b0-finetuned-cityscapes-512-1024` (HuggingFace)
- Cityscapes classes used: `road(0)`, `sidewalk(1)`, `terrain(9)`
- Maps to 3 zones: `ZONE_ROAD`, `ZONE_FOOTPATH`, `ZONE_CROSSWALK`
- Post-processing: morphological opening (9×9 kernel) removes stray bleed pixels
- Bottom 10% of frame masked (GoPro bonnet/hood reflection)
- Runs every 8 frames, cached between runs for performance
- **Geometric fallback** if transformers not installed: perspective trapezoid road region
- Requires `transformers>=4.19` and `torch` — see installation notes

---

## Dataset

**IDD-Pedestrian (IDD-PeD)** — Indian Driving Dataset Pedestrian  
Source: IIIT Hyderabad + UCSD  
Website: https://cvit.iiit.ac.in/research/projects/cvit-projects/pedestrian-detection-in-idd

- GoPro dashcam footage from Indian roads (Hyderabad)
- CVAT XML annotations with per-frame bounding boxes
- Crossing intent labels: `1`=crossing, `0.5`=maybe, `0`=not crossing
- Direction codes: `CD`/`CU` (crossing down/up), `CFD`/`CFU` (crossing fast)
- ~45GB across multiple video sets

### Download Dataset

```bash
python download_dataset.py        # downloads gp_set_0001 videos + annotations
python download_annotations_only.py  # annotations only (small)
python extract_annotations.py     # extract tar archive
python peek_annotations.py        # inspect annotation structure
```

---

## Installation

### Requirements

```
Python >= 3.8
opencv-python >= 4.5
ultralytics >= 8.0
mediapipe >= 0.10
numpy >= 1.21
```

### For Scene Segmentation (Module 4)

```bash
pip install transformers>=4.19
pip install torch torchvision
```

> GPU (CUDA) is optional but recommended for segmentation. CPU mode runs at ~8 FPS.

### Install All Dependencies

```bash
pip install opencv-python ultralytics mediapipe numpy
pip install transformers torch torchvision   # optional, for segmentation
```

### Clone and Setup

```bash
git clone https://github.com/yourusername/crosssafe.git
cd crosssafe

# Download YOLOv8n weights (auto-downloads on first run, ~6MB)
# OR manually place yolov8n.pt in project root

# Download dataset
python download_annotations_only.py   # annotations (~small)
python download_dataset.py            # full videos (~few hundred MB per set)
python extract_annotations.py
```

---

## Usage

### Run Full Perception Pipeline

```bash
python perception_pipeline.py
```

Edit the config section at the top of `perception_pipeline.py` to change video:

```python
VIDEO_DIR  = 'data/IDDPedestrian/videos/gp_set_0001'
ANNOT_DIR  = 'data/IDDPedestrian/annotations/gopro/gp_set_0001'
VIDEO_NAME = 'gp_set_0001_vid_0008'   # change this
DISPLAY_W  = 1100
```

### Keyboard Controls (during playback)

| Key | Action |
|---|---|
| `Q` | Quit |
| `P` | Pause / Resume |
| `S` | Step one frame (when paused) |
| `G` | Toggle ground truth overlay |
| `K` | Toggle pose skeleton |
| `I` | Toggle intent classification |
| `X` | Toggle segmentation overlay |

### Run Individual Modules

```bash
python module1_detection.py      # YOLO detection only (webcam)
python module2_pose.py           # Pose estimation only (webcam)
python module4_segmentation.py   # Segmentation demo (webcam or video path)
```

---

## Project Structure

```
crosssafe/
│
├── perception_pipeline.py       # Main pipeline — run this
│
├── module1_detection.py         # Standalone: YOLOv8 detection demo
├── module2_pose.py              # Standalone: MediaPipe pose demo
├── module3_intent.py            # Intent classifier (importable module)
├── module4_segmentation.py      # Scene segmenter (importable module)
│
├── download_dataset.py          # Download IDD-PeD videos
├── download_annotations_only.py # Download annotations only
├── extract_annotations.py       # Extract tar archive
├── peek_annotations.py          # Inspect annotation XML structure
│
├── yolov8n.pt                   # YOLOv8 nano weights (auto-downloaded)
│
└── data/
    └── IDDPedestrian/
        ├── videos/
        │   └── gp_set_0001/     # Video files (.MP4)
        └── annotations/
            └── gopro/
                └── gp_set_0001/ # XML annotation files
```

---

## Performance

| Metric | Value |
|---|---|
| FPS (CPU, no segmentation) | ~10–12 FPS |
| FPS (CPU, with segmentation) | ~7–9 FPS |
| YOLOv8n model size | 6 MB |
| SegFormer-b0 model size | ~14 MB |
| Tested on | Intel CPU laptop, Windows 10/11 |
| Video resolution | 1920×1080 (GoPro), displayed at 1100px width |

---

## Roadmap

This project implements **Stage 1: Perception Layer** of the full CrossSafe architecture.

- [x] Module 1: Real-time pedestrian detection (YOLOv8n)
- [x] Module 1b: 3-layer rider suppression
- [x] Module 2: Pose estimation + skeleton visualization (MediaPipe)
- [x] Module 3: Rule-based crossing intent classifier
- [x] Module 3b: Ground truth comparison overlay (IDD-PeD)
- [x] Module 4: Scene segmentation 
- [ ] Module 4: SegFormer DNN segmentation (pending library setup)
- [ ] Temporal feature extraction (16-frame pose sequences)
- [ ] Training dataset builder from IDD-PeD annotations
- [ ] BiLSTM / Temporal Intention Transformer (TIT) training
- [ ] Cross-modal attention fusion (CMAF module)
- [ ] NVIDIA Jetson Orin Nano deployment
- [ ] In-vehicle warning module (auditory + visual)

---

## Research Context

CrossSafe addresses three gaps identified in current ADAS technology:

**1. Reactive vs Proactive:** Current ADAS detects pedestrians after they enter the road. CrossSafe predicts intent before movement begins, giving the driver or system more reaction time.

**2. Indian Road Conditions:** Existing datasets (JAAD, PIE, TITAN) are from Western/Japanese roads. Indian traffic has unique challenges — mixed road users, no lane discipline, bollard-lined footpaths, tuk-tuks, and very different pedestrian behavior. IDD-PeD is the first large-scale Indian pedestrian intention dataset.

**3. Edge Deployment:** High inference complexity prevents real-time use on embedded hardware. CrossSafe targets the NVIDIA Jetson Orin Nano (8GB, 40 TOPS) with model pruning and TensorRT optimization planned for Stage 2.

---

## Acknowledgements

- IDD-PeD Dataset: IIIT Hyderabad + UCSD
- YOLOv8: Ultralytics
- MediaPipe: Google
- SegFormer: NVIDIA + HuggingFace
- Funded by: Transportation Research Cell (TRC), Kerala
