# TapNFloAIcam

> AI-powered offline exam surveillance — a module of the **TapNFlo** startup platform.

TapNFloAIcam is designed to monitor students during physical offline examinations using existing CCTV infrastructure. The system processes live camera feeds on-premise, detects suspicious behaviour in real time, and alerts invigilators — with every incident tied to a specific seat and roll number.

**No internet required. No cloud. No replacement of existing cameras.**

---

## The Problem

Traditional exam invigilation is manual, inconsistent, and unscalable. One invigilator cannot watch 40 students simultaneously. Malpractice often goes undetected simply because no one was looking at the right student at the right moment.

Existing online proctoring tools (like Mettl, ProctorU) only work for computer-based exams. They are useless for pen-and-paper offline exams — which are still the standard across most Indian universities and competitive examinations.

---

## The Solution

TapNFloAIcam plugs into the exam hall's existing CCTV network via an on-premise AI box. It watches every student simultaneously, accumulates suspicious behaviour over time (not single incidents), and surfaces only genuine concerns to the invigilator — with a clip, a timestamp, and a roll number attached.

---

## How Detection Works

### Core principle — cumulative time, not single incidents

A student glancing sideways once is thinking. A student whose gaze drops below the desk line for a cumulative 2 minutes across an exam is cheating.

The system does **not** flag on single events. It accumulates time spent in suspicious postures and only raises an alert once configurable thresholds are crossed.

| Behaviour | Method | SUSPICIOUS | FLAGGED |
|-----------|--------|------------|---------|
| Gaze below desk level | Eye gaze + desk plane estimation | 30s cumulative | 2 min cumulative |
| Sustained sideways orientation | Body pose + head yaw deviation | 30s cumulative | 2 min cumulative |
| Shoulder tap / nudge | Sudden shoulder keypoint spike | 2 incidents | 4 incidents |
| Inter-student interaction | Two-body proximity + orientation | 30s cumulative | 2 min cumulative |

### Calibration-first approach

When monitoring begins, the system runs a **3-second calibration** per student to establish their personal neutral posture. All detections are deviations from that baseline — not absolute angles. This means:

- Works at any camera mounting angle (eye level, ceiling, CCTV)
- Natural writing posture (head slightly down) does not trigger alerts
- Each student's baseline is stored independently

---

## Current Prototype vs Production

This repository is a **proof-of-concept prototype** built to demonstrate the system architecture, detection logic, and invigilator dashboard. It runs on a single laptop webcam.

| Aspect | Prototype (this repo) | Production |
|--------|----------------------|------------|
| Camera input | Single laptop webcam | Multiple RTSP/ONVIF CCTV feeds |
| Camera angle | Eye level | Ceiling mounted, top-down |
| Student tracking | Manual seat switching (arrow keys) | Automatic — all students in parallel |
| Seat mapping | CSV preloaded | Calibration grid overlaid on camera |
| Gaze detection | Head pose approximation | True gaze estimation (requires CCTV-angle training data) |
| Detection model | Pre-trained + rule engine | Fine-tuned on real exam footage |
| Deployment | Laptop | On-premise AI box on hall LAN |
| Multi-camera | No | Yes — one tracker per student region |

### Why the webcam prototype looks different from the real use case

A ceiling-mounted CCTV sees the top of a student's head. A laptop webcam sees their face straight-on. These are fundamentally different viewing angles, and the same detection model cannot work for both without retraining.

The prototype uses head pose estimation (pitch/yaw angles) as a stand-in for what the production system would do with proper gaze estimation from a CCTV angle. The **architecture, scoring logic, seat mapping, alert system, and dashboard are all production-ready**. The detection layer is the part that requires real exam footage to train and calibrate properly.

---

## What Is Needed for Production

### 1. Training Data
- Recorded CCTV footage from real exam halls (with consent)
- Labelled examples of: normal writing, looking under desk, sideways glancing, shoulder tapping, passing notes
- Minimum ~50 hours of footage across varied lighting and hall configurations
- Annotation tool: [CVAT](https://cvat.ai) or [Label Studio](https://labelstud.io)

### 2. Hardware
- **AI processing box**: NVIDIA Jetson Orin NX or a small form-factor PC with GPU (e.g. Intel NUC with RTX)
- **Cameras**: Existing IP cameras with RTSP support, or dahua/hikvision CCTV — most institutions already have these
- **Network**: Local LAN only — no internet required
- **Invigilator screen**: Any monitor connected to the AI box

### 3. Camera Setup Requirements
- Minimum **1 camera per 20 students** for reliable coverage
- Camera mounted at **2.5–3m height**, angled to cover 4–5 rows
- Minimum **1080p resolution** at 25fps
- Adequate hall lighting (no strong backlighting)

### 4. Model Fine-tuning Stack
- **Pose estimation**: MediaPipe Pose or YOLOv8-Pose (pre-trained, needs angle-specific fine-tuning)
- **Gaze estimation**: L2CS-Net or ETH-XGaze (requires top-down angle retraining)
- **Gesture / interaction**: Custom classifier trained on labelled exam footage
- **Framework**: PyTorch + ONNX for deployment on edge hardware

### 5. Calibration Step
- Before each exam: operator walks the hall with a calibration board
- System maps each camera's field of view to the seat grid
- Desk plane is established per camera for accurate "below desk" gaze detection

---

## Project Structure

```
TapNFloAIcam/
├── main.py                  # entry point — camera loop
├── config.py                # all thresholds in one place
├── seat_allocation.csv      # seat → roll number → name mapping
├── requirements.txt
├── README.md
├── core/
│   ├── detectors.py         # HeadPoseDetector + ShoulderDetector
│   ├── tracker.py           # BehaviourTracker — cumulative time scoring
│   └── seat_manager.py      # seat grid, student records, hall summary
└── utils/
    └── hud.py               # HUD rendering — fully separate from logic
```

---

## Setup (Windows)

### Requirements
- Python 3.10 or 3.11 (not 3.12+)
- Webcam
- Windows / macOS / Linux

### Install

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/TapNFloAIcam.git
cd TapNFloAIcam

# 2. Create virtual environment with Python 3.10
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies (specific versions required)
pip install opencv-python==4.8.1.78 mediapipe==0.10.9 numpy==1.26.4

# 4. Run
python main.py
```

> **Note:** MediaPipe 0.10.9 and Python 3.10/3.11 are required. Newer versions of MediaPipe (0.10.14+) break the `solutions` API on Windows.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit — prints session summary to terminal |
| `R` | Reset current student's score |
| `RIGHT` / `D` | Switch to next seat (triggers recalibration) |
| `LEFT` / `A` | Switch to previous seat (triggers recalibration) |

---

## Seat Allocation

Before the exam, fill in `seat_allocation.csv`:

```csv
seat,roll_number,name
A1,21CS001,Aarav Sharma
A2,21CS002,Priya Mehta
B1,21CS003,Rohit Nair
```

Every alert and flag is tied to a seat → roll number → name. Session summary at the end lists all flagged students with cumulative suspicious time.

---

## Configuration

All thresholds are in `config.py` — no other file needs to be touched to adjust sensitivity.

```python
PITCH_DOWN_THRESHOLD    = 12    # degrees below neutral → looking down
YAW_SIDE_THRESHOLD      = 18    # degrees sideways → glancing

LOOK_DOWN_DURATION      = 2.0   # seconds sustained before counting
LOOK_SIDE_DURATION      = 1.5

CUMULATIVE_WARN_SEC     = 30    # total seconds → SUSPICIOUS
CUMULATIVE_FLAG_SEC     = 120   # total seconds → FLAGGED (2 minutes)

SHOULDER_TAP_WARN_COUNT = 2
SHOULDER_TAP_FLAG_COUNT = 4
SHOULDER_MOVE_THRESHOLD = 0.06  # 6% of frame height
```

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| Face mesh | MediaPipe FaceMesh | 468 landmarks, pre-trained |
| Head pose | OpenCV solvePnP | Pitch + yaw estimation |
| Body pose | MediaPipe Pose | 33 keypoints, pre-trained |
| Shoulder tracking | NumPy rolling delta | Tap detection |
| Rendering | OpenCV | Live HUD overlay |

No custom model training required to run this prototype.

---

## Roadmap

- [ ] RTSP stream input (connect to existing CCTV)
- [ ] True gaze estimation from top-down angle (requires training data)
- [ ] Desk plane calibration for "below desk" detection
- [ ] Multi-student parallel tracking (one tracker per seat region)
- [ ] Web-based invigilator dashboard (React)
- [ ] Timestamped clip export for evidence
- [ ] Audio — whisper/speech detection via microphone
- [ ] Post-exam PDF report generation per student

---

## Part of TapNFlo

TapNFloAIcam is the AI surveillance module of **TapNFlo** — a platform for managing, securing, and analysing offline academic examinations.

---

*Prototype v0.1 — architecture demonstration. Detection layer requires CCTV-angle training data for production use.*
