# TapNFloAIcam

> AI-powered offline exam surveillance — a module of the **TapNFlo** startup platform.

TapNFloAIcam monitors students during physical offline examinations using existing CCTV infrastructure. It processes live camera feeds entirely on-premise, detects suspicious behaviour in real time, and ties every alert to a specific seat number and roll number — with no internet dependency and no cloud.

---

## What It Does

- Detects all students in the camera frame using YOLOv8 pose estimation
- Automatically maps detected persons to seat positions during a calibration phase
- Classifies behaviour per student every frame: **Normal Sitting**, **Leaning to Copy**, **Looking Around**
- Tracks each student across frames so roll numbers stay tied to the right person
- Prints a full session report at the end with percentage breakdowns and flagged students

---

## How to Run

### Requirements
- Python 3.10 or 3.11
- Windows / macOS / Linux
- A webcam or a video file

### Install

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/TapNFloAIcam.git
cd TapNFloAIcam

# Create virtual environment with Python 3.10
py -3.10 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install opencv-python==4.8.1.78 mediapipe==0.10.9 numpy==1.26.4 ultralytics scipy
```

### Run on a video file

```bash
python analyze_video.py --video path\to\exam.mp4
```

### Run with output saved

```bash
python analyze_video.py --video exam.mp4 --output result.mp4
```

### Run at faster playback speed

```bash
python analyze_video.py --video exam.mp4 --output result.mp4 --speed 1.5
```

### Controls during playback

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `S` | Skip 10 seconds forward |
| `Q` | Quit — still saves output and prints report |

> **Note:** The first run downloads `yolov8s-pose.pt` (~23MB) automatically.

---

## Seat Allocation

Before running, fill in `seat_allocation.csv` with your exam hall's seating plan:

```csv
seat,roll_number,name
A1,21CS001,Aarav Sharma
A2,21CS002,Priya Mehta
B1,21CS003,Rohit Nair
```

During the first 1.5 seconds of video the system detects all students, sorts them left-to-right and assigns roll numbers from the CSV in order. Every alert and report entry is tied to a seat and roll number from that point forward.

---

## Project Structure

```
TapNFloAIcam/
├── analyze_video.py         # main script — run this
├── seat_allocation.csv      # seat → roll number → name
├── config.py                # detection thresholds
├── requirements.txt
├── README.md
├── core/
│   ├── detectors.py         # head pose + shoulder detection (webcam mode)
│   ├── tracker.py           # cumulative time scoring
│   └── seat_manager.py      # seat grid and student records
└── utils/
    └── hud.py               # HUD rendering
```

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Person detection | YOLOv8s-pose (Ultralytics) | Detect all students per frame |
| Pose estimation | YOLOv8 keypoints (COCO 17-point) | Body keypoints per student |
| Seat assignment | Hungarian algorithm (SciPy) | Match detections to seats |
| Tiled inference | OpenCV + NumPy | Detect distant/edge students |
| Contrast enhancement | CLAHE (OpenCV) | Improve dark corner detection |
| Rendering | OpenCV | Live HUD overlay |

---

## Current Prototype — Known Drawbacks

This is a working proof-of-concept. The following limitations exist in the current version and are known:

### 1. Generic pre-trained model
The system uses `yolov8s-pose.pt`, which was trained on general internet images of front-facing people in normal lighting. It was not trained on:
- Top-down or diagonal CCTV footage
- Indian exam hall environments
- Students in similar uniforms at similar desks
- Compressed low-bitrate CCTV video quality

This causes missed detections at the edges of the frame and occasional false behaviour classifications.

### 2. Camera angle limitations
The behaviour classifier was designed for corner-mounted diagonal CCTV. It uses torso lean and wrist reach as signals. For other mounting angles (directly overhead, eye level) the thresholds may need manual adjustment in `config.py`.

### 3. No temporal memory across sessions
There is no database or persistent storage. Reports exist only as terminal output for the current session. Evidence clips are not automatically saved.

### 4. Seat calibration is position-based
Seat assignment during calibration sorts detected people by position (left to right). If a student is missing from the first 1.5 seconds of video, their seat may be assigned to the wrong person.

### 5. No audio detection
Whispering and verbal communication between students is not detected. A microphone input channel is not yet integrated.

---

## What Training Would Fix

The right way to solve the detection and classification problems is to fine-tune the model on real exam hall footage. This is not training from scratch — it is transfer learning on top of YOLOv8, which takes a few hours on a free GPU.

### What you need
- 200–500 labelled video frames from your actual exam hall CCTV
- Labels for each student: `Normal Sitting`, `Leaning to Copy`, `Looking Around`, `Looking Under Desk`
- Free labelling tool: [Roboflow](https://roboflow.com)
- Free training environment: Google Colab (free GPU)

### Why this matters for the product
Every exam hall that TapNFlo is deployed in contributes labelled footage. The model improves with each deployment. This creates a data advantage that competitors cannot easily replicate — the longer TapNFlo operates, the more accurate it gets for Indian exam hall conditions specifically.

---

## Roadmap

- [ ] Fine-tuned model on real exam hall CCTV footage
- [ ] Multi-camera parallel tracking (one tracker per camera feed)
- [ ] RTSP stream input (connect directly to existing IP cameras)
- [ ] ByteTrack integration — keeps tracking students even when detector misses them briefly
- [ ] Looking-under-desk detection (phone/chit below table level)
- [ ] Audio channel — whisper and speech detection
- [ ] Automatic evidence clip export per flagged student
- [ ] Web dashboard for invigilator (React)
- [ ] Post-exam PDF report generation

---

## Part of TapNFlo

TapNFloAIcam is the AI surveillance module of **TapNFlo** — a platform for managing, securing, and analysing offline academic examinations in Indian universities and competitive exam centres.

---

*Prototype v0.1 — proof of concept. Production accuracy requires fine-tuning on real exam hall footage.*
