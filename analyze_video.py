# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — analyze_video.py
#
# Option B: Person detection + automatic seat mapping.
#
# How it works:
#   1. First N frames — detect all people with YOLOv8 pose
#   2. Sort detected people by position (left to right, back to front)
#   3. Lock seat assignments: person at position X → Seat A1 etc.
#   4. Every subsequent frame — re-detect, match each detection
#      to the closest known seat center (Hungarian matching)
#   5. Classify behaviour from pose keypoints per person
#   6. Boxes move with the students, roll numbers stay correct
#
# Usage:
#   python analyze_video.py --video path\to\exam.mp4
#   python analyze_video.py --video exam.mp4 --output result.mp4 --speed 1.5
#   python analyze_video.py --video exam.mp4 --rows 2 --cols 3
#
# Controls:
#   SPACE — pause / resume
#   S     — skip 10 seconds
#   Q     — quit (still saves + prints report)
# ──────────────────────────────────────────────────────────────

import cv2
import numpy as np
import argparse
import sys
import csv
import os
from collections import deque, Counter
from datetime import datetime, timedelta
from scipy.optimize import linear_sum_assignment   # Hungarian algorithm

# ── Arguments ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video',     required=True,         help='Input video path')
parser.add_argument('--output',    default='output.mp4',  help='Output video path')
parser.add_argument('--seats-csv', default='seat_allocation.csv',
                                                           help='Seat allocation CSV')
parser.add_argument('--speed',     type=float, default=1.0)
parser.add_argument('--no-save',   action='store_true')
parser.add_argument('--calib-frames', type=int, default=45,
                    help='Frames used to lock seat positions (default: 45 = 1.5s)')
args = parser.parse_args()

# ── Colours ───────────────────────────────────────────────────
COL = {
    'Normal Sitting':  (255, 255,   0),
    'Leaning to Copy': (  0,   0, 255),
    'Looking Around':  (  0, 165, 255),
    'Examiner':        (255,  80,  80),
    'Calibrating':     (180, 180, 180),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── MediaPipe keypoint indices (YOLOv8 pose uses same COCO order) ──
KP = {
    'nose':           0,
    'left_eye':       1,  'right_eye':      2,
    'left_ear':       3,  'right_ear':       4,
    'left_shoulder':  5,  'right_shoulder':  6,
    'left_elbow':     7,  'right_elbow':     8,
    'left_wrist':     9,  'right_wrist':    10,
    'left_hip':      11,  'right_hip':      12,
}

# ── Load seat allocation CSV ───────────────────────────────────
def load_seats(csv_path):
    seats = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found — using placeholder names")
        return seats
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seats[row['seat'].strip().upper()] = {
                'roll': row['roll_number'].strip(),
                'name': row['name'].strip(),
            }
    return seats

seat_data = load_seats(args.seats_csv)

# ── Seat tracker ──────────────────────────────────────────────
class SeatTracker:
    """
    Calibration phase: collect person detections for first N frames,
    compute stable centre per person, sort into seat grid order,
    assign seat IDs left-to-right then front-to-back.

    Tracking phase: each frame, match new detections to known seat
    centres using Hungarian algorithm on Euclidean distance.
    """

    def __init__(self, seat_data, calib_frames=45):
        self.calib_frames  = calib_frames
        self.seat_data     = seat_data
        self.calibrated    = False
        self._calib_buf    = []       # list of per-frame detection lists
        self.seats         = {}       # seat_id → {centre, roll, name, ...}
        self.seat_order    = []       # ordered seat IDs

        # Per-seat behaviour state
        self.label_smooth  = {}       # seat_id → deque of recent labels
        self.stats         = {}       # seat_id → {label: count}
        self.cumulative    = {}       # seat_id → {label: seconds}

        # Examiner: typically the person who stays near the front/sides
        self.examiner_idx  = None

    # ── Calibration ───────────────────────────────────────────

    def feed_calib(self, detections, frame_idx):
        """
        detections: list of (cx, cy, x1, y1, x2, y2, keypoints)
        Returns True when calibration is complete.
        """
        self._calib_buf.append(detections)
        if frame_idx < self.calib_frames:
            return False

        # Average centre per detection track across calib frames
        # Simple approach: take the median frame's detections, sort them
        # by position (y desc = back row first, then x asc = left to right)
        # Use the frame with the most detections as reference
        best_frame = max(self._calib_buf, key=len)
        if not best_frame:
            return False

        # Sort detections: primarily by y (ascending = back of hall first)
        # then by x (ascending = left to right)
        sorted_dets = sorted(best_frame, key=lambda d: (round(d[1] / 60), d[0]))

        # Identify examiner as the person closest to the camera (highest y value)
        # who is significantly lower in frame than students
        # Heuristic: if a person's cy > 0.65 * H they're in foreground = examiner
        H_guess = 360   # approximate, updated properly in main
        students = [d for d in sorted_dets if d[1] < H_guess * 0.65]
        examiners = [d for d in sorted_dets if d[1] >= H_guess * 0.65]

        if not students:
            students = sorted_dets

        # Assign seat IDs from CSV in order
        csv_seats = list(self.seat_data.keys())

        for i, det in enumerate(students):
            cx, cy = det[0], det[1]
            if i < len(csv_seats):
                sid = csv_seats[i]
                info = self.seat_data[sid]
            else:
                sid  = f"S{i+1}"
                info = {'roll': f'Unknown{i+1}', 'name': f'Student {i+1}'}

            self.seats[sid] = {
                'centre':   np.array([cx, cy], dtype=float),
                'roll':     info['roll'],
                'name':     info['name'],
                'last_box': det[2:6],
                'last_kps': det[6],
            }
            self.seat_order.append(sid)
            self.label_smooth[sid] = deque(maxlen=30)  # 1 second of smoothing at 30fps
            self.stats[sid]       = {'Normal Sitting': 0, 'Leaning to Copy': 0, 'Looking Around': 0}
            self.cumulative[sid]  = {'Normal Sitting': 0.0, 'Leaning to Copy': 0.0, 'Looking Around': 0.0}

        # Mark examiners
        self.examiner_centres = [np.array([d[0], d[1]]) for d in examiners]

        self.calibrated = True
        print(f"\n [SeatTracker] Calibrated {len(self.seats)} seats: {self.seat_order}")
        for sid, s in self.seats.items():
            print(f"   {sid}: {s['roll']} — {s['name']}  centre=({s['centre'][0]:.0f},{s['centre'][1]:.0f})")
        return True

    # ── Tracking ──────────────────────────────────────────────

    def update(self, detections, delta_sec):
        """
        Match detections to seats, update state, return
        dict of seat_id → (label, box, keypoints).
        """
        if not detections or not self.seats:
            return {}

        seat_ids    = self.seat_order
        seat_centres = np.array([self.seats[s]['centre'] for s in seat_ids])
        det_centres  = np.array([[d[0], d[1]] for d in detections])

        # Cost matrix: Euclidean distance between each detection and seat centre
        cost = np.linalg.norm(
            det_centres[:, None, :] - seat_centres[None, :, :], axis=2
        )   # shape: (n_detections, n_seats)

        row_ind, col_ind = linear_sum_assignment(cost)

        result = {}
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > 150:    # too far — probably a new person / noise
                continue
            sid = seat_ids[c]
            det = detections[r]
            cx, cy = det[0], det[1]

            # Update centre with exponential moving average (smooth tracking)
            self.seats[sid]['centre'] = (
                0.85 * self.seats[sid]['centre'] + 0.15 * np.array([cx, cy])
            )
            self.seats[sid]['last_box'] = det[2:6]
            self.seats[sid]['last_kps'] = det[6]

            label = self._classify(sid, det[6])
            self.stats[sid][label]      += 1
            self.cumulative[sid][label] += delta_sec

            result[sid] = (label, det[2:6], det[6])

        return result

    # ── Behaviour classification ───────────────────────────────

    def _classify(self, seat_id, kps):
        """
        Classify behaviour from YOLOv8 pose keypoints.

        kps: numpy array of shape (17, 3) — (x, y, confidence)

        Signals:
          Leaning to Copy — shoulders tilted significantly sideways
                            (one shoulder much higher than other)
          Looking Around  — nose far from shoulder midpoint horizontally,
                            or head turned so one ear is hidden
          Normal Sitting  — upright, facing forward
        """
        label = self._pose_label(kps)
        lq    = self.label_smooth[seat_id]
        lq.append(label)
        return Counter(lq).most_common(1)[0][0]

    def _pose_label(self, kps):
        """
        Classifier tuned for TOP-DOWN CCTV angle.

        From above, faces aren't visible so nose/ear signals are
        useless. Instead we use:

        1. Hip-to-shoulder vector direction — tells us which way
           the torso is pointing. If it points sharply sideways
           instead of toward the camera = leaning to copy.

        2. Wrist position relative to hip — if wrist is far
           outside the body width = arm reaching sideways.

        3. Bounding box aspect ratio change — a person leaning
           sideways gets a wider box relative to height.

        Default is Normal Sitting — only flag on clear signals.
        """
        if kps is None or len(kps) < 13:
            return 'Normal Sitting'

        def get(name):
            idx = KP[name]
            k = kps[idx]
            return k if k[2] > 0.25 else None

        ls = get('left_shoulder')
        rs = get('right_shoulder')
        lh = get('left_hip')
        rh = get('right_hip')
        lw = get('left_wrist')
        rw = get('right_wrist')

        if ls is None or rs is None:
            return 'Normal Sitting'

        # ── Shoulder width (horizontal spread) ───────────────
        shoulder_width = abs(float(ls[0]) - float(rs[0])) + 1e-6
        shoulder_mid_x = (float(ls[0]) + float(rs[0])) / 2
        shoulder_mid_y = (float(ls[1]) + float(rs[1])) / 2

        # ── Hip midpoint ──────────────────────────────────────
        if lh is not None and rh is not None:
            hip_mid_x = (float(lh[0]) + float(rh[0])) / 2
            hip_mid_y = (float(lh[1]) + float(rh[1])) / 2

            # Torso lean: horizontal offset between shoulder and hip midpoints
            # From top-down: a seated person has shoulders roughly above hips
            # Leaning sideways = shoulder mid shifts significantly left/right of hip mid
            torso_lean = abs(shoulder_mid_x - hip_mid_x) / shoulder_width
            if torso_lean > 0.55:
                return 'Leaning to Copy'

        # ── Wrist reach: arm extended far to the side ─────────
        for wrist in [lw, rw]:
            if wrist is not None:
                wrist_offset = abs(float(wrist[0]) - shoulder_mid_x) / shoulder_width
                if wrist_offset > 1.4:   # wrist way outside body width
                    return 'Leaning to Copy'

        # ── Shoulder rotation: one shoulder much closer to cam ─
        # From top-down, if torso is rotated sideways one shoulder
        # appears higher in the frame (smaller y) than the other
        shoulder_y_diff = abs(float(ls[1]) - float(rs[1])) / shoulder_width
        if shoulder_y_diff > 0.60:
            return 'Looking Around'

        return 'Normal Sitting'

    # ── Examiner detection ────────────────────────────────────

    def is_examiner(self, cx, cy):
        """Returns True if this detection matches an examiner position."""
        for ec in self.examiner_centres:
            if np.linalg.norm(np.array([cx, cy]) - ec) < 120:
                return True
        return False


# ── Drawing ───────────────────────────────────────────────────

def draw_person_box(frame, x1, y1, x2, y2, label, color, roll, name, kps=None):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label tag above box
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.46, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.46, (0, 0, 0), 1)

    # Roll number below box
    cv2.putText(frame, roll, (x1 + 2, y2 + 14), FONT, 0.36, color, 1)

    # Draw skeleton if keypoints available
    if kps is not None:
        _draw_skeleton(frame, kps, color)


def _draw_skeleton(frame, kps, color):
    """Draw simplified upper-body skeleton."""
    connections = [
        ('left_shoulder',  'right_shoulder'),
        ('left_shoulder',  'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow',     'left_wrist'),
        ('right_elbow',    'right_wrist'),
        ('left_shoulder',  'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('nose',           'left_shoulder'),
        ('nose',           'right_shoulder'),
    ]
    for a, b in connections:
        ia, ib = KP[a], KP[b]
        if ia >= len(kps) or ib >= len(kps):
            continue
        pa, pb = kps[ia], kps[ib]
        if pa[2] > 0.3 and pb[2] > 0.3:
            cv2.line(frame,
                     (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])),
                     color, 1)
    # Draw keypoint dots
    for name in ['nose', 'left_shoulder', 'right_shoulder',
                 'left_elbow', 'right_elbow']:
        idx = KP[name]
        if idx < len(kps) and kps[idx][2] > 0.3:
            cv2.circle(frame, (int(kps[idx][0]), int(kps[idx][1])), 3, color, -1)


def draw_hud(frame, frame_idx, total_frames, video_ts, paused, calibrated, calib_frames):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 22), (15, 15, 15), -1)
    cv2.putText(frame, 'TapNFlo  AI Exam Surveillance',
                (6, 15), FONT, 0.44, (200, 200, 200), 1)
    vts = str(timedelta(seconds=int(video_ts)))
    cv2.putText(frame, vts, (w - 65, 15), FONT, 0.42, (150, 150, 150), 1)

    if not calibrated:
        prog = min(frame_idx / calib_frames, 1.0)
        msg  = f'Calibrating seat positions... {int(prog*100)}%'
        cv2.putText(frame, msg, (w // 2 - 160, 15), FONT, 0.42, (0, 220, 255), 1)

    if paused:
        cv2.putText(frame, '|| PAUSED', (w // 2 - 40, 15), FONT, 0.42, (0, 165, 255), 1)

    # Progress bar
    bw = w - 12
    cv2.rectangle(frame, (6, h - 5), (6 + bw, h - 2), (40, 40, 40), -1)
    prog = int(bw * frame_idx / max(total_frames, 1))
    cv2.rectangle(frame, (6, h - 5), (6 + prog, h - 2), (0, 200, 80), -1)

    # Controls
    cv2.putText(frame, 'SPACE: pause   S: +10s   Q: quit',
                (6, h - 8), FONT, 0.30, (80, 80, 80), 1)

    # Watermark
    cv2.putText(frame, 'TapNFlo AI Cam', (w - 120, h - 20), FONT, 0.36, (100, 100, 100), 1)


def fmt(secs):
    s = int(secs)
    return f"{s // 60}:{s % 60:02d}"


def print_report(tracker, analysed_sec):
    print("\n" + "═" * 60)
    print("  TapNFloAIcam — Session Report")
    print("═" * 60)
    print(f"  Video    : {args.video}")
    print(f"  Analysed : {fmt(analysed_sec)}")
    print(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("─" * 60)

    flagged = []
    for sid in tracker.seat_order:
        s      = tracker.seats[sid]
        counts = tracker.stats[sid]
        total  = sum(counts.values())
        if total == 0:
            continue
        print(f"\n  Seat {sid}  |  {s['roll']}  |  {s['name']}")
        for label, count in counts.items():
            pct = count / total * 100
            bar = '█' * int(pct / 4)
            print(f"    {label:20s}: {pct:5.1f}%  {bar}")
        sus = (counts['Leaning to Copy'] + counts['Looking Around']) / total * 100
        if sus > 10:
            flagged.append((sid, s['roll'], s['name'], sus))

    print("\n" + "─" * 60)
    if flagged:
        print("  ⚑ Flagged (>10% suspicious time):")
        for sid, roll, name, pct in flagged:
            print(f"    Seat {sid}  {roll}  {name}  —  {pct:.1f}% suspicious")
    else:
        print("  ✓ No students flagged.")
    print("═" * 60 + "\n")


# ── Tiling helpers ────────────────────────────────────────────

def _get_tiles(frame):
    """
    Returns list of (tile_img, origin_x, origin_y, scale).

    Splits the frame into overlapping quadrants + full frame.
    Each tile is upscaled to 640px wide so YOLO sees distant
    students at a reasonable resolution.

    For a corner-mounted camera this means:
      - Full frame pass: catches everyone at low res
      - Top-left tile:   catches far/distant students at higher res
      - Top-right tile:  catches edge students at higher res
      - Bottom tiles:    catches close foreground students
    """
    H, W = frame.shape[:2]
    TARGET = 640
    tiles  = []

    # Full frame (baseline)
    scale_full = TARGET / W
    full = cv2.resize(frame, (TARGET, int(H * scale_full)))
    tiles.append((full, 0, 0, scale_full))

    # Quadrants with 20% overlap
    overlap = 0.20
    regions = [
        (0,         0,         int(W*0.55), int(H*0.55)),   # top-left
        (int(W*0.45), 0,       W,           int(H*0.55)),   # top-right
        (0,         int(H*0.45), int(W*0.55), H),           # bottom-left
        (int(W*0.45), int(H*0.45), W,       H),             # bottom-right
    ]
    for x1, y1, x2, y2 in regions:
        crop  = frame[y1:y2, x1:x2]
        cW    = x2 - x1
        scale = TARGET / cW
        tile  = cv2.resize(crop, (TARGET, int((y2-y1)*scale)))
        # Enhance contrast for far/dark regions
        lab   = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        tile  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        tiles.append((tile, x1, y1, scale))

    return tiles


def _nms(boxes, iou_thresh=0.45):
    """Simple NMS to remove duplicate detections across tiles."""
    if not boxes:
        return []
    boxes  = np.array(boxes, dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = areas.argsort()[::-1]
    kept   = []
    while len(order):
        i = order[0]
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1  = np.maximum(x1[i], x1[rest])
        iy1  = np.maximum(y1[i], y1[rest])
        ix2  = np.minimum(x2[i], x2[rest])
        iy2  = np.minimum(y2[i], y2[rest])
        iw   = np.maximum(0, ix2 - ix1)
        ih   = np.maximum(0, iy2 - iy1)
        inter = iw * ih
        iou  = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thresh]
    return kept


# ── Main ──────────────────────────────────────────────────────

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed.")
        print("Run: pip install ultralytics\n")
        sys.exit(1)

    print("\n Loading YOLOv8 pose model...")
    model = YOLO('yolov8s-pose.pt')   # small model — much better than nano for crowded scenes

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open: {args.video}\n")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms     = max(1, int((1000 / fps) / args.speed))

    print(f" File    : {args.video}")
    print(f" Duration: {fmt(total_frames/fps)}  |  {W}x{H}  |  {fps:.0f}fps")
    print(f" Calibration: first {args.calib_frames} frames (~{args.calib_frames/fps:.1f}s)")
    print(f" Sit students normally during the first {args.calib_frames/fps:.1f} seconds\n")

    writer = None
    if not args.no_save:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        print(f" Saving output to: {args.output}\n")

    tracker   = SeatTracker(seat_data, calib_frames=args.calib_frames)
    tracker.examiner_centres = []  # will be set after calibration

    prev_time  = None
    frame_idx  = 0
    paused     = False
    frame      = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        if frame is None:
            break

        video_ts = frame_idx / fps
        delta    = 1.0 / fps   # time per frame

        # ── Tiled detection for corner-mounted cameras ────────
        # Split frame into overlapping tiles so distant/edge students
        # get a full-resolution detection pass instead of being tiny
        # specks in the full frame.
        all_boxes = []
        all_kps   = []

        tiles = _get_tiles(frame)
        for tile_img, tx, ty, scale in tiles:
            res = model(tile_img, verbose=False, conf=0.20, iou=0.5)
            r   = res[0]
            if r.boxes is None or r.keypoints is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            kps   = r.keypoints.data.cpu().numpy()
            for i in range(len(boxes)):
                # Map tile coordinates back to full frame
                x1 = (boxes[i][0] / scale) + tx
                y1 = (boxes[i][1] / scale) + ty
                x2 = (boxes[i][2] / scale) + tx
                y2 = (boxes[i][3] / scale) + ty
                mapped_kps = kps[i].copy() if i < len(kps) else None
                if mapped_kps is not None:
                    mapped_kps[:, 0] = mapped_kps[:, 0] / scale + tx
                    mapped_kps[:, 1] = mapped_kps[:, 1] / scale + ty
                all_boxes.append([x1, y1, x2, y2])
                all_kps.append(mapped_kps)

        # NMS across all tiles to remove duplicates
        detections = []
        if all_boxes:
            kept = _nms(all_boxes, iou_thresh=0.45)
            for i in kept:
                x1, y1, x2, y2 = all_boxes[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append((cx, cy, x1, y1, x2, y2, all_kps[i]))

        # ── Calibration phase ─────────────────────────────────
        if not tracker.calibrated:
            # Pass H so examiner threshold is correct
            tracker.H = H
            done = tracker.feed_calib(detections, frame_idx)

            # Draw raw detections during calibration
            for det in detections:
                cx, cy, x1, y1, x2, y2, _ = det
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)),
                              COL['Calibrating'], 1)
                cv2.putText(frame, 'Calibrating...',
                            (int(x1)+2, int(y1)-4), FONT, 0.38, COL['Calibrating'], 1)

            draw_hud(frame, frame_idx, total_frames, video_ts,
                     paused, False, args.calib_frames)

        # ── Tracking phase ────────────────────────────────────
        else:
            seat_results = tracker.update(detections, delta)

            # Draw matched seats
            for sid, (label, box, kps) in seat_results.items():
                x1, y1, x2, y2 = box
                color = COL[label]
                s     = tracker.seats[sid]
                draw_person_box(frame, x1, y1, x2, y2,
                                label, color, s['roll'], s['name'], kps)

            # Draw examiner boxes for unmatched detections
            matched_boxes = {sid: seat_results[sid][1] for sid in seat_results}
            for det in detections:
                cx, cy, x1, y1, x2, y2, _ = det
                if tracker.is_examiner(cx, cy):
                    cv2.rectangle(frame,
                                  (int(x1),int(y1)),(int(x2),int(y2)),
                                  COL['Examiner'], 2)
                    cv2.putText(frame, 'Examiner',
                                (int(x1)+3, int(y2)-6),
                                FONT, 0.46, COL['Examiner'], 1)

            draw_hud(frame, frame_idx, total_frames, video_ts,
                     paused, True, args.calib_frames)

        if writer:
            writer.write(frame)

        cv2.imshow('TapNFloAIcam — Video Analysis', frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            skip = int(fps * 10)
            frame_idx += skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print_report(tracker, frame_idx / fps)


if __name__ == '__main__':
    main()
