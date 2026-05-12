# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — core/detectors.py
#
# KEY DESIGN: Deviation-from-neutral detection.
#
# Instead of using absolute pitch/yaw angles (which break when
# the camera is mounted at a different height, like a CCTV),
# the system first CALIBRATES a neutral baseline for each student
# over the first few seconds. After that, only DEVIATIONS from
# that baseline are measured.
#
# This means:
#   - Works at any camera angle (eye level, ceiling mount, CCTV)
#   - Each student gets their own personal neutral — no one-size
#     fits all threshold
#   - Natural posture variation doesn't trigger false positives
# ──────────────────────────────────────────────────────────────

import time
import numpy as np
import config as cfg


class HeadPoseDetector:
    """
    Calibration phase (first CALIBRATION_SEC seconds):
      Collects pitch/yaw samples while student sits normally.
      Computes mean as the personal neutral baseline.

    Detection phase (after calibration):
      Only flags if head deviates from baseline by more than
      the configured threshold for longer than the min duration.
    """

    CALIBRATION_SEC = 3.0   # seconds to collect neutral samples

    _PTS_3D = np.array([
        [ 0.0,    0.0,    0.0  ],
        [ 0.0,  -63.6,  -12.5 ],
        [-43.3,  32.7,  -26.0 ],
        [ 43.3,  32.7,  -26.0 ],
        [-28.9, -28.9,  -24.1 ],
        [ 28.9, -28.9,  -24.1 ],
    ], dtype=np.float64)

    _LM_IDX = [1, 152, 33, 263, 61, 291]

    def __init__(self):
        # Calibration state
        self._calibrated        = False
        self._calib_start       = None
        self._calib_pitches     = []
        self._calib_yaws        = []
        self._neutral_pitch     = 0.0
        self._neutral_yaw       = 0.0

        # Streak timers (how long head has been deviated continuously)
        self._down_streak_start = None
        self._side_streak_start = None
        self._last_frame_time   = time.time()

    def get_angles(self, face_landmarks, frame_w: int, frame_h: int):
        """Returns (pitch_deg, yaw_deg) or (0, 0) if solvePnP fails."""
        import cv2
        pts_2d = np.array([
            [face_landmarks.landmark[i].x * frame_w,
             face_landmarks.landmark[i].y * frame_h]
            for i in self._LM_IDX
        ], dtype=np.float64)

        focal      = frame_w
        cam_matrix = np.array([
            [focal, 0,     frame_w / 2],
            [0,     focal, frame_h / 2],
            [0,     0,     1          ]
        ], dtype=np.float64)

        dist = np.zeros((4, 1))
        success, rot_vec, _ = cv2.solvePnP(
            self._PTS_3D, pts_2d, cam_matrix, dist
        )
        if not success:
            return 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj        = np.hstack((rot_mat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        return float(euler[0]), float(euler[1])

    def analyse(self, pitch: float, yaw: float, tracker, frame):
        """
        Runs calibration first, then deviation detection.
        Draws a calibration overlay on the frame during warmup.
        """
        now   = time.time()
        delta = min(now - self._last_frame_time, 0.5)
        self._last_frame_time = now

        # ── Calibration phase ─────────────────────────────────
        if not self._calibrated:
            self._run_calibration(pitch, yaw, now, frame)
            return   # don't detect during calibration

        # ── Deviation from neutral ────────────────────────────
        pitch_dev = self._neutral_pitch - pitch   # positive = looking down
        yaw_dev   = abs(yaw - self._neutral_yaw)  # absolute = either side

        # ── Looking DOWN ──────────────────────────────────────
        if pitch_dev > cfg.PITCH_DOWN_THRESHOLD:
            if self._down_streak_start is None:
                self._down_streak_start = now
            elif now - self._down_streak_start >= cfg.LOOK_DOWN_DURATION:
                tracker.accumulate_down(delta)
        else:
            self._down_streak_start = None

        # ── Looking SIDEWAYS ──────────────────────────────────
        if yaw_dev > cfg.YAW_SIDE_THRESHOLD:
            if self._side_streak_start is None:
                self._side_streak_start = now
            elif now - self._side_streak_start >= cfg.LOOK_SIDE_DURATION:
                tracker.accumulate_side(delta)
        else:
            self._side_streak_start = None

    def _run_calibration(self, pitch: float, yaw: float, now: float, frame):
        """Collect neutral samples and draw progress on screen."""
        import cv2

        if self._calib_start is None:
            self._calib_start = now

        self._calib_pitches.append(pitch)
        self._calib_yaws.append(yaw)

        elapsed  = now - self._calib_start
        progress = min(elapsed / self.CALIBRATION_SEC, 1.0)

        # Draw calibration overlay
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), 8)

        msg1 = "Calibrating neutral position..."
        msg2 = "Look straight at the camera and sit normally"
        cv2.putText(frame, msg1,
                    (w // 2 - 200, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(frame, msg2,
                    (w // 2 - 240, h // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Progress bar
        bx, by, bw, bh = w // 2 - 150, h // 2 + 40, 300, 14
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * progress), by + bh),
                      (0, 220, 255), -1)

        if elapsed >= self.CALIBRATION_SEC:
            self._neutral_pitch = float(np.mean(self._calib_pitches))
            self._neutral_yaw   = float(np.mean(self._calib_yaws))
            self._calibrated    = True
            print(f" Calibrated — neutral pitch: {self._neutral_pitch:.1f}  yaw: {self._neutral_yaw:.1f}")

    def recalibrate(self):
        """Called when switching seats — resets calibration for new student."""
        self.__init__()

    @property
    def is_calibrated(self):
        return self._calibrated

    def neutral_info(self):
        """Returns neutral angles as string for debug display."""
        if not self._calibrated:
            return "calibrating..."
        return f"neutral  pitch: {self._neutral_pitch:+.1f}  yaw: {self._neutral_yaw:+.1f}"


class ShoulderDetector:
    """
    Detects shoulder tapping via sudden vertical movement.

    Uses a RELATIVE delta within a rolling window — so the
    baseline shoulder height doesn't matter (works at any
    camera angle). Only sharp sudden movements trigger it,
    not slow postural shifts.

    Threshold is raised significantly to avoid breathing /
    natural movement false positives.
    """

    def analyse(self, pose_landmarks, frame_h: int, tracker):
        if pose_landmarks is None:
            return

        import mediapipe as mp
        lm           = pose_landmarks.landmark
        PoseLandmark = mp.solutions.pose.PoseLandmark

        # Use visibility score — ignore if shoulders not clearly visible
        left  = lm[PoseLandmark.LEFT_SHOULDER]
        right = lm[PoseLandmark.RIGHT_SHOULDER]

        if left.visibility < 0.7 or right.visibility < 0.7:
            return   # shoulders not clearly in frame, skip

        avg_y = ((left.y + right.y) / 2) * frame_h
        tracker.shoulder_history.append(avg_y)

        if len(tracker.shoulder_history) < 20:
            return   # need more history

        arr   = np.array(tracker.shoulder_history)
        delta = float(np.max(arr) - np.min(arr))

        # Only count sharp sudden spikes — use std deviation to
        # distinguish a tap (sharp spike) from slow drift
        std = float(np.std(arr))

        if (delta > frame_h * cfg.SHOULDER_MOVE_THRESHOLD and
                std > frame_h * 0.008):   # must be spiky, not just drifting
            if tracker.tap_cooldown <= 0:
                tracker.add_tap()
                tracker.tap_cooldown = cfg.TAP_COOLDOWN_FRAMES

        if tracker.tap_cooldown > 0:
            tracker.tap_cooldown -= 1
