# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — main.py
#
# Entry point. Run with:  python main.py
#
# Controls:
#   Q          — quit
#   R          — reset current student's score
#   LEFT/RIGHT — switch which seat/student is in focus
# ──────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp

import config as cfg
from core.tracker      import BehaviourTracker
from core.detectors    import HeadPoseDetector, ShoulderDetector
from core.seat_manager import SeatManager
from utils.hud         import draw_all

# ── MediaPipe models ──────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces            = 1,
    refine_landmarks         = True,
    min_detection_confidence = cfg.FACE_DETECTION_CONFIDENCE,
    min_tracking_confidence  = cfg.FACE_TRACKING_CONFIDENCE,
)
pose_model = mp_pose.Pose(
    min_detection_confidence = cfg.POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence  = cfg.POSE_TRACKING_CONFIDENCE,
)

# ── PROTOTYPE NOTE ────────────────────────────────────────────
# This is a single-camera proof of concept using a laptop webcam.
# The operator manually switches between seat focuses using arrow
# keys to simulate monitoring different students one at a time.
#
# Production version would require:
#   • Multiple RTSP camera feeds (one per hall zone)
#   • A one-time calibration step mapping camera regions to seat
#     coordinates automatically — no manual switching needed
#   • Sample exam footage to fine-tune detection thresholds and
#     reduce false positives under real hall lighting conditions
#   • A parallel multi-tracker pipeline (one BehaviourTracker
#     per student) running across all camera feeds simultaneously
# ──────────────────────────────────────────────────────────────

# ── Components ────────────────────────────────────────────────
tracker      = BehaviourTracker()
head_det     = HeadPoseDetector()
shoulder_det = ShoulderDetector()
seats        = SeatManager("seat_allocation.csv")


def sync_tracker_to_seat():
    """
    When switching seats, save the current tracker state into the
    outgoing student record, then load the incoming student's
    saved state into the tracker.
    """
    incoming = seats.active_student()
    if incoming:
        tracker.cumulative_down_sec = incoming.cumulative_down
        tracker.cumulative_side_sec = incoming.cumulative_side
        tracker.tap_count           = incoming.tap_count
        tracker.signals             = list(incoming.signals)
        tracker.last_alert_time     = {}
        tracker.shoulder_history.clear()
        tracker.tap_cooldown        = 0


def main():
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check CAMERA_INDEX in config.py")
        return

    print("\n TapNFloAIcam started.")
    print(" Controls:  Q = quit  |  R = reset student  |  LEFT/RIGHT = switch seat\n")

    pitch, yaw = 0.0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        face_results = face_mesh.process(rgb)
        pose_results = pose_model.process(rgb)
        rgb.flags.writeable = True

        # ── Face mesh + head pose ──────────────────────────────
        if face_results.multi_face_landmarks:
            fl = face_results.multi_face_landmarks[0]
            if cfg.SHOW_FACE_MESH:
                mp_drawing.draw_landmarks(
                    frame, fl,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec    = None,
                    connection_drawing_spec  = mp_styles.get_default_face_mesh_contours_style()
                )
            pitch, yaw = head_det.get_angles(fl, w, h)
            head_det.analyse(pitch, yaw, tracker, frame)

        # ── Pose skeleton + shoulder tap ──────────────────────
        if pose_results.pose_landmarks:
            if cfg.SHOW_POSE_SKELETON:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec   = mp_styles.get_default_pose_landmarks_style()
                )
            shoulder_det.analyse(pose_results.pose_landmarks, h, tracker)

        # ── Persist tracker state into active student record ──
        if seats.active_student():
            seats.update_active(tracker, tracker.status()[0])

        # ── HUD ───────────────────────────────────────────────
        draw_all(frame, tracker, pitch, yaw, seats, head_det)

        # ── Persist tracker state into active student record ──

        cv2.imshow("TapNFloAIcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            # Reset current student's score and signals
            tracker.reset()
            seats.reset_active()
            pitch, yaw = 0.0, 0.0
            print(f" Reset: {seats.active_student().roll_number if seats.active_student() else 'unknown'}")

        elif key == 83 or key == ord('d'):   # RIGHT arrow or D
            seats.update_active(tracker, tracker.status()[0])
            seats.next_seat()
            sync_tracker_to_seat()
            head_det.recalibrate()   # recalibrate for new student's posture
            s = seats.active_student()
            print(f" Switched to: Seat {s.seat} — {s.roll_number} — {s.name}")

        elif key == 81 or key == ord('a'):   # LEFT arrow or A
            seats.update_active(tracker, tracker.status()[0])
            seats.prev_seat()
            sync_tracker_to_seat()
            head_det.recalibrate()   # recalibrate for new student's posture
            s = seats.active_student()
            print(f" Switched to: Seat {s.seat} — {s.roll_number} — {s.name}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pose_model.close()
    print("\n Session ended.")
    _print_session_summary()


def _print_session_summary():
    """Print a final summary of all flagged students to the terminal."""
    flagged = seats.flagged_students()
    if not flagged:
        print(" No students flagged during this session.")
        return
    print(f"\n ── Session Summary ── {len(flagged)} student(s) flagged ──")
    for s in flagged:
        total = int(s.cumulative_down + s.cumulative_side)
        m, sec = total // 60, total % 60
        print(f"  Seat {s.seat}  |  {s.roll_number}  |  {s.name}  |  {m}:{sec:02d} cumulative")
        for sig in s.signals:
            print(f"    • {sig}")
    print()


if __name__ == "__main__":
    main()
