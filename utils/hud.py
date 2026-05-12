# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — utils/hud.py
#
# All OpenCV drawing logic lives here, completely separate from
# detection logic. This makes it easy to restyle the HUD without
# touching any AI code.
# ──────────────────────────────────────────────────────────────

import cv2
from datetime import datetime
import config as cfg


COL_BG        = (15,  15,  15 )
COL_TEXT      = (200, 200, 200)
COL_MUTED     = (120, 120, 120)
COL_SIGNAL    = (80,  200, 80 )
COL_ALERT_HI  = (0,   0,   220)
COL_ALERT_MED = (0,   165, 255)
COL_ALERT_LO  = (80,  200, 80 )
COL_INFO      = (200, 180, 80 )

SEVERITY_COLORS = {
    "HIGH":   COL_ALERT_HI,
    "MEDIUM": COL_ALERT_MED,
    "LOW":    COL_ALERT_LO,
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.40
FONT_MED   = 0.50
FONT_LARGE = 0.60


def draw_top_bar(frame, status_text, status_color, student=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 54), COL_BG, -1)
    cv2.putText(frame, "TapNFlo",
                (12, 22), FONT, FONT_LARGE, (255, 255, 255), 1)
    cv2.putText(frame, "AI Exam Surveillance  |  prototype v0.1",
                (100, 22), FONT, FONT_MED, COL_MUTED, 1)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 95, 22), FONT, FONT_MED, COL_MUTED, 1)
    cv2.rectangle(frame, (10, 30), (195, 50), status_color, -1)
    cv2.putText(frame, f"  STATUS: {status_text}",
                (12, 45), FONT, FONT_SMALL, (255, 255, 255), 1)
    if student:
        label = f"Seat {student.seat}  |  {student.roll_number}  |  {student.name}"
        cv2.putText(frame, label, (205, 45), FONT, FONT_SMALL, COL_INFO, 1)


def draw_score_bar(frame, tracker, status_color):
    """
    Bar shows cumulative suspicious time as a fraction of the
    flag threshold — so it fills up across the whole exam as
    behaviour accumulates, not per-incident.
    """
    import config as cfg
    h, w = frame.shape[:2]
    bx, by, bw, bh = w - 230, 30, 210, 12
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill = int(bw * tracker.progress())
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), status_color, -1)
    total = int(tracker.cumulative_down_sec + tracker.cumulative_side_sec)
    limit = int(cfg.CUMULATIVE_FLAG_SEC)
    m, s  = total // 60, total % 60
    lm, ls = limit // 60, limit % 60
    cv2.putText(frame,
                f"Cumulative time: {m}:{s:02d} / {lm}:{ls:02d}",
                (bx, by + bh + 14), FONT, FONT_SMALL, COL_MUTED, 1)


def draw_student_card(frame, student):
    """Top-right card — seat, roll number, name of active student."""
    if not student:
        return
    h, w = frame.shape[:2]
    cx, cy, cw, ch = w - 310, 58, 300, 76
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), COL_BG, -1)
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (50, 50, 50), 1)
    cv2.putText(frame, "Currently monitoring",
                (cx + 8, cy + 16), FONT, 0.38, COL_MUTED, 1)
    cv2.putText(frame, f"Seat :  {student.seat}",
                (cx + 8, cy + 34), FONT, FONT_SMALL, COL_TEXT, 1)
    cv2.putText(frame, f"Roll :  {student.roll_number}",
                (cx + 8, cy + 50), FONT, FONT_SMALL, COL_TEXT, 1)
    cv2.putText(frame, f"Name :  {student.name}",
                (cx + 8, cy + 66), FONT, FONT_SMALL, COL_INFO, 1)


def draw_hall_summary(frame, summary, flagged_list):
    """Hall-wide status counts + list of flagged roll numbers."""
    h, w = frame.shape[:2]
    px, py, pw = w - 310, 142, 300
    cv2.rectangle(frame, (px, py), (px + pw, py + 92), COL_BG, -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + 92), (50, 50, 50), 1)
    cv2.putText(frame, "Hall summary",
                (px + 8, py + 16), FONT, 0.38, COL_MUTED, 1)
    col_w = pw // 3
    labels = [
        ("NORMAL",     summary.get("NORMAL",     0), COL_ALERT_LO),
        ("SUSPICIOUS", summary.get("SUSPICIOUS", 0), COL_ALERT_MED),
        ("FLAGGED",    summary.get("FLAGGED",    0), COL_ALERT_HI),
    ]
    for i, (lbl, count, color) in enumerate(labels):
        x = px + 8 + i * col_w
        cv2.putText(frame, str(count),  (x, py + 42), FONT, 0.7,  color, 1)
        cv2.putText(frame, lbl,         (x, py + 58), FONT, 0.32, color, 1)
    if flagged_list:
        rolls = "Flagged: " + "  ".join(s.roll_number for s in flagged_list[:4])
        cv2.putText(frame, rolls, (px + 8, py + 78), FONT, 0.36, COL_ALERT_HI, 1)
    else:
        cv2.putText(frame, "No flags raised",
                    (px + 8, py + 78), FONT, 0.36, COL_ALERT_LO, 1)


def draw_signals_panel(frame, signals):
    h, w = frame.shape[:2]
    px, py = 10, h - 175
    cv2.rectangle(frame, (px, py), (px + 270, h - 10), COL_BG, -1)
    cv2.putText(frame, "Detected signals",
                (px + 8, py + 18), FONT, FONT_SMALL, COL_MUTED, 1)
    if not signals:
        cv2.putText(frame, "  None yet",
                    (px + 8, py + 38), FONT, FONT_SMALL, COL_MUTED, 1)
        return
    for i, sig in enumerate(signals[-cfg.MAX_SIGNALS_SHOWN:]):
        cv2.putText(frame, f"  {sig}",
                    (px + 8, py + 36 + i * 22), FONT, FONT_SMALL, COL_SIGNAL, 1)


def draw_alert_log(frame, alert_log):
    h, w = frame.shape[:2]
    ax, ay = w - 310, h - 175
    cv2.rectangle(frame, (ax, ay), (w - 10, h - 10), COL_BG, -1)
    cv2.putText(frame, "Alert log",
                (ax + 8, ay + 18), FONT, FONT_SMALL, COL_MUTED, 1)
    if not alert_log:
        cv2.putText(frame, "  No alerts",
                    (ax + 8, ay + 38), FONT, FONT_SMALL, COL_MUTED, 1)
        return
    for i, (ts, severity, msg) in enumerate(alert_log[:5]):
        color = SEVERITY_COLORS.get(severity, COL_MUTED)
        cv2.putText(frame, f"{ts}  [{severity}]",
                    (ax + 8, ay + 36 + i * 26), FONT, 0.36, color, 1)
        cv2.putText(frame, f"  {msg}",
                    (ax + 8, ay + 50 + i * 26), FONT, 0.36, COL_TEXT, 1)


def draw_angle_debug(frame, head_det, pitch, yaw):
    if not cfg.SHOW_ANGLE_DEBUG:
        return
    h = frame.shape[0]
    info = head_det.neutral_info()
    cv2.putText(frame,
                f"raw pitch: {pitch:+.1f}  yaw: {yaw:+.1f}   {info}",
                (10, h - 185), FONT, FONT_SMALL, (90, 90, 90), 1)


def draw_controls_hint(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame,
                "Q: quit   R: reset student   LEFT / RIGHT arrow: switch seat",
                (w // 2 - 195, h - 10), FONT, 0.38, (70, 70, 70), 1)


def draw_calibration(frame, message: str):
    """Shows a calibration overlay until neutral pose is locked in."""
    if not message:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - 30), (w, h//2 + 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, message,
                (w//2 - 200, h//2 + 8), FONT, 0.65, (0, 220, 200), 2)


def draw_all(frame, tracker, pitch, yaw, seat_manager=None, head_det=None):
    """Single call — renders the full HUD onto the frame."""
    status_text, status_color = tracker.status()
    student = seat_manager.active_student() if seat_manager else None

    draw_top_bar(frame, status_text, status_color, student)
    draw_score_bar(frame, tracker, status_color)
    draw_student_card(frame, student)

    if seat_manager:
        draw_hall_summary(
            frame,
            seat_manager.summary(),
            seat_manager.flagged_students()
        )

    draw_signals_panel(frame, tracker.signals)
    draw_alert_log(frame, tracker.alert_log)
    if head_det:
        draw_angle_debug(frame, head_det, pitch, yaw)
    draw_controls_hint(frame)
