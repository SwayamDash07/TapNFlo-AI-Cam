# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — Configuration
# Tune all detection thresholds and scoring weights here.
# No need to touch any other file to adjust sensitivity.
# ──────────────────────────────────────────────────────────────

# ── Camera ────────────────────────────────────────────────────
CAMERA_INDEX        = 0       # 0 = default webcam, change if using external cam
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720

# ── Head pose thresholds (degrees, relative to neutral) ───────
# These are DEVIATIONS from the calibrated neutral position,
# not absolute angles — so they work for any camera height.
#
# After ~2s of sitting normally the system locks in neutral.
# Then: how far must the head deviate to count as suspicious?
#
PITCH_DOWN_THRESHOLD    = 12    # degrees below neutral → "looking down"
YAW_SIDE_THRESHOLD      = 18    # degrees left/right of neutral → "sideways glance"

# How long (seconds) head must stay deviated before accumulating
LOOK_DOWN_DURATION      = 2.0   # must look down for 2s straight before counting
LOOK_SIDE_DURATION      = 1.5   # must look sideways for 1.5s straight before counting

# ── Shoulder movement (tap detection) ────────────────────────
SHOULDER_MOVE_THRESHOLD = 0.06  # 6% of frame height — raised to avoid breathing FP
SHOULDER_HISTORY_FRAMES = 30
TAP_COOLDOWN_FRAMES     = 60    # ~2 seconds at 30fps before re-triggering

# ── Cumulative time thresholds (seconds) ──────────────────────
#
# Instead of flagging on a single incident, the system accumulates
# total time the student has spent in each suspicious posture.
# Peeking once is fine — peeking for 2 minutes total is not.
#
# Example: student looks down for 20s, then straight for 30s,
# then down again for 20s → cumulative = 40s → still SUSPICIOUS.
# Once it crosses CUMULATIVE_FLAG_SEC → FLAGGED.

CUMULATIVE_WARN_SEC     = 30    # total seconds looking down/sideways → SUSPICIOUS
CUMULATIVE_FLAG_SEC     = 120   # total seconds looking down/sideways → FLAGGED (2 min)

# Shoulder taps are incident-based (not time-based) since a tap
# is a discrete action, not a sustained posture.
SHOULDER_TAP_WARN_COUNT = 2     # taps to reach SUSPICIOUS
SHOULDER_TAP_FLAG_COUNT = 4     # taps to reach FLAGGED

ALERT_COOLDOWN_SEC      = 10.0  # minimum seconds between same alert being logged

# ── MediaPipe confidence ──────────────────────────────────────
FACE_DETECTION_CONFIDENCE   = 0.6
FACE_TRACKING_CONFIDENCE    = 0.6
POSE_DETECTION_CONFIDENCE   = 0.6
POSE_TRACKING_CONFIDENCE    = 0.6

# ── HUD / Display ─────────────────────────────────────────────
SHOW_FACE_MESH      = True   # draw face landmark mesh
SHOW_POSE_SKELETON  = True   # draw body pose skeleton
SHOW_ANGLE_DEBUG    = True   # show raw pitch/yaw values (useful for demos)
MAX_ALERT_LOG       = 6      # number of alerts shown on screen
MAX_SIGNALS_SHOWN   = 5      # number of signals shown on screen
