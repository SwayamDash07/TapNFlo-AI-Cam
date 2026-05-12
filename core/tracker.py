# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — core/tracker.py
#
# BehaviourTracker uses CUMULATIVE TIME to decide status —
# not per-incident scoring. This mirrors how real proctoring
# works: one peek is human, two minutes of peeking is cheating.
#
# How it works:
#   • Every frame where a suspicious posture is detected,
#     the elapsed time is added to that behaviour's cumulative
#     counter (e.g. cumulative_down_sec += 0.033 per frame).
#   • Status is derived purely from those cumulative totals
#     against thresholds defined in config.py.
#   • Shoulder taps are discrete events (not time-based) so
#     they use a separate incident counter.
# ──────────────────────────────────────────────────────────────

import time
from collections import deque
from datetime import datetime
import config as cfg


class BehaviourTracker:
    def __init__(self):
        # ── Cumulative time accumulators (seconds) ────────────
        self.cumulative_down_sec  = 0.0   # total time looking down
        self.cumulative_side_sec  = 0.0   # total time looking sideways

        # ── Discrete incident counters ────────────────────────
        self.tap_count            = 0     # number of shoulder tap events

        # ── Display state ─────────────────────────────────────
        self.signals              = []    # human-readable signal labels
        self.alert_log            = []    # (timestamp, severity, message)
        self.last_alert_time      = {}    # cooldown per alert label

        # ── Shoulder / tap tracking ───────────────────────────
        self.shoulder_history     = deque(maxlen=cfg.SHOULDER_HISTORY_FRAMES)
        self.tap_cooldown         = 0

    # ── Time accumulation (called every frame by detectors) ───

    def accumulate_down(self, delta_sec: float):
        """Add frame time to the looking-down accumulator."""
        self.cumulative_down_sec += delta_sec
        self._update_signals()

    def accumulate_side(self, delta_sec: float):
        """Add frame time to the sideways-glance accumulator."""
        self.cumulative_side_sec += delta_sec
        self._update_signals()

    def add_tap(self):
        """Record one shoulder tap event."""
        self.tap_count += 1
        self._update_signals()
        self._log_alert("Shoulder tap", "HIGH")

    # ── Alert logging ─────────────────────────────────────────

    def _log_alert(self, label: str, severity: str):
        """Append to alert log with cooldown to avoid spam."""
        now = time.time()
        if label in self.last_alert_time:
            if now - self.last_alert_time[label] < cfg.ALERT_COOLDOWN_SEC:
                return
        self.last_alert_time[label] = now
        ts = datetime.now().strftime("%H:%M:%S")
        self.alert_log.insert(0, (ts, severity, label))
        if len(self.alert_log) > cfg.MAX_ALERT_LOG:
            self.alert_log.pop()

    # ── Signal label sync ─────────────────────────────────────

    def _update_signals(self):
        """
        Rebuild the human-readable signals list from current
        accumulators so the HUD always shows up-to-date totals.
        """
        self.signals = []

        if self.cumulative_down_sec >= 1.0:
            secs = int(self.cumulative_down_sec)
            self.signals.append(
                f"Looking down — {_fmt(secs)}"
            )
            self._log_alert("Looking down", "MEDIUM")

        if self.cumulative_side_sec >= 1.0:
            secs = int(self.cumulative_side_sec)
            self.signals.append(
                f"Sideways glance — {_fmt(secs)}"
            )
            self._log_alert("Sideways glance", "MEDIUM")

        if self.tap_count > 0:
            self.signals.append(
                f"Shoulder tap — {self.tap_count}x"
            )

    # ── Status ────────────────────────────────────────────────

    def status(self) -> tuple:
        """
        Derives status from cumulative times and tap count.
        Returns (label, BGR_color).

        FLAGGED    if: total down+side time >= CUMULATIVE_FLAG_SEC
                       OR taps >= SHOULDER_TAP_FLAG_COUNT
        SUSPICIOUS if: total down+side time >= CUMULATIVE_WARN_SEC
                       OR taps >= SHOULDER_TAP_WARN_COUNT
        NORMAL     otherwise
        """
        total_time = self.cumulative_down_sec + self.cumulative_side_sec

        if (total_time >= cfg.CUMULATIVE_FLAG_SEC or
                self.tap_count >= cfg.SHOULDER_TAP_FLAG_COUNT):
            return "FLAGGED",    (0,   0,   220)

        if (total_time >= cfg.CUMULATIVE_WARN_SEC or
                self.tap_count >= cfg.SHOULDER_TAP_WARN_COUNT):
            return "SUSPICIOUS", (0,   165, 255)

        return "NORMAL",         (0,   200, 80)

    def progress(self) -> float:
        """
        Returns 0.0–1.0 representing how close the student is
        to the FLAGGED threshold — used to draw the score bar.
        """
        total_time = self.cumulative_down_sec + self.cumulative_side_sec
        return min(total_time / cfg.CUMULATIVE_FLAG_SEC, 1.0)

    def reset(self):
        """Hard reset — bound to the R key during demo."""
        self.__init__()


# ── Helpers ───────────────────────────────────────────────────

def _fmt(secs: int) -> str:
    """Format seconds as m:ss for display (e.g. 1:45)."""
    return f"{secs // 60}:{secs % 60:02d}"
