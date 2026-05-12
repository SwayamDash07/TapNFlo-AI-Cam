# ──────────────────────────────────────────────────────────────
# TapNFloAIcam — core/seat_manager.py
#
# Loads seat_allocation.csv and maps each seat label (e.g. "B3")
# to a student's roll number and name.
#
# In a real deployment the camera would cover the full hall and
# each student's face region would be matched to their seat via
# a calibration step. For this prototype, the operator selects
# which seat is currently in frame using the arrow keys — this
# simulates switching between students during the demo.
# ──────────────────────────────────────────────────────────────

import csv
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Student:
    seat:               str
    roll_number:        str
    name:               str
    cumulative_down:    float = 0.0
    cumulative_side:    float = 0.0
    tap_count:          int   = 0
    signals:            list  = field(default_factory=list)
    status:             str   = "NORMAL"


class SeatManager:
    """
    Manages the full hall seating plan for one exam session.

    Seats are identified by a grid label: row letter + column number
    e.g.  A1, A2 ... A5
          B1, B2 ... B5
          ...

    The CSV format is:
        seat, roll_number, name
        A1,   21CS001,     Aarav Sharma

    PROTOTYPE NOTE:
    In this prototype the operator manually selects which seat is
    in camera focus using arrow keys — one student at a time.
    In production, each seat region would be mapped automatically
    from a calibration grid overlaid on the camera feed, allowing
    all students to be tracked simultaneously without any manual
    switching. Real exam footage would also be used to calibrate
    the seat boundaries accurately per hall layout.
    """

    def __init__(self, csv_path: str = "seat_allocation.csv"):
        self.students:    dict[str, Student] = {}   # seat → Student
        self.seat_order:  list[str]          = []   # ordered list of seat labels
        self.active_seat: Optional[str]      = None # seat currently in camera focus
        self._load(csv_path)

    # ── Loading ───────────────────────────────────────────────

    def _load(self, csv_path: str):
        if not os.path.exists(csv_path):
            print(f"[SeatManager] '{csv_path}' not found — running without seat data.")
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seat = row["seat"].strip().upper()
                student = Student(
                    seat        = seat,
                    roll_number = row["roll_number"].strip(),
                    name        = row["name"].strip(),
                )
                self.students[seat]  = student
                self.seat_order.append(seat)

        if self.seat_order:
            self.active_seat = self.seat_order[0]
            print(f"[SeatManager] Loaded {len(self.students)} seats from '{csv_path}'")
            print(f"[SeatManager] Starting at seat {self.active_seat}\n")

    # ── Navigation (arrow keys cycle through seats) ───────────

    def next_seat(self):
        """Move camera focus to the next seat in order."""
        if not self.seat_order:
            return
        idx = self.seat_order.index(self.active_seat)
        self.active_seat = self.seat_order[(idx + 1) % len(self.seat_order)]

    def prev_seat(self):
        """Move camera focus to the previous seat in order."""
        if not self.seat_order:
            return
        idx = self.seat_order.index(self.active_seat)
        self.active_seat = self.seat_order[(idx - 1) % len(self.seat_order)]

    # ── Active student helpers ────────────────────────────────

    def active_student(self) -> Optional[Student]:
        if self.active_seat:
            return self.students.get(self.active_seat)
        return None

    def update_active(self, tracker, status: str):
        """Persist tracker state back into the student record."""
        s = self.active_student()
        if s:
            s.cumulative_down = tracker.cumulative_down_sec
            s.cumulative_side = tracker.cumulative_side_sec
            s.tap_count       = tracker.tap_count
            s.signals         = list(tracker.signals)
            s.status          = status

    def reset_active(self):
        s = self.active_student()
        if s:
            s.cumulative_down = 0.0
            s.cumulative_side = 0.0
            s.tap_count       = 0
            s.signals         = []
            s.status          = "NORMAL"

    # ── Hall summary ──────────────────────────────────────────

    def summary(self) -> dict:
        """Returns counts for the stats panel."""
        counts = {"NORMAL": 0, "SUSPICIOUS": 0, "FLAGGED": 0}
        for s in self.students.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    def flagged_students(self) -> list[Student]:
        return [s for s in self.students.values() if s.status == "FLAGGED"]
