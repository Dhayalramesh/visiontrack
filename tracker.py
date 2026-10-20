"""
Multi-object tracker: assigns persistent track IDs to per-frame detections
by matching boxes frame-to-frame with IoU (Intersection-over-Union) via the
Hungarian algorithm (scipy.optimize.linear_sum_assignment) — the standard
real algorithm for this (same family as SORT), not a placeholder nearest-
neighbor heuristic.
"""
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import numpy as np


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple
    age: int = 0          # frames since last matched
    hits: int = 0         # total frames matched
    history: list = field(default_factory=list)


class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self._next_id = 0
        self.id_switches = 0

    def update(self, detections, frame_idx):
        """
        detections: list of (x1, y1, x2, y2) for the current frame.
        Returns list of (track_id, box) for tracks matched/created this frame.
        """
        if not self.tracks:
            matches, unmatched_dets, unmatched_tracks = [], list(range(len(detections))), []
        else:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = 1 - iou(tr.box, det)
            row_ind, col_ind = linear_sum_assignment(cost)
            matches, unmatched_tracks, unmatched_dets = [], list(range(len(self.tracks))), list(range(len(detections)))
            for r, c in zip(row_ind, col_ind):
                if 1 - cost[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    unmatched_tracks.remove(r)
                    unmatched_dets.remove(c)

        results = []
        matched_track_indices = set()
        for r, c in matches:
            tr = self.tracks[r]
            tr.box = detections[c]
            tr.age = 0
            tr.hits += 1
            tr.history.append((frame_idx, detections[c]))
            results.append((tr.track_id, tr.box))
            matched_track_indices.add(r)

        for r in unmatched_tracks:
            self.tracks[r].age += 1

        for c in unmatched_dets:
            new_track = Track(track_id=self._next_id, box=detections[c], hits=1)
            new_track.history.append((frame_idx, detections[c]))
            self.tracks.append(new_track)
            results.append((new_track.track_id, new_track.box))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return results


def track_sequence(detections_by_frame, tracker=None):
    """
    detections_by_frame: dict frame_idx -> [boxes]
    Returns dict frame_idx -> [(track_id, box)]
    """
    if tracker is None:
        tracker = IOUTracker()
    tracks_by_frame = {}
    for t in sorted(detections_by_frame.keys()):
        tracks_by_frame[t] = tracker.update(detections_by_frame[t], t)
    return tracks_by_frame
