"""
Dense optical flow (Farneback) between consecutive frames — a real motion-
estimation algorithm, used here to flag motion events (e.g. "activity spike
in region X") independent of the detector/tracker, the way a production
video-analytics system would use optical flow as a second, cheaper signal.
"""
import cv2
import numpy as np


def compute_flow(frame_prev, frame_curr):
    prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow


def motion_magnitude(flow):
    """Mean optical-flow magnitude across the whole frame."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def grid_motion(flow, grid=(4, 4)):
    """Mean motion magnitude per grid cell — a coarse spatial activity map."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h, w = mag.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw
    cells = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            cell = mag[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cells[i, j] = cell.mean() if cell.size else 0.0
    return cells


def analyze_sequence(frames, event_threshold=0.5):
    """
    Returns a list of per-transition records:
      {"frame": t, "mean_motion": float, "event": bool}
    event=True when mean_motion exceeds event_threshold (a motion spike).
    """
    records = []
    for t in range(1, len(frames)):
        flow = compute_flow(frames[t - 1], frames[t])
        mag = motion_magnitude(flow)
        records.append({"frame": t, "mean_motion": mag, "event": mag > event_threshold})
    return records
