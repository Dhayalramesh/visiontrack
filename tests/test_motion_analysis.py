import numpy as np
from visiontrack.motion_analysis import analyze_sequence, motion_magnitude, compute_flow


def test_no_motion_between_identical_frames():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[20:40, 30:50] = (255, 255, 255)
    flow = compute_flow(frame, frame.copy())
    assert motion_magnitude(flow) < 0.01


def test_detects_motion_when_object_moves():
    f1 = np.zeros((60, 80, 3), dtype=np.uint8)
    f1[20:40, 10:30] = (255, 255, 255)
    f2 = np.zeros((60, 80, 3), dtype=np.uint8)
    f2[20:40, 30:50] = (255, 255, 255)
    flow = compute_flow(f1, f2)
    assert motion_magnitude(flow) > 0.01


def test_analyze_sequence_returns_one_record_per_transition():
    frames = [np.zeros((60, 80, 3), dtype=np.uint8) for _ in range(5)]
    records = analyze_sequence(frames)
    assert len(records) == 4
