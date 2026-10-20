"""
End-to-end: generate a synthetic sequence with known ground truth, run the
full detect -> track -> evaluate pipeline, and assert the tracking actually
recovers the ground truth within a real tolerance — not just "the code runs
without crashing". Mirrors LakeForge's row-count reconciliation test and
ExperimentLab's Monte Carlo false-positive-rate check.
"""
from visiontrack.synthetic_video import generate_synthetic_sequence
from visiontrack.detector import BackgroundSubtractionDetector, detect_sequence
from visiontrack.tracker import IOUTracker, track_sequence
from visiontrack.evaluate import evaluate_detection, evaluate_tracking

WARMUP = 3


def test_full_pipeline_recovers_synthetic_objects():
    frames, gt = generate_synthetic_sequence(n_frames=60, n_objects=3, seed=42)

    detections = detect_sequence(frames, BackgroundSubtractionDetector())
    eval_det = {t: b for t, b in detections.items() if t >= WARMUP}
    eval_gt = {t: g for t, g in gt.items() if t >= WARMUP}
    det_metrics = evaluate_detection(eval_det, eval_gt)

    assert det_metrics["precision"] > 0.85
    assert det_metrics["recall"] > 0.85
    assert det_metrics["f1"] > 0.85

    tracks = track_sequence(detections, IOUTracker())
    track_metrics = evaluate_tracking(tracks, gt)

    assert track_metrics["coverage"] > 0.85
    # 3 objects bouncing around a shared frame occasionally pass close to
    # each other; a handful of ID switches from those near-crossings is
    # expected for an IoU-only tracker (no appearance model) — this bounds
    # it well below "the tracker is randomly reassigning IDs every frame"
    assert track_metrics["id_switches"] <= 4


def test_pipeline_is_deterministic_given_same_seed():
    frames1, gt1 = generate_synthetic_sequence(n_frames=30, n_objects=2, seed=7)
    frames2, gt2 = generate_synthetic_sequence(n_frames=30, n_objects=2, seed=7)

    det1 = detect_sequence(frames1, BackgroundSubtractionDetector())
    det2 = detect_sequence(frames2, BackgroundSubtractionDetector())

    m1 = evaluate_detection(det1, gt1)
    m2 = evaluate_detection(det2, gt2)
    assert m1 == m2
