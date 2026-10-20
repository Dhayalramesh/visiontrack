from visiontrack.synthetic_video import generate_synthetic_sequence
from visiontrack.detector import BackgroundSubtractionDetector, detect_sequence
from visiontrack.evaluate import evaluate_detection

WARMUP = 3


def test_detector_finds_moving_objects_after_warmup():
    frames, gt = generate_synthetic_sequence(n_frames=30, n_objects=3)
    detections = detect_sequence(frames, BackgroundSubtractionDetector())
    eval_det = {t: b for t, b in detections.items() if t >= WARMUP}
    eval_gt = {t: g for t, g in gt.items() if t >= WARMUP}
    metrics = evaluate_detection(eval_det, eval_gt)
    assert metrics["recall"] > 0.8


def test_detector_returns_no_boxes_on_empty_frame():
    import numpy as np
    detector = BackgroundSubtractionDetector()
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(3):
        detector.detect(blank)
    boxes = detector.detect(blank)
    assert boxes == []
