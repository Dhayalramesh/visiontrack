from visiontrack.evaluate import evaluate_detection, evaluate_tracking


def test_perfect_detections_score_precision_recall_1():
    gt = {0: [{"obj_id": 0, "cls": "circle", "bbox": (10, 10, 30, 30)}]}
    dets = {0: [(10, 10, 30, 30)]}
    metrics = evaluate_detection(dets, gt)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_empty_detections_score_zero_recall():
    gt = {0: [{"obj_id": 0, "cls": "circle", "bbox": (10, 10, 30, 30)}]}
    dets = {0: []}
    metrics = evaluate_detection(dets, gt)
    assert metrics["recall"] == 0.0
    assert metrics["fn"] == 1


def test_extra_detection_counts_as_false_positive():
    gt = {0: [{"obj_id": 0, "cls": "circle", "bbox": (10, 10, 30, 30)}]}
    dets = {0: [(10, 10, 30, 30), (200, 200, 220, 220)]}
    metrics = evaluate_detection(dets, gt)
    assert metrics["fp"] == 1
    assert metrics["tp"] == 1


def test_id_switch_is_detected():
    gt = {
        0: [{"obj_id": 0, "cls": "circle", "bbox": (10, 10, 30, 30)}],
        1: [{"obj_id": 0, "cls": "circle", "bbox": (12, 10, 32, 30)}],
    }
    tracks = {
        0: [(99, (10, 10, 30, 30))],
        1: [(100, (12, 10, 32, 30))],
    }
    metrics = evaluate_tracking(tracks, gt)
    assert metrics["id_switches"] == 1


def test_no_id_switch_when_id_is_stable():
    gt = {
        0: [{"obj_id": 0, "cls": "circle", "bbox": (10, 10, 30, 30)}],
        1: [{"obj_id": 0, "cls": "circle", "bbox": (12, 10, 32, 30)}],
    }
    tracks = {
        0: [(5, (10, 10, 30, 30))],
        1: [(5, (12, 10, 32, 30))],
    }
    metrics = evaluate_tracking(tracks, gt)
    assert metrics["id_switches"] == 0
