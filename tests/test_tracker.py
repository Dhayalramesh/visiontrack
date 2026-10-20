from visiontrack.tracker import IOUTracker, iou, track_sequence


def test_iou_identical_boxes_is_one():
    box = (10, 10, 30, 30)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_tracker_keeps_same_id_for_smoothly_moving_object():
    tracker = IOUTracker()
    detections_by_frame = {
        t: [(10 + 2 * t, 10, 30 + 2 * t, 30)] for t in range(10)
    }
    tracks = track_sequence(detections_by_frame, tracker)
    ids_seen = {tid for frame_tracks in tracks.values() for tid, _ in frame_tracks}
    assert len(ids_seen) == 1


def test_tracker_assigns_separate_ids_to_distant_objects():
    tracker = IOUTracker()
    detections_by_frame = {0: [(0, 0, 20, 20), (200, 200, 220, 220)]}
    tracks = track_sequence(detections_by_frame, tracker)
    ids = {tid for tid, _ in tracks[0]}
    assert len(ids) == 2


def test_tracker_terminates_track_after_max_age():
    tracker = IOUTracker(max_age=2)
    detections_by_frame = {0: [(0, 0, 20, 20)], 1: [], 2: [], 3: [], 4: []}
    track_sequence(detections_by_frame, tracker)
    assert len(tracker.tracks) == 0
