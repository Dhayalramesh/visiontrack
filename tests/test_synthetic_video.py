from visiontrack.synthetic_video import generate_synthetic_sequence, FRAME_W, FRAME_H


def test_generates_requested_frame_count():
    frames, gt = generate_synthetic_sequence(n_frames=20, n_objects=2)
    assert len(frames) == 20
    assert len(gt) == 20


def test_every_frame_has_all_objects():
    frames, gt = generate_synthetic_sequence(n_frames=15, n_objects=3)
    for t, objs in gt.items():
        assert len(objs) == 3


def test_objects_stay_in_frame_bounds():
    _, gt = generate_synthetic_sequence(n_frames=60, n_objects=3)
    for t, objs in gt.items():
        for obj in objs:
            x1, y1, x2, y2 = obj["bbox"]
            assert 0 <= x1 < x2 <= FRAME_W
            assert 0 <= y1 < y2 <= FRAME_H


def test_deterministic_with_fixed_seed():
    _, gt1 = generate_synthetic_sequence(n_frames=10, n_objects=2, seed=42)
    _, gt2 = generate_synthetic_sequence(n_frames=10, n_objects=2, seed=42)
    assert gt1[5][0]["bbox"] == gt2[5][0]["bbox"]
