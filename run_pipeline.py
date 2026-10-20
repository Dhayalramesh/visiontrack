"""
Full orchestration: generate a synthetic video with known ground truth, run
background-subtraction detection, IoU/Hungarian multi-object tracking, dense
optical-flow motion analysis, then score detection + tracking against the
ground truth.
"""
from visiontrack.synthetic_video import generate_synthetic_sequence
from visiontrack.detector import BackgroundSubtractionDetector, detect_sequence
from visiontrack.tracker import IOUTracker, track_sequence
from visiontrack.motion_analysis import analyze_sequence
from visiontrack.evaluate import evaluate_detection, evaluate_tracking

WARMUP = 3  # MOG2 needs a few frames to learn the background


def main():
    print("=" * 60)
    print("1. Generating synthetic video (3 moving objects, 60 frames)")
    frames, ground_truth = generate_synthetic_sequence(n_frames=60, n_objects=3)
    print(f"   {len(frames)} frames, {sum(len(v) for v in ground_truth.values())} ground-truth boxes")

    print("\n2. Running background-subtraction detector")
    detector = BackgroundSubtractionDetector()
    detections = detect_sequence(frames, detector)
    eval_detections = {t: b for t, b in detections.items() if t >= WARMUP}
    eval_gt = {t: g for t, g in ground_truth.items() if t >= WARMUP}
    det_metrics = evaluate_detection(eval_detections, eval_gt)
    print(f"   precision={det_metrics['precision']:.3f}  recall={det_metrics['recall']:.3f}  f1={det_metrics['f1']:.3f}")
    print(f"   tp={det_metrics['tp']}  fp={det_metrics['fp']}  fn={det_metrics['fn']}")

    print("\n3. Running IoU/Hungarian multi-object tracker")
    tracker = IOUTracker()
    tracks = track_sequence(detections, tracker)
    track_metrics = evaluate_tracking(tracks, ground_truth)
    print(f"   coverage={track_metrics['coverage']:.3f}  id_switches={track_metrics['id_switches']}")
    print(f"   matched_frames={track_metrics['matched_frames']} / {track_metrics['total_gt']}")

    print("\n4. Running dense optical-flow motion analysis")
    motion_records = analyze_sequence(frames, event_threshold=0.15)
    n_events = sum(1 for r in motion_records if r["event"])
    avg_motion = sum(r["mean_motion"] for r in motion_records) / len(motion_records)
    print(f"   avg_motion={avg_motion:.4f}  motion_events={n_events}/{len(motion_records)}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
