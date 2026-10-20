"""
Scores detections/tracks against the synthetic ground truth — the same
"self-validating test" discipline used in the earlier projects (a known
malformed-record rate reconciled end-to-end in LakeForge, a known false-
positive rate recovered in ExperimentLab). Here: a known object count/
position per frame lets detection and tracking be scored for real, with
precision/recall/F1 and ID-switch counting, rather than eyeballed.
"""
from .tracker import iou


def evaluate_detection(detections_by_frame, ground_truth, iou_threshold=0.3):
    """
    Matches detections to ground-truth boxes per frame via greedy IoU
    matching. Returns {"precision", "recall", "f1", "tp", "fp", "fn"}.
    """
    tp = fp = fn = 0
    for t, gt_objs in ground_truth.items():
        gt_boxes = [o["bbox"] for o in gt_objs]
        dets = list(detections_by_frame.get(t, []))
        matched_gt = set()
        for det in dets:
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                i = iou(det, gt_box)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_tracking(tracks_by_frame, ground_truth, iou_threshold=0.3):
    """
    For each ground-truth object, follows which track_id was matched to it
    frame-over-frame and counts ID switches (the track_id assigned to a
    continuously-tracked object changing mid-sequence) — the key tracking-
    specific failure mode beyond plain per-frame detection accuracy.
    """
    gt_last_track = {}   # obj_id -> last matched track_id
    id_switches = 0
    matched_frames = 0
    total_gt = 0

    for t in sorted(ground_truth.keys()):
        gt_objs = ground_truth[t]
        tracks = tracks_by_frame.get(t, [])
        total_gt += len(gt_objs)
        used_tracks = set()
        for gt_obj in gt_objs:
            best_iou, best_track_id = 0.0, None
            for track_id, box in tracks:
                if track_id in used_tracks:
                    continue
                i = iou(gt_obj["bbox"], box)
                if i > best_iou:
                    best_iou, best_track_id = i, track_id
            if best_iou >= iou_threshold:
                used_tracks.add(best_track_id)
                matched_frames += 1
                prev = gt_last_track.get(gt_obj["obj_id"])
                if prev is not None and prev != best_track_id:
                    id_switches += 1
                gt_last_track[gt_obj["obj_id"]] = best_track_id

    coverage = matched_frames / total_gt if total_gt else 0.0
    return {"coverage": coverage, "id_switches": id_switches, "matched_frames": matched_frames, "total_gt": total_gt}
