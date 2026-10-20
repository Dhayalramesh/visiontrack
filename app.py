"""
VisionTrack dashboard: generate a synthetic ground-truth video, run
detection + tracking + motion analysis, and inspect the results frame by
frame alongside the metrics scored against the known ground truth.
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from visiontrack.synthetic_video import generate_synthetic_sequence
from visiontrack.detector import BackgroundSubtractionDetector, detect_sequence
from visiontrack.tracker import IOUTracker, track_sequence
from visiontrack.motion_analysis import analyze_sequence
from visiontrack.evaluate import evaluate_detection, evaluate_tracking

st.set_page_config(page_title="VisionTrack", layout="wide")
st.title("VisionTrack — Video Detection, Tracking & Motion Analysis")
st.caption(
    "A classical computer-vision pipeline (background-subtraction detection, "
    "IoU/Hungarian multi-object tracking, dense optical-flow motion analysis) "
    "run against a synthetic video with exact ground truth, so every stage is "
    "scored against a known answer instead of eyeballed."
)

WARMUP = 3
TRACK_COLORS = [
    (255, 99, 71), (60, 179, 113), (30, 144, 255),
    (238, 130, 238), (255, 215, 0), (255, 140, 0),
]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Run Pipeline", "Frame Viewer", "Detection & Tracking Metrics", "Motion Analysis"]
)

with tab1:
    st.subheader("Generate a synthetic video and run the full pipeline")
    col1, col2 = st.columns(2)
    n_frames = col1.slider("Frames", 20, 150, 60, 10)
    n_objects = col2.slider("Objects", 1, 6, 3)

    if st.button("Run pipeline", type="primary"):
        frames, gt = generate_synthetic_sequence(n_frames=n_frames, n_objects=n_objects)
        detections = detect_sequence(frames, BackgroundSubtractionDetector())
        tracks = track_sequence(detections, IOUTracker())
        motion_records = analyze_sequence(frames, event_threshold=0.15)

        eval_det = {t: b for t, b in detections.items() if t >= WARMUP}
        eval_gt = {t: g for t, g in gt.items() if t >= WARMUP}
        det_metrics = evaluate_detection(eval_det, eval_gt)
        track_metrics = evaluate_tracking(tracks, gt)

        st.session_state.update({
            "frames": frames, "gt": gt, "detections": detections, "tracks": tracks,
            "motion_records": motion_records, "det_metrics": det_metrics,
            "track_metrics": track_metrics,
        })

    if "frames" in st.session_state:
        m = st.session_state["det_metrics"]
        tm = st.session_state["track_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection precision", f"{m['precision']:.2f}")
        c2.metric("Detection recall", f"{m['recall']:.2f}")
        c3.metric("Tracking coverage", f"{tm['coverage']:.2f}")
        c4.metric("ID switches", tm["id_switches"])
        st.success(f"Pipeline run complete — {len(st.session_state['frames'])} frames processed. See the other tabs.")

with tab2:
    st.subheader("Step through frames with detections and tracks drawn")
    if "frames" not in st.session_state:
        st.info("Run the pipeline in the first tab first.")
    else:
        frames = st.session_state["frames"]
        tracks = st.session_state["tracks"]
        gt = st.session_state["gt"]
        t = st.slider("Frame", 0, len(frames) - 1, 0)

        show_gt = st.checkbox("Show ground-truth boxes (thin white)", value=True)
        frame = frames[t].copy()
        if show_gt:
            for obj in gt[t]:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for track_id, (x1, y1, x2, y2) in tracks.get(t, []):
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{track_id}", (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {t} — colored boxes are tracker output, white are ground truth", use_container_width=False)

with tab3:
    st.subheader("Detection & tracking accuracy, scored against ground truth")
    if "det_metrics" not in st.session_state:
        st.info("Run the pipeline in the first tab first.")
    else:
        m = st.session_state["det_metrics"]
        tm = st.session_state["track_metrics"]
        st.markdown("**Detection** (background subtraction, scored post-warmup)")
        st.table(pd.DataFrame([{
            "precision": f"{m['precision']:.3f}", "recall": f"{m['recall']:.3f}",
            "f1": f"{m['f1']:.3f}", "true positives": m["tp"],
            "false positives": m["fp"], "false negatives": m["fn"],
        }]))
        st.markdown("**Tracking** (IoU + Hungarian assignment)")
        st.table(pd.DataFrame([{
            "coverage": f"{tm['coverage']:.3f}", "id switches": tm["id_switches"],
            "matched frames": tm["matched_frames"], "ground-truth boxes": tm["total_gt"],
        }]))
        st.caption(
            "Coverage = fraction of ground-truth object-frames matched to a track at IoU ≥ 0.3. "
            "An ID switch is counted when the track ID assigned to the same ground-truth object "
            "changes between frames — the tracking-specific failure mode beyond raw detection accuracy."
        )

with tab4:
    st.subheader("Dense optical-flow motion analysis")
    if "motion_records" not in st.session_state:
        st.info("Run the pipeline in the first tab first.")
    else:
        df = pd.DataFrame(st.session_state["motion_records"])
        st.line_chart(df.set_index("frame")["mean_motion"])
        n_events = int(df["event"].sum())
        st.metric("Motion events flagged", f"{n_events} / {len(df)}")
        st.caption(
            "Mean Farneback optical-flow magnitude per frame transition, independent of the "
            "detector/tracker — the way a production system would use optical flow as a cheap "
            "secondary signal (e.g. to flag activity spikes) without running full detection."
        )
