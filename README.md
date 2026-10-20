# VisionTrack

**Live demo:** https://visiontrack-chws7ei9cccmnlipgtnyku.streamlit.app

A video-analysis pipeline — background-subtraction object detection,
IoU/Hungarian multi-object tracking, and dense optical-flow motion
analysis — run against a synthetic video with exact, known ground truth,
so every stage is scored for real instead of eyeballed. Streamlit
dashboard included.

Built to close a specific gap: a hands-on video-analysis project (object
detection + multi-object tracking across frames, not just a single-image
classifier) to back up an M.Tech specialization in Computer Vision for
roles that explicitly screen for video-analysis algorithms.

## Why synthetic video with ground truth

Real video has no ground truth to check answers against — you can watch
the output and guess whether it looks right. This project generates
synthetic clips (moving colored shapes with known class, position, and
bounding box on every frame) so the whole pipeline can be scored exactly:
precision/recall/F1 for detection, coverage and ID-switch count for
tracking. The detector and tracker themselves are real, standard
algorithms (OpenCV `BackgroundSubtractorMOG2`, dense Farneback optical
flow, IoU cost + the Hungarian algorithm via `scipy.optimize.
linear_sum_assignment` — the same family of matching SORT uses) — only
the input video is synthetic, so the ground truth is known exactly.

## Architecture
synthetic_video.py generates frames + exact ground truth
|
v
detector.py (MOG2 background subtraction + morphological cleanup)
|
v per-frame bounding boxes
v
tracker.py (IoU cost matrix + Hungarian assignment, track birth/death)
|
v per-frame (track_id, box) pairs
v
evaluate.py — scores detection (precision/recall/F1) and tracking
(coverage, ID switches) against ground truth

motion_analysis.py — dense Farneback optical flow between frames,
independent signal for motion-event flagging

app.py — Streamlit dashboard: run the pipeline, step through frames with
detections/tracks drawn over ground truth, inspect metrics,
and view the motion-magnitude signal

## What it does

- **Synthetic video generation with exact ground truth**: moving shapes
  (circle/square/triangle) that bounce within the frame, each with a
  persistent object ID, class, and exact bounding box recorded every
  frame — deterministic given a fixed seed.
- **Background-subtraction detection**: OpenCV `BackgroundSubtractorMOG2`
  with the background model frozen (`learningRate=0`) after a short
  seed period, so a moving object is never slowly re-absorbed as
  "background" — plus morphological closing to bridge the leading/
  trailing foreground fragments ("ghosting") a fast-moving object can
  otherwise split into.
- **IoU/Hungarian multi-object tracking**: a cost matrix of
  `1 - IoU(track, detection)` solved by the Hungarian algorithm each
  frame, with track birth on unmatched detections and death after a
  configurable number of unmatched frames.
- **Dense optical-flow motion analysis**: Farneback optical flow between
  consecutive frames, aggregated into a per-frame motion magnitude and
  a coarse spatial grid — a cheap, detector-independent activity signal.
- **Ground-truth scoring, not eyeballing**: detection precision/recall/F1
  via greedy IoU matching against known boxes; tracking coverage and
  ID-switch counting by following which track ID was matched to each
  ground-truth object frame over frame.
- **Streamlit dashboard**: run the pipeline, step through frames with
  tracker output and ground truth drawn side by side, inspect the
  accuracy tables, and view the motion-magnitude trace.

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=. python run_pipeline.py     # full pipeline, printed to the terminal
streamlit run app.py                    # interactive dashboard
PYTHONPATH=. pytest tests/ -v           # 21 tests, including an end-to-end accuracy check
```

## Design notes

- **Debugged against real failure modes, not just "does it run".**
  Three real bugs surfaced and were fixed during development: (1) a
  moving object left two separate foreground fragments per frame
  ("ghosting") — fixed with morphological closing sized to bridge the
  gap; (2) slow-moving objects were silently absorbed into the adaptive
  background model within a few frames — fixed by freezing the
  background's learning rate after a short seed period; (3) every real
  frame already contains objects, so there was no naturally empty frame
  to learn a clean background from — fixed by seeding the detector with
  a few blank frames before the real sequence.
- **Ground-truth reconciliation, not just "it ran without crashing".**
  `test_full_pipeline_recovers_synthetic_objects` asserts detection
  precision/recall/F1 and tracking coverage all exceed 0.85 against the
  known ground truth, and bounds ID switches — mirroring the same
  "score against a known answer" discipline as the row-count
  reconciliation and false-positive-rate checks in earlier projects.
- **Determinism is tested.** The same seed produces the same ground
  truth and the same detection/tracking metrics, verified directly.
- **The tracking-specific failure mode is measured separately from
  detection accuracy.** ID switches (the same real object getting
  reassigned a different track ID) are counted independently of
  per-frame detection precision/recall, because a tracker can detect
  objects well per-frame while still failing at tracking continuity.
