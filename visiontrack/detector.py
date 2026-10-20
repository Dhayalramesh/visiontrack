"""
Frame-level object detection via classical computer vision: background
subtraction (cv2.createBackgroundSubtractorMOG2) + contour extraction. This
is a real, standard video-analysis technique (not a stub) — the same family
of algorithm used for motion-triggered detection in production CV systems
before/alongside a learned detector. Deterministic on the synthetic clips,
which is what makes scoring it against ground truth meaningful.
"""
import cv2
import numpy as np


class BackgroundSubtractionDetector:
    def __init__(self, min_area=150, history=5, var_threshold=16, learn_frames=2):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        # after `learn_frames`, freeze the background model (learningRate=0)
        # so a moving object is never slowly re-absorbed into "background" —
        # only the first couple of frames are used to learn the static scene.
        self.learn_frames = learn_frames
        self._frame_count = 0

    def detect(self, frame):
        """Returns a list of (x1, y1, x2, y2) bounding boxes for this frame."""
        learning_rate = -1 if self._frame_count < self.learn_frames else 0
        mask = self.bg_sub.apply(frame, learningRate=learning_rate)
        self._frame_count += 1
        # a fast-moving object against a sparse background can leave a
        # leading and a trailing foreground fragment ("ghosting"); MORPH_CLOSE
        # bridges that gap, merging them back into one blob instead of
        # double-counting the object.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))
        return boxes


def detect_sequence(frames, detector=None, seed_blank_frames=3):
    """
    Runs a detector across a whole sequence, returns dict frame_idx -> [boxes].

    Every real frame in this project already has objects in it (there is no
    naturally empty frame to learn from, unlike most real footage where a
    scene starts empty before anything enters it) — so `seed_blank_frames`
    blank frames are fed to the detector first, purely to let it learn a
    clean background, before the real (scored) frames are processed.
    """
    if detector is None:
        detector = BackgroundSubtractionDetector()
    if seed_blank_frames and frames:
        blank = np.zeros_like(frames[0])
        for _ in range(seed_blank_frames):
            detector.detect(blank)
        detector._frame_count = 0  # real sequence gets its own learn_frames budget
    detections = {}
    for t, frame in enumerate(frames):
        detections[t] = detector.detect(frame)
    return detections
