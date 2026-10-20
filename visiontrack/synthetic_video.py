"""
Generates synthetic video sequences with KNOWN ground truth — moving colored
shapes ("objects") on a frame, each with a persistent id, class, and exact
bounding box per frame. This is the same local-verification idea used across
these projects (moto = real S3 API, DuckDB = real SQL engine): here, a real
OpenCV-rendered video with exact ground truth lets every downstream stage
(detection, tracking, motion analysis) be scored against the truth, not just
run-and-hope.
"""
import numpy as np

FRAME_W, FRAME_H = 480, 360
RNG_SEED = 42

CLASSES = ["circle", "square", "triangle"]
COLORS = {
    "circle": (60, 160, 255),   # BGR
    "square": (80, 220, 100),
    "triangle": (220, 90, 220),
}


class SyntheticObject:
    def __init__(self, obj_id, cls, x, y, vx, vy, size):
        self.obj_id = obj_id
        self.cls = cls
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size

    def step(self):
        self.x += self.vx
        self.y += self.vy
        # bounce off walls so the object stays in-frame for the whole clip,
        # then clamp so a single overshoot can't leave it 1px out of bounds
        if self.x - self.size < 0 or self.x + self.size > FRAME_W:
            self.vx *= -1
        if self.y - self.size < 0 or self.y + self.size > FRAME_H:
            self.vy *= -1
        self.x = min(max(self.x, self.size), FRAME_W - self.size)
        self.y = min(max(self.y, self.size), FRAME_H - self.size)

    def bbox(self):
        return (
            int(self.x - self.size), int(self.y - self.size),
            int(self.x + self.size), int(self.y + self.size),
        )


def _draw_object(frame, obj):
    x1, y1, x2, y2 = obj.bbox()
    color = COLORS[obj.cls]
    cx, cy = int(obj.x), int(obj.y)
    if obj.cls == "circle":
        import cv2
        cv2.circle(frame, (cx, cy), obj.size, color, -1)
    elif obj.cls == "square":
        import cv2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    else:  # triangle
        import cv2
        pts = np.array([
            [cx, y1],
            [x1, y2],
            [x2, y2],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], color)


def generate_synthetic_sequence(n_frames=60, n_objects=3, seed=RNG_SEED):
    """
    Returns (frames, ground_truth):
      frames: list of n_frames numpy arrays (H, W, 3) uint8, BGR, black background
      ground_truth: dict frame_idx -> list of {"obj_id", "cls", "bbox"}
    """
    rng = np.random.RandomState(seed)
    objects = []
    for i in range(n_objects):
        cls = CLASSES[i % len(CLASSES)]
        size = int(rng.randint(14, 22))
        x = rng.randint(size + 5, FRAME_W - size - 5)
        y = rng.randint(size + 5, FRAME_H - size - 5)
        speed = rng.uniform(3, 6)
        angle = rng.uniform(0, 2 * np.pi)
        vx, vy = speed * np.cos(angle), speed * np.sin(angle)
        objects.append(SyntheticObject(i, cls, float(x), float(y), vx, vy, size))

    frames = []
    ground_truth = {}
    for t in range(n_frames):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        gt_this_frame = []
        for obj in objects:
            if t > 0:
                obj.step()
            _draw_object(frame, obj)
            gt_this_frame.append({"obj_id": obj.obj_id, "cls": obj.cls, "bbox": obj.bbox()})
        frames.append(frame)
        ground_truth[t] = gt_this_frame

    return frames, ground_truth


def write_video(frames, path, fps=20):
    import cv2
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
