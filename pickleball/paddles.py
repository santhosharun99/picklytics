"""Paddle detections, computed lazily for the frames that need them and cached per run.

Only frames around hit candidates are needed (the hit model looks at +-3 frames), so the
paddle model runs on a fraction of the video and a re-analysis reuses the cache.
Cache: <run>/paddles.json  {frame: [[x1, y1, x2, y2, conf], ...]}
"""
import json
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models" / "paddle_detector.pt"


class PaddleCache:
    def __init__(self, run_dir, video: str | None = None, imgsz: int = 960, conf: float = 0.1):
        self.path = Path(run_dir) / "paddles.json"
        self.video, self.imgsz, self.conf = video, imgsz, conf
        self.data = {int(k): v for k, v in json.loads(self.path.read_text()).items()} if self.path.exists() else {}
        self._model = None

    def get(self, f: int) -> list:
        return self.data.get(f, [])

    def ensure(self, frames) -> int:
        """Detect paddles on frames not yet cached. Returns the number of new frames."""
        todo = sorted(set(frames) - set(self.data))
        if not todo or not self.video or not MODEL.exists():
            return 0
        import cv2
        import torch
        from ultralytics import YOLO
        if self._model is None:
            self._model = YOLO(str(MODEL))
        dev = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
        cap = cv2.VideoCapture(self.video)
        pos = None
        for i, f in enumerate(todo):
            if pos != f:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            pos = f + 1
            if not ok:
                self.data[f] = []
                continue
            b = self._model(img, imgsz=self.imgsz, conf=self.conf, verbose=False, device=dev)[0].boxes
            self.data[f] = [[*(round(float(v), 1) for v in xy), round(float(c), 3)]
                            for xy, c in zip(b.xyxy.tolist(), b.conf.tolist())]
            if i % 500 == 499:
                print(f"  paddles {i + 1}/{len(todo)}", flush=True)
                self.save()
        self.save()
        return len(todo)

    def save(self):
        self.path.write_text(json.dumps(self.data))
