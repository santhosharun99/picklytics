"""Local server for the dashboard and the labelling tool.

    python3 serve.py            # http://127.0.0.1:8001/dashboard/  and  /dashboard/label.html

Unlike `python -m http.server` it supports HTTP Range requests (needed to seek in
videos) and lets the labelling page save labels into labels/<name>.json.
"""
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "labels"
VIDEOS = ROOT / "data" / "videos"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):  # quieter: skip successful range reads
        if not (len(args) > 1 and str(args[1]) in ("200", "206", "304")):
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path == "/api/videos":
            vids = sorted(p.name for p in VIDEOS.glob("*.mp4")) if VIDEOS.exists() else []
            return self._json(vids)
        if self.path.startswith("/api/labels/"):
            f = self._label_path()
            return self._json(json.loads(f.read_text()) if f and f.exists() else None)
        if "Range" in self.headers:
            return self._range()
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/labels/"):
            return self.send_error(404)
        f = self._label_path()
        if f is None:
            return self.send_error(400, "bad name")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        data = json.loads(body)
        LABELS.mkdir(exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(f)
        self._json({"ok": True, "saved": str(f.relative_to(ROOT))})

    def _label_path(self):
        name = self.path.rsplit("/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
            return None
        return LABELS / name

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _range(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers["Range"])
        if not m:
            return super().do_GET()
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            return self.end_headers()
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Dashboard:  http://127.0.0.1:{port}/dashboard/\nLabelling:  http://127.0.0.1:{port}/dashboard/label.html")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
