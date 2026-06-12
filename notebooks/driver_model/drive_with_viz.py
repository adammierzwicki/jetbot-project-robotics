#!/usr/bin/env python3
"""JetBot self-driver with a browser-viewable HUD.

Since the JetBot is headless (SSH, no X11), the annotated feed is served
as MJPEG over HTTP — open  http://<jetbot-ip>:8080  in a browser on your
laptop. No GUI needed on the Jetson.

Usage:
    python drive_with_hud.py                   # drive + stream on :8080
    python drive_with_hud.py --port 9090       # different port
    python drive_with_hud.py --record out.mp4  # also save to disk
    python drive_with_hud.py --dry-run         # HUD only, motors off

Ctrl+C to stop cleanly.
"""
import argparse
import collections
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from jetbot import Camera, Robot

INPUT_SIZE   = 224
DISPLAY_SIZE = 512

# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class DriverModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1   = nn.Linear(40000, 256)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model(path, device):
    model = DriverModel()
    obj   = torch.load(path, map_location=device)
    model.load_state_dict(obj.state_dict() if isinstance(obj, nn.Module) else obj)
    model = model.to(device).eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        for _ in range(5):
            model(dummy)
    return model


def make_predict(model, device):
    @torch.no_grad()
    def predict(image):
        if image.shape[0] != INPUT_SIZE or image.shape[1] != INPUT_SIZE:
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        t = torch.from_numpy(image).to(device)
        t = t.permute(2, 0, 1)       # HWC -> CHW
        t = t[[2, 1, 0]]             # BGR -> RGB
        t = t.float().div_(255.0)
        out = model(t.unsqueeze(0)).squeeze()
        return float(out[0]), float(out[1])   # forward, turn
    return predict


# --------------------------------------------------------------------------- #
#  HUD drawing
# --------------------------------------------------------------------------- #
def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def draw_hud(frame_bgr, fwd, turn, left, right, fps, size=DISPLAY_SIZE):
    vis  = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_NEAREST)
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # translucent readout panel
    panel = vis.copy()
    cv2.rectangle(panel, (0, 0), (190, 128), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, vis, 0.55, 0, vis)

    def put(text, y, color):
        cv2.putText(vis, text, (10, y), font, 0.5, color, 1, cv2.LINE_AA)

    GREEN, ORANGE, CYAN, GREY = (0,255,0), (0,165,255), (0,200,255), (180,180,180)
    put(f"fwd  : {fwd:+.3f}",  26,  GREEN)
    put(f"turn : {turn:+.3f}", 50,  ORANGE)
    put(f"L    : {left:+.3f}", 78,  CYAN)
    put(f"R    : {right:+.3f}",102, CYAN)
    put(f"{fps:4.1f} fps",     122, GREY)

    # direction compass (bottom-centre)
    cx, cy = w // 2, h - 78
    reach  = 62
    cv2.line(vis, (cx-reach, cy), (cx+reach, cy), (70,70,70), 1, cv2.LINE_AA)
    cv2.line(vis, (cx, cy-reach), (cx, cy+reach), (70,70,70), 1, cv2.LINE_AA)
    f, t = _clamp(fwd), _clamp(turn)
    if abs(f) > 1e-3:
        cv2.arrowedLine(vis, (cx, cy), (cx, int(cy - f*reach)),
                        GREEN, 3, cv2.LINE_AA, tipLength=0.35)
    if abs(t) > 1e-3:
        cv2.arrowedLine(vis, (cx, cy), (int(cx + t*reach), cy),
                        ORANGE, 3, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(vis, (cx, cy), 4, (255,255,255), -1)

    # bipolar motor bars (edges)
    def motor_bar(x, value, label):
        half = 85
        mid  = h // 2
        top, bottom = mid - half, mid + half
        cv2.rectangle(vis, (x-13, top),  (x+13, bottom), GREY, 1)
        cv2.line(vis,      (x-13, mid),  (x+13, mid),    GREY, 1)
        v  = _clamp(value)
        y2 = mid - int(v * half)
        color = CYAN if v >= 0 else (0, 0, 255)
        cv2.rectangle(vis, (x-11, min(mid,y2)), (x+11, max(mid,y2)), color, -1)
        cv2.putText(vis, label, (x-7, bottom+22), font, 0.6, (255,255,255), 1, cv2.LINE_AA)

    motor_bar(34,    left,  "L")
    motor_bar(w-34,  right, "R")
    return vis


# --------------------------------------------------------------------------- #
#  MJPEG server
# --------------------------------------------------------------------------- #
class MJPEGServer:
    """Serves the latest annotated JPEG as a multipart stream.

    Opening http://<jetbot-ip>:<port>/ in a browser shows a minimal page
    that embeds the stream. /stream is the raw multipart endpoint (e.g. for
    VLC: vlc http://<ip>:<port>/stream).
    """

    _PAGE = (
        b"<!doctype html><html><head><title>JetBot HUD</title>"
        b"<style>body{margin:0;background:#111;display:flex;"
        b"justify-content:center;align-items:center;height:100vh}"
        b"img{max-width:100%;max-height:100vh;image-rendering:pixelated}"
        b"</style></head><body><img src='/stream'></body></html>"
    )

    def __init__(self, port, running_event):
        self.port    = port
        self.running = running_event
        self._lock   = threading.Lock()
        self._jpeg   = None
        self._httpd  = None

    def update(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def _latest(self):
        with self._lock:
            return self._jpeg

    def start(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass   # silence per-request noise

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', str(len(srv._PAGE)))
                    self.end_headers()
                    self.wfile.write(srv._PAGE)
                    return
                if self.path != '/stream':
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Cache-Control', 'no-cache, private')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while srv.running.is_set():
                        jpg = srv._latest()
                        if jpg is None:
                            time.sleep(0.02); continue
                        self.wfile.write(
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: %d\r\n\r\n' % len(jpg)
                        )
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                        time.sleep(0.033)   # ~30 fps cap
                except (BrokenPipeError, ConnectionResetError):
                    pass   # browser tab was closed

        self._httpd = ThreadingHTTPServer(('0.0.0.0', self.port), Handler)
        self._httpd.daemon_threads = True
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    here = Path(__file__).resolve().parent
    ap   = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',        default=str(here / 'driver_model2.pth'))
    ap.add_argument('--speed-gain',   type=float, default=0.15)
    ap.add_argument('--turn-gain',    type=float, default=0.15)
    ap.add_argument('--turn-bias',    type=float, default=0.0)
    ap.add_argument('--alpha',        type=float, default=0.30,
                    help='EMA smoothing on turn (lower = smoother, laggier)')
    ap.add_argument('--forward-sign', type=float, default=1.0, choices=[-1.0, 1.0])
    ap.add_argument('--turn-sign',    type=float, default=1.0, choices=[-1.0, 1.0])
    ap.add_argument('--port',         type=int,   default=8080)
    ap.add_argument('--record',       default=None, metavar='PATH',
                    help='also save annotated video to this .mp4')
    ap.add_argument('--dry-run',      action='store_true',
                    help='run HUD without driving the motors')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {args.model} on {device} ...')
    model   = load_model(args.model, device)
    predict = make_predict(model, device)
    print('Model ready.')

    camera = Camera.instance(width=INPUT_SIZE, height=INPUT_SIZE)
    robot  = Robot()

    running = threading.Event()
    running.set()

    lock   = threading.Lock()
    shared = {'frame': None, 'fwd': 0.0, 'turn': 0.0, 'left': 0.0, 'right': 0.0}
    ema    = {'turn': 0.0}

    def on_frame(change):
        if not running.is_set():
            return
        frame          = change['new']
        fwd, turn_pred = predict(frame)
        speed = args.forward_sign * fwd   * args.speed_gain
        turn  = args.turn_sign    * turn_pred * args.turn_gain + args.turn_bias
        ema['turn'] = args.alpha * turn + (1.0 - args.alpha) * ema['turn']
        left  = float(np.clip(speed + ema['turn'], -1.0, 1.0))
        right = float(np.clip(speed - ema['turn'], -1.0, 1.0))
        if not args.dry_run:
            robot.left_motor.value  = left
            robot.right_motor.value = right
        with lock:
            shared['frame'] = frame.copy()
            shared['fwd'], shared['turn'] = fwd, turn_pred
            shared['left'], shared['right'] = left, right
        if args.verbose:
            print(f'  fwd={fwd:+.3f} turn={turn_pred:+.3f} '
                  f'L={left:+.3f} R={right:+.3f}', end='\r')

    def shutdown(sig, _):
        print(f'\nSignal {sig} — stopping.')
        running.clear()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server = MJPEGServer(args.port, running)
    server.start()
    ip = _lan_ip()
    print(f'\n  Open in your browser  →  http://{ip}:{args.port}/\n')

    print(f'Driving  speed_gain={args.speed_gain}  turn_gain={args.turn_gain}  '
          f'bias={args.turn_bias}   Ctrl+C to stop.')
    camera.observe(on_frame, names='value')

    writer    = None
    fps_times = collections.deque(maxlen=30)

    try:
        while running.is_set():
            with lock:
                frame        = shared['frame']
                fwd, turn    = shared['fwd'],  shared['turn']
                left, right  = shared['left'], shared['right']
            if frame is None:
                time.sleep(0.01); continue

            t = time.time()
            fps_times.append(t)
            fps = (len(fps_times)-1) / (fps_times[-1]-fps_times[0]) \
                  if len(fps_times) > 1 else 0.0

            vis = draw_hud(frame, fwd, turn, left, right, fps)

            # push to MJPEG clients
            ok, buf = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                server.update(buf.tobytes())

            # optional disk recording
            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.record, fourcc, 21.0,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)

            time.sleep(0.005)

    finally:
        camera.unobserve(on_frame, names='value')
        time.sleep(0.1)
        robot.stop()
        camera.stop()
        server.stop()
        if writer:
            writer.release()
            print(f'Saved recording → {args.record}')
        print('Stopped cleanly.')


if __name__ == '__main__':
    main()
