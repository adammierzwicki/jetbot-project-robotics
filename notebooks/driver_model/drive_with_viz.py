#!/usr/bin/env python3
"""Run the trained DriverModel on the JetBot camera and show a live HUD.

The HUD overlays the model's current outputs on top of the camera feed:
  - numeric readouts  (forward / turn predictions, left / right motor commands)
  - a proportional "compass" of arrows  (vertical = forward, horizontal = turn)
  - bipolar motor bars on the left / right edges

Press q or ESC in the window to stop. On a headless JetBot (no X / no
X-forwarding) imshow will fail; pass --no-display, and/or --record out.mp4
to write the annotated video to disk instead.
"""
import argparse
import collections
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from jetbot import Camera, Robot

INPUT_SIZE = 224
DISPLAY_SIZE = 512


class DriverModel(nn.Module):
    """Architecture identical to model_training.py.

    fc1 is a concrete Linear(40000, 256) — that's 64 channels * 25 * 25 for a
    224x224 input, the size LazyLinear materialized to during training.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(40000, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model(path, device):
    model = DriverModel()
    obj = torch.load(path, map_location=device)
    # Accept either a state_dict or a pickled module.
    model.load_state_dict(obj.state_dict() if isinstance(obj, nn.Module) else obj)
    model = model.to(device).eval()

    # Warm up (kernels/allocator) so the first real frame isn't slow.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        for _ in range(5):
            model(dummy)
    return model


def make_predict(model, device):
    """Mirror training preprocessing: convert(RGB) + ToTensor() — RGB order,
    values scaled to [0, 1], no normalization. The camera gives BGR uint8 HWC."""

    @torch.no_grad()
    def predict(image):
        if image.shape[0] != INPUT_SIZE or image.shape[1] != INPUT_SIZE:
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        tensor = torch.from_numpy(image).to(device)
        tensor = tensor.permute(2, 0, 1)      # HWC -> CHW
        tensor = tensor[[2, 1, 0]]            # BGR -> RGB
        tensor = tensor.float().div_(255.0)   # ToTensor() scaling, no normalization
        out = model(tensor.unsqueeze(0)).squeeze()
        return float(out[0]), float(out[1])   # forward, turn

    return predict


# --------------------------------------------------------------------------- #
#  HUD drawing
# --------------------------------------------------------------------------- #
def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def draw_hud(frame_bgr, fwd, turn, left, right, fps, size=DISPLAY_SIZE):
    """Return an upscaled copy of the BGR frame with the HUD drawn on top."""
    vis = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_NEAREST)
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- translucent readout panel (top-left) ----
    panel = vis.copy()
    cv2.rectangle(panel, (0, 0), (190, 128), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, vis, 0.55, 0, vis)

    def put(text, y, color):
        cv2.putText(vis, text, (10, y), font, 0.5, color, 1, cv2.LINE_AA)

    green, orange, cyan, grey = (0, 255, 0), (0, 165, 255), (0, 200, 255), (180, 180, 180)
    put(f"fwd  : {fwd:+.3f}", 26, green)
    put(f"turn : {turn:+.3f}", 50, orange)
    put(f"L    : {left:+.3f}", 78, cyan)
    put(f"R    : {right:+.3f}", 102, cyan)
    put(f"{fps:4.1f} fps", 122, grey)

    # ---- direction compass (bottom-center): proportional arrows ----
    cx, cy = w // 2, h - 78
    reach = 62
    # faint axis guides
    cv2.line(vis, (cx - reach, cy), (cx + reach, cy), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(vis, (cx, cy - reach), (cx, cy + reach), (70, 70, 70), 1, cv2.LINE_AA)

    f, t = _clamp(fwd), _clamp(turn)
    # forward arrow: up = +forward
    if abs(f) > 1e-3:
        cv2.arrowedLine(vis, (cx, cy), (cx, int(cy - f * reach)),
                        green, 3, cv2.LINE_AA, tipLength=0.35)
    # turn arrow: right = +turn
    if abs(t) > 1e-3:
        cv2.arrowedLine(vis, (cx, cy), (int(cx + t * reach), cy),
                        orange, 3, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)

    # ---- bipolar motor bars (left / right edges) ----
    def motor_bar(x, value, label):
        bar_h, half = 170, 85
        mid = h // 2
        top, bottom = mid - half, mid + half
        cv2.rectangle(vis, (x - 13, top), (x + 13, bottom), grey, 1)
        cv2.line(vis, (x - 13, mid), (x + 13, mid), grey, 1)        # zero line
        v = _clamp(value)
        y2 = mid - int(v * half)
        color = cyan if v >= 0 else (0, 0, 255)                     # red when reversing
        cv2.rectangle(vis, (x - 11, min(mid, y2)), (x + 11, max(mid, y2)), color, -1)
        cv2.putText(vis, label, (x - 7, bottom + 22), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    motor_bar(34, left, "L")
    motor_bar(w - 34, right, "R")
    return vis


# --------------------------------------------------------------------------- #
def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=str(here / 'driver_model2.pth'),
                        help='path to the trained .pth')
    parser.add_argument('--speed-gain', type=float, default=0.15,
                        help='scales predicted forward command into motor speed')
    parser.add_argument('--turn-gain', type=float, default=0.15,
                        help='scales predicted turn command into the left/right split')
    parser.add_argument('--turn-bias', type=float, default=0.0,
                        help='steady-state correction for motor/camera offset')
    parser.add_argument('--alpha', type=float, default=0.30,
                        help='EMA smoothing on turn; lower = smoother but laggier')
    parser.add_argument('--forward-sign', type=float, default=1.0, choices=[-1.0, 1.0],
                        help='polarity of the recorded forward axis (-1 if stick-up was negative)')
    parser.add_argument('--turn-sign', type=float, default=1.0, choices=[-1.0, 1.0],
                        help='polarity of the recorded turn axis')
    parser.add_argument('--no-display', action='store_true',
                        help='disable the GUI window (for headless / no X-forwarding)')
    parser.add_argument('--record', default=None, metavar='PATH',
                        help='also write the annotated video to this .mp4')
    parser.add_argument('--dry-run', action='store_true',
                        help='do not drive the motors (HUD only)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print per-frame predictions and motor commands')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {args.model} on {device}...')
    model = load_model(args.model, device)
    predict = make_predict(model, device)
    print('Model ready.')

    camera = Camera.instance(width=INPUT_SIZE, height=INPUT_SIZE)
    robot = Robot()

    running = threading.Event()
    running.set()

    # Shared state written by the camera thread, read by the main thread.
    lock = threading.Lock()
    shared = {'frame': None, 'fwd': 0.0, 'turn': 0.0, 'left': 0.0, 'right': 0.0}
    ema = {'turn': 0.0}

    def on_frame(change):
        if not running.is_set():
            return

        frame = change['new']                     # BGR uint8 HWC
        forward_pred, turn_pred = predict(frame)

        speed = args.forward_sign * forward_pred * args.speed_gain
        turn = args.turn_sign * turn_pred * args.turn_gain + args.turn_bias
        ema['turn'] = args.alpha * turn + (1.0 - args.alpha) * ema['turn']

        # Differential mix: positive turn slows the right wheel (turns right).
        left = float(np.clip(speed + ema['turn'], -1.0, 1.0))
        right = float(np.clip(speed - ema['turn'], -1.0, 1.0))

        if not args.dry_run:
            robot.left_motor.value = left
            robot.right_motor.value = right

        with lock:
            shared['frame'] = frame.copy()
            shared['fwd'], shared['turn'] = forward_pred, turn_pred
            shared['left'], shared['right'] = left, right

        if args.verbose:
            print(f'  forward={forward_pred:+.3f} turn={turn_pred:+.3f}  '
                  f'L={left:+.3f} R={right:+.3f}', end='\r')

    def shutdown(signum, _frame):
        print(f'\nReceived signal {signum}, stopping.')
        running.clear()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f'Driving. speed_gain={args.speed_gain} turn_gain={args.turn_gain} '
          f'bias={args.turn_bias}.'
          + ('' if args.no_display else ' Press q / ESC in the window to stop.'))
    camera.observe(on_frame, names='value')

    writer = None
    show = not args.no_display
    fps_times = collections.deque(maxlen=30)
    win = 'JetBot HUD'

    try:
        while running.is_set():
            with lock:
                frame = shared['frame']
                fwd, turn = shared['fwd'], shared['turn']
                left, right = shared['left'], shared['right']

            if frame is None:                     # no frame yet
                time.sleep(0.01)
                continue

            now = time.time()
            fps_times.append(now)
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) \
                if len(fps_times) > 1 else 0.0

            vis = draw_hud(frame, fwd, turn, left, right, fps)

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.record, fourcc, 21.0,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)

            if show:
                try:
                    cv2.imshow(win, vis)
                    # quit on q / ESC, or if the window is closed
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27) or \
                       cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        running.clear()
                except cv2.error as e:
                    print(f'\n[display disabled — no GUI available: {e}]')
                    show = False
                    if not args.record:
                        print('[tip: re-run with --no-display, or --record out.mp4]')
            else:
                time.sleep(0.02)                  # don't busy-spin when headless
    finally:
        camera.unobserve(on_frame, names='value')
        time.sleep(0.1)
        robot.stop()
        camera.stop()
        if writer is not None:
            writer.release()
            print(f'Saved recording to {args.record}')
        if show:
            cv2.destroyAllWindows()
        print('Stopped cleanly.')


if __name__ == '__main__':
    main()