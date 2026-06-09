#!/usr/bin/env python3
"""Autonomous driving for JetBot using the trained DriverModel.

Runs driver_model.pth against the live camera feed and drives the motors with
the same arcade-drive mixing as teleoperation.py. Stop with Ctrl+C — or with
the gamepad button 0 (○) if pygame and a joystick are connected to the JetBot.
"""

import argparse
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from jetbot import Camera, Robot


class DriverModel(nn.Module):
    # Same architecture as notebooks/driver_model/model_training.py.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


INPUT_SIZE = 224

# Dataset stored forward = controller.axes[1].value (negative when stick is
# pushed up), turn = controller.axes[0].value. Teleop reads them as
# raw_y = -axes[1], raw_x = +axes[0] — we replicate that here. Flip these if
# the robot drives backwards or steers the wrong way on first run.
FORWARD_SIGN = -1.0
TURN_SIGN = 1.0


def load_model(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        model = DriverModel()
        # materialise LazyLinear so its weight tensor exists before load_state_dict
        with torch.no_grad():
            model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
        model.load_state_dict(obj)
    else:
        model = obj
    return model.to(device).eval()


def preprocess(frame, device):
    # Matches model_training.py: BGR->RGB, scale to [0,1], CHW, batch dim.
    # No mean/std normalisation — training only used transforms.ToTensor().
    if frame.shape[0] != INPUT_SIZE or frame.shape[1] != INPUT_SIZE:
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)


def arcade_mix(y, x, speed_scale, turn_scale):
    # Identical body to arcade_mix in teleoperation.py.
    y *= speed_scale
    x *= turn_scale
    if x > 0:
        left = max(-1.0, min(1.0, y))
        right = max(-1.0, min(1.0, y - x))
    else:
        left = max(-1.0, min(1.0, y + x))
        right = max(-1.0, min(1.0, y))
    return left, right


def try_open_gamepad():
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return None
        js = pygame.joystick.Joystick(0)
        js.init()
        return js
    except Exception as e:
        print(f"Gamepad unavailable ({e})")
        return None


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(here / 'driver_model.pth'),
                        help='path to the trained .pth')
    parser.add_argument('--speed-gain', type=float, default=0.15,
                        help='max forward speed scale')
    parser.add_argument('--turn-gain', type=float, default=0.15,
                        help='turn sensitivity scale')
    parser.add_argument('--no-gamepad', action='store_true',
                        help='disable pygame gamepad e-stop')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print per-frame inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.model, device)
    print(f"Loaded {args.model}")

    camera = Camera.instance(width=INPUT_SIZE, height=INPUT_SIZE)
    robot = Robot()

    running = threading.Event()
    running.set()

    joystick = None if args.no_gamepad else try_open_gamepad()
    if joystick is not None:
        print(f"Gamepad: {joystick.get_name()}  (button 0 / ○ = emergency stop)")

        def gamepad_poll():
            import pygame
            while running.is_set():
                pygame.event.pump()
                if joystick.get_button(0):
                    print("Emergency stop (button 0).")
                    running.clear()
                    robot.stop()
                    return
                time.sleep(0.02)

        threading.Thread(target=gamepad_poll, daemon=True).start()

    @torch.no_grad()
    def on_frame(change):
        if not running.is_set():
            return
        tensor = preprocess(change['new'], device)
        out = model(tensor)[0].cpu().numpy()
        forward, turn = float(out[0]), float(out[1])
        left, right = arcade_mix(
            FORWARD_SIGN * forward, TURN_SIGN * turn,
            args.speed_gain, args.turn_gain,
        )
        robot.left_motor.value = left
        robot.right_motor.value = right
        if args.verbose:
            print(f"  fwd={forward:+.3f}  turn={turn:+.3f}  "
                  f"L={left:+.3f}  R={right:+.3f}")

    def shutdown(signum, _frame):
        print(f"\nReceived signal {signum}, stopping.")
        running.clear()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Driving. speed_gain={args.speed_gain} turn_gain={args.turn_gain}. "
          f"Ctrl+C to stop.")
    camera.observe(on_frame, names='value')

    try:
        while running.is_set():
            time.sleep(0.1)
    finally:
        camera.unobserve(on_frame, names='value')
        time.sleep(0.1)
        robot.stop()
        camera.stop()
        print("Stopped cleanly.")


if __name__ == '__main__':
    main()
