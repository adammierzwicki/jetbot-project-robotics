#!/usr/bin/env python3
import argparse
import signal
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from jetbot import Camera, Robot

INPUT_SIZE = 224
GRAY_MEAN  = 0.449
GRAY_STD   = 0.226


class DriverModelGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3   = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1     = nn.Linear(40000, 256)
        self.fc2     = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model(path, device):
    model = DriverModelGray()
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj.state_dict() if isinstance(obj, nn.Module) else obj)
    model = model.to(device).eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, device=device)
        for _ in range(5):
            model(dummy)
    return model


def make_predict(model, device):
    @torch.no_grad()
    def predict(image):
        if image.shape[0] != INPUT_SIZE or image.shape[1] != INPUT_SIZE:
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # (H, W) uint8
        t = torch.from_numpy(gray).float().div(255.0)       # (H, W)
        t = (t - GRAY_MEAN) / GRAY_STD
        t = t.unsqueeze(0).unsqueeze(0).to(device)          # (1, 1, H, W)
        out = model(t).squeeze()
        return float(out[0]), float(out[1])
    return predict


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=str(here / 'driver_model_gray.pth'))
    parser.add_argument('--speed-gain', type=float, default=0.20)
    parser.add_argument('--turn-gain', type=float, default=0.18)
    parser.add_argument('--turn-bias', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.30)
    parser.add_argument('--forward-sign', type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument('--turn-sign', type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {args.model} on {device}...')
    model = load_model(args.model, device)
    predict = make_predict(model, device)
    print('Model ready (grayscale input).')

    camera = Camera.instance(width=INPUT_SIZE, height=INPUT_SIZE)
    robot = Robot()

    running = True
    smooth_turn = 0.0

    def shutdown(signum, _frame):
        nonlocal running
        print(f'\nReceived signal {signum}, stopping.')
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f'Driving. speed_gain={args.speed_gain} turn_gain={args.turn_gain} '
          f'bias={args.turn_bias}. Ctrl+C to stop.')

    try:
        while running:
            frame = camera.value

            if frame is None:
                time.sleep(0.01)
                continue

            forward_pred, turn_pred = predict(frame)
            forward_pred = -forward_pred

            speed = args.forward_sign * forward_pred * args.speed_gain
            turn = args.turn_sign * turn_pred * args.turn_gain + args.turn_bias
            smooth_turn = args.alpha * turn + (1.0 - args.alpha) * smooth_turn

            left  = float(np.clip(speed + smooth_turn, -1.0, 1.0))
            right = float(np.clip(speed - smooth_turn, -1.0, 1.0))

            robot.left_motor.value = left
            robot.right_motor.value = right

            if args.verbose:
                print(f'  forward={forward_pred:+.3f} turn={turn_pred:+.3f}  '
                      f'L={left:+.3f} R={right:+.3f}', end='\r')

    finally:
        robot.stop()
        camera.stop()
        print('Stopped cleanly.')


if __name__ == '__main__':
    main()
