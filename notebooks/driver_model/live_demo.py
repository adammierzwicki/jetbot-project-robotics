#!/usr/bin/env python3
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

INPUT_SIZE = 224


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
    state = {'turn': 0.0}

    def on_frame(change):
        if not running.is_set():
            return

        forward_pred, turn_pred = predict(change['new'])

        speed = args.forward_sign * forward_pred * args.speed_gain
        turn = args.turn_sign * turn_pred * args.turn_gain + args.turn_bias

        state['turn'] = args.alpha * turn + (1.0 - args.alpha) * state['turn']

        # Differential mix: positive turn slows the right wheel (turns right),
        # matching the data-collection convention.
        left = float(np.clip(speed + state['turn'], -1.0, 1.0))
        right = float(np.clip(speed - state['turn'], -1.0, 1.0))

        robot.left_motor.value = left
        robot.right_motor.value = right

        if args.verbose:
            print(f'  forward={forward_pred:+.3f} turn={turn_pred:+.3f}  '
                  f'L={left:+.3f} R={right:+.3f}', end='\r')

    def shutdown(signum, _frame):
        print(f'\nReceived signal {signum}, stopping.')
        running.clear()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f'Driving. speed_gain={args.speed_gain} turn_gain={args.turn_gain} '
          f'bias={args.turn_bias}. Ctrl+C to stop.')
    camera.observe(on_frame, names='value')

    try:
        while running.is_set():
            time.sleep(0.1)
    finally:
        camera.unobserve(on_frame, names='value')
        time.sleep(0.1)
        robot.stop()
        camera.stop()
        print('Stopped cleanly.')


if __name__ == '__main__':
    main()
