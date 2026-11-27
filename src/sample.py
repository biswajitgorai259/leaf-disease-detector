"""Nakli 'sample leaf' - UI ke "Try sample leaf" button ke liye (BGR image)."""
import cv2
import numpy as np


def make_sample(size: int = 400) -> np.ndarray:
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    cv2.ellipse(img, (c, c), (int(size * 0.375), int(size * 0.24)), 30, 0, 360, (60, 155, 60), -1)
    cv2.line(img, (int(size * 0.2), int(size * 0.72)), (int(size * 0.8), int(size * 0.28)), (45, 115, 45), 3)
    for x, y in [(170, 190), (230, 150), (210, 240), (140, 230)]:
        cv2.circle(img, (int(x * size / 400), int(y * size / 400)), int(14 * size / 400), (20, 70, 130), -1)
    cv2.circle(img, (int(260 * size / 400), int(210 * size / 400)), int(22 * size / 400), (30, 200, 220), -1)
    return img
