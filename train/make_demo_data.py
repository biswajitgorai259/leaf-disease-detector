"""NAKLI demo dataset banata hai (sirf pipeline test karne ke liye).

Isse train hua model asli bimari nahi pehchan sakta. Asli training ke liye
PlantVillage use karein.

    python train/make_demo_data.py --out data/demo --per-class 40
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["Demo___healthy", "Demo___brown_spots", "Demo___yellowing", "Demo___dark_spots"]
LEAF_BGR = (40, 150, 50)
SPOT_BGR = {
    "Demo___brown_spots": (20, 70, 130),
    "Demo___yellowing": (30, 200, 220),
    "Demo___dark_spots": (28, 30, 32),
}


def make_leaf(cls, rng, size=300):
    img = np.full((size, size, 3), 255, np.uint8)
    cx, cy = size // 2 + rng.integers(-15, 15), size // 2 + rng.integers(-15, 15)
    axes = (int(rng.integers(100, 130)), int(rng.integers(65, 90)))
    angle = float(rng.integers(0, 180))
    shade = rng.integers(-15, 15)
    leaf_color = tuple(int(np.clip(c + shade, 0, 255)) for c in LEAF_BGR)
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, leaf_color, -1)
    if cls != "Demo___healthy":
        n = int(rng.integers(4, 12)) if cls != "Demo___yellowing" else int(rng.integers(2, 5))
        for _ in range(n):
            ys, xs = np.where(mask > 0)
            k = int(rng.integers(0, len(ys)))
            r = int(rng.integers(6, 14)) if cls != "Demo___yellowing" else int(rng.integers(18, 32))
            spot = np.zeros_like(mask)
            cv2.circle(spot, (int(xs[k]), int(ys[k])), r, 255, -1)
            spot = cv2.bitwise_and(spot, mask)
            img[spot > 0] = SPOT_BGR[cls]
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls in CLASSES:
        d = Path(a.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            cv2.imwrite(str(d / f"{i:03d}.png"), make_leaf(cls, rng))
    print(f"Ban gaya: {a.out} ({len(CLASSES)} classes x {a.per_class})")


if __name__ == "__main__":
    main()
