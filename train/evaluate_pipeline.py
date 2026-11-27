"""Pipeline ko SYNTHETIC leaves par test karta hai jinka ground truth pata hai.

Ye real photos ki accuracy NAHI hai. Real accuracy ke liye aapko labelled real photos
(patta + bimari wale hisse ka mask/spot list) chahiye.

Run:  python train/evaluate_pipeline.py --n 40
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import run_pipeline  # noqa: E402

MATCH_PX = 14


def make_case(rng):
    H = W = 512
    base = int(rng.integers(130, 175))
    img = np.full((H, W, 3), base, np.float64) + rng.normal(0, 6, (H, W, 3))
    img = img.clip(0, 255).astype(np.uint8)
    cx, cy = W // 2 + int(rng.integers(-30, 30)), H // 2 + int(rng.integers(-30, 30))
    axes = (int(rng.integers(150, 190)), int(rng.integers(95, 125)))
    ang = float(rng.integers(0, 180))
    gt_leaf = np.zeros((H, W), np.uint8)
    cv2.ellipse(gt_leaf, (cx, cy), axes, ang, 0, 360, 255, -1)
    green = (int(rng.integers(35, 60)), int(rng.integers(130, 165)), int(rng.integers(40, 65)))
    img[gt_leaf > 0] = green
    cv2.line(img, (cx - 110, cy + 60), (cx + 110, cy - 60), (40, 110, 45), 3)

    inner = cv2.erode(gt_leaf, np.ones((41, 41), np.uint8))
    ys, xs = np.where(inner > 0)
    gt_les = np.zeros((H, W), np.uint8)
    centers = []
    for _ in range(int(rng.integers(3, 9))):
        for _try in range(50):
            k = int(rng.integers(0, len(xs)))
            x, y = int(xs[k]), int(ys[k])
            if all(np.hypot(x - a, y - b) > 40 for a, b in centers):
                break
        r = int(rng.integers(6, 14))
        col = (15, 15, 15) if rng.random() < 0.5 else (25, 75, 140)
        cv2.circle(img, (x, y), r, col, -1)
        cv2.circle(gt_les, (x, y), r, 255, -1)
        centers.append((x, y))
    return img, gt_leaf, gt_les, centers


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def prf(pred, gt):
    pred, gt = pred > 0, gt > 0
    tp = (pred & gt).sum()
    return float(tp / max(pred.sum(), 1)), float(tp / max(gt.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    leaf_iou, les_p, les_r, spot_rec, spot_prec = [], [], [], [], []
    for _ in range(a.n):
        img, gl, gles, centers = make_case(rng)
        r = run_pipeline(img)
        leaf_iou.append(iou(r["leaf_mask"], gl))
        p, rc = prf(r["disease_mask"], gles)
        les_p.append(p)
        les_r.append(rc)
        pred = [(s["cx"], s["cy"]) for s in r["spots"]]
        hit = sum(any(np.hypot(px - cx, py - cy) <= MATCH_PX for px, py in pred) for cx, cy in centers)
        spot_rec.append(hit / len(centers))
        ok = sum(any(np.hypot(px - cx, py - cy) <= MATCH_PX for cx, cy in centers) for px, py in pred)
        spot_prec.append(ok / max(len(pred), 1))
    m = lambda v: 100 * float(np.mean(v))
    print(f"Synthetic test: {a.n} images")
    print(f"Leaf edge/segmentation IoU : {m(leaf_iou):.1f}%  (worst {100*min(leaf_iou):.1f}%)")
    print(f"Lesion pixel precision     : {m(les_p):.1f}%")
    print(f"Lesion pixel recall        : {m(les_r):.1f}%")
    print(f"Spot pin recall            : {m(spot_rec):.1f}%  (spot ko pin mila)")
    print(f"Spot pin precision         : {m(spot_prec):.1f}%  (pin asli spot par)")


if __name__ == "__main__":
    main()
