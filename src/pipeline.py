"""Poora pipeline: preprocess -> leaf segmentation -> lesion pinpointing -> features -> stats."""
import cv2

from .features import extract_features
from .preprocess import preprocess
from .segment import leaf_mask, lesion_masks, overlay

MAX_PINS = 80


def severity_of(pct: float) -> str:
    return "Low" if pct < 3 else ("Medium" if pct < 15 else "High")


def run_pipeline(bgr, sensitivity=1.0, clahe=False, style="outline", size=512):
    img = preprocess(bgr, size=size, clahe=clahe)
    leaf = leaf_mask(img)
    les = lesion_masks(img, leaf, sensitivity)

    disease = les["lesion"]
    masks = {"yellow": les["chlorosis"], "brown": les["brown"], "dark": les["dark"]}
    feats = extract_features(img, leaf, disease, masks)
    over, _ = overlay(img, disease, style)
    leaf_cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(over, leaf_cnts, -1, (120, 255, 200), 1, cv2.LINE_AA)

    leaf_px = max(int((leaf > 0).sum()), 1)
    pct = 100.0 * int((disease > 0).sum()) / leaf_px
    colors = {
        "yellow": 100.0 * int((les["chlorosis"] > 0).sum()) / leaf_px,
        "brown": 100.0 * int((les["brown"] > 0).sum()) / leaf_px,
        "dark": 100.0 * int((les["dark"] > 0).sum()) / leaf_px,
    }

    spots = sorted(les["spots"], key=lambda s: s["area"], reverse=True)
    spot_count = len(spots)
    spots = spots[:MAX_PINS]
    for i, s in enumerate(spots, 1):
        s["id"] = i
        s["area_pct"] = 100.0 * s["area"] / leaf_px

    health = max(0.0, min(100.0, 100.0 - 1.6 * pct - 0.25 * colors["yellow"]))

    return {
        "image": img,
        "leaf_mask": leaf,
        "disease_mask": disease,
        "overlay": over,
        "features": feats,
        "pct": pct,
        "health": health,
        "spots": spots,
        "spot_count": spot_count,
        "severity": severity_of(pct),
        "colors": colors,  # % of leaf: yellow = chlorosis, brown, dark
    }
