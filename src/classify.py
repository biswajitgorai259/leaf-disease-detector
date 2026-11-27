"""Step 4: Classification + advice.

Priority:
  1) models/leaf_model.joblib   -> trained Random Forest (train/train_classifier.py se)
  2) models/leaf_cnn.onnx + models/classes.json -> CNN (optional, bahar se train karke rakhein)
  3) colour rules               -> prototype fallback
"""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "leaf_model.joblib"
ONNX_PATH = ROOT / "models" / "leaf_cnn.onnx"
CLASSES_PATH = ROOT / "models" / "classes.json"

# Advice generic hai. Dawai ya dose ke liye local agriculture officer / KVK se zaroor puchein.
INFO = {
    "healthy": {
        "title": "Healthy leaf",
        "desc": "No significant lesions were found on this leaf.",
        "actions": [
            "Keep checking plants once a week.",
            "Water at the base and avoid wetting the leaves.",
            "Keep enough space between plants for airflow.",
        ],
    },
    "early_blight": {
        "title": "Early Blight / Leaf Spot",
        "desc": "Brown lesions, usually fungal, that spread in warm and humid conditions.",
        "actions": [
            "Remove and destroy affected leaves.",
            "Water at the base and keep foliage dry.",
            "Improve spacing and airflow between plants.",
            "If it keeps spreading, ask a local agriculture officer which fungicide suits your crop.",
        ],
    },
    "late_blight": {
        "title": "Late Blight / Dark Spots",
        "desc": "Dark, water-soaked patches that can spread fast in cool, wet weather.",
        "actions": [
            "Remove affected leaves and bag them. Do not compost them.",
            "Avoid watering from above and keep leaves dry.",
            "Clean tools after touching infected plants.",
            "Contact a local agriculture officer soon, because this can spread quickly.",
        ],
    },
    "yellow": {
        "title": "Yellow Leaf (chlorosis)",
        "desc": "Yellowing often points to a nutrient shortage, too much water or a viral infection.",
        "actions": [
            "Check drainage and avoid overwatering.",
            "Consider a soil test (nitrogen, iron).",
            "Look under the leaves for aphids or whitefly.",
            "Separate the plant if yellowing spreads fast with mottled patterns.",
        ],
    },
    "generic": {
        "title": None,
        "desc": "The model matched this leaf to the class above. Confirm it before treating.",
        "actions": [
            "Remove badly affected leaves.",
            "Keep foliage dry and improve airflow.",
            "Confirm the diagnosis with a local agriculture officer before using any treatment.",
        ],
    },
}


def pretty(name: str) -> str:
    return name.replace("___", " - ").replace("_", " ").strip()


def key_for_class(name: str) -> str:
    low = name.lower()
    if "healthy" in low:
        return "healthy"
    if "early" in low and "blight" in low:
        return "early_blight"
    if "late" in low and "blight" in low:
        return "late_blight"
    if "yellow" in low:
        return "yellow"
    return "generic"


def describe(key: str, class_name: str = None) -> dict:
    info = dict(INFO[key])
    if info["title"] is None:
        info["title"] = pretty(class_name or "Unknown")
    return info


def load_model(path=MODEL_PATH):
    p = Path(path)
    return joblib.load(p) if p.exists() else None


def predict(bundle, feats, topk=3):
    """Random Forest se [(class_name, probability), ...], best pehle."""
    proba = bundle["model"].predict_proba(np.asarray(feats, dtype=np.float32).reshape(1, -1))[0]
    idx = np.argsort(proba)[::-1][:topk]
    classes = bundle["classes"]
    return [(str(classes[i]), float(proba[i])) for i in idx]


def load_onnx():
    if not (ONNX_PATH.exists() and CLASSES_PATH.exists()):
        return None
    net = cv2.dnn.readNetFromONNX(str(ONNX_PATH))
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return {"net": net, "classes": classes}


def predict_onnx(bundle, bgr, size=224, topk=3):
    """CNN (ONNX) prediction. Input 224x224 RGB, 0-1 scale."""
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (size, size)), 1 / 255.0, (size, size), (0, 0, 0), swapRB=True)
    bundle["net"].setInput(blob)
    z = bundle["net"].forward().flatten().astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    idx = np.argsort(p)[::-1][:topk]
    return [(str(bundle["classes"][i]), float(p[i])) for i in idx]


def rule_based(res, topk=3):
    """Prototype: lesion rang ke hisaab se score. Returns [(key, confidence), ...]."""
    c = res["colors"]
    lesion = c["brown"] + c["dark"]
    scores = {
        "healthy": max(0.05, 1.0 - lesion / 2.0 - c["yellow"] / 40.0),
        "early_blight": c["brown"] / 3.0 + 0.001,
        "late_blight": c["dark"] / 3.0 + 0.001,
        "yellow": c["yellow"] / 20.0 + 0.001,
    }
    total = sum(scores.values())
    ranked = sorted(((k, v / total) for k, v in scores.items()), key=lambda p: p[1], reverse=True)
    return ranked[:topk]
