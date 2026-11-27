"""Leaf Check - Flask server (port 5005).

Run:   python app.py      ->  http://localhost:5005
Phone: same WiFi pe terminal mein dikhaya gaya "Phone" link
"""
import base64
import os
import socket

import cv2
from flask import Flask, jsonify, render_template, request

from src.classify import (
    MODEL_PATH, describe, key_for_class, load_model, load_onnx, predict, predict_onnx, pretty, rule_based,
)
from src.pipeline import run_pipeline
from src.preprocess import decode_image
from src.sample import make_sample

PORT = int(os.environ.get("PORT", 5005))
TYPE_NAMES = {"dark": "Dark lesion", "brown": "Brown lesion", "yellow": "Yellow spot"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

BUNDLE = load_model()             # trained Random Forest
ONNX = None if BUNDLE else load_onnx()  # optional CNN


def data_url(img, ext=".jpg"):
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def annotate(res):
    """Download ke liye image: har spot par numbered pin."""
    img = res["overlay"].copy()
    for s in res["spots"]:
        p = (int(round(s["cx"])), int(round(s["cy"])))
        cv2.circle(img, p, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, p, 9, (20, 30, 25), -1, cv2.LINE_AA)
        t = str(s["id"])
        scale = 0.42 if len(t) < 3 else 0.34
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(img, t, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def analyze(bgr, sensitivity=1.0, clahe=False):
    res = run_pipeline(bgr, sensitivity=sensitivity, clahe=clahe)

    if BUNDLE:
        top = predict(BUNDLE, res["features"])
        engine = "Random Forest (trained)"
        preds = [{"name": pretty(n), "conf": round(p * 100, 1)} for n, p in top]
        info = describe(key_for_class(top[0][0]), top[0][0])
    elif ONNX:
        top = predict_onnx(ONNX, res["image"])
        engine = "CNN (ONNX)"
        preds = [{"name": pretty(n), "conf": round(p * 100, 1)} for n, p in top]
        info = describe(key_for_class(top[0][0]), top[0][0])
    else:
        top = rule_based(res)
        engine = "Colour rules (prototype)"
        preds = [{"name": describe(k)["title"], "conf": round(p * 100, 1)} for k, p in top]
        info = describe(top[0][0])

    h, w = res["image"].shape[:2]
    pins = [
        {
            "id": s["id"],
            "x": round(s["cx"] / w, 4),
            "y": round(s["cy"] / h, 4),
            "area": round(s["area_pct"], 2),
            "type": s["type"],
            "label": TYPE_NAMES[s["type"]],
        }
        for s in res["spots"]
    ]

    c = res["colors"]
    total = sum(c.values())
    mix = []
    if total > 0:
        mix = sorted(
            ({"name": n, "pct": round(100 * c[k] / total, 1)}
             for k, n in (("brown", "Brown"), ("dark", "Dark"), ("yellow", "Yellow"))),
            key=lambda m: m["pct"], reverse=True,
        )
        mix = [m for m in mix if m["pct"] > 0]

    return {
        "engine": engine,
        "label": info["title"],
        "desc": info["desc"],
        "actions": info["actions"],
        "severity": res["severity"],
        "pct": round(res["pct"], 1),
        "health": round(res["health"]),
        "chlorosis": round(c["yellow"], 1),
        "spot_count": res["spot_count"],
        "pins": pins,
        "predictions": preds,
        "mix": mix,
        "images": {
            "highlighted": data_url(res["overlay"]),
            "original": data_url(res["image"]),
            "mask": data_url(res["disease_mask"], ".png"),
            "annotated": data_url(annotate(res)),
        },
    }


def read_options():
    try:
        sens = float(request.values.get("sensitivity", 1.0))
    except ValueError:
        sens = 1.0
    sens = min(1.5, max(0.5, sens))
    clahe = request.values.get("clahe", "0") in ("1", "true", "on")
    return sens, clahe


@app.get("/")
def index():
    return render_template("index.html", model_loaded=BUNDLE is not None)


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="Please choose a leaf photo first."), 400
    try:
        bgr = decode_image(f.read())
        sens, clahe = read_options()
        return jsonify(analyze(bgr, sens, clahe))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("analysis failed")
        return jsonify(error="Could not analyse this photo. Try a clearer one."), 500


@app.get("/api/sample")
def api_sample():
    sens, clahe = read_options()
    return jsonify(analyze(make_sample(), sens, clahe))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is too large. Please use a photo under 15 MB."), 413


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


if __name__ == "__main__":
    ip = lan_ip()
    print("\n  Leaf Check is running")
    print(f"  Laptop : http://localhost:{PORT}")
    if ip:
        print(f"  Phone  : http://{ip}:{PORT}   (same WiFi)")
    if BUNDLE:
        engine = "Random Forest (trained)"
    elif ONNX:
        engine = "CNN (ONNX)"
    else:
        engine = f"colour-rule prototype ({MODEL_PATH.name} not found)"
    print("  Engine :", engine)
    print("  Stop   : Ctrl+C\n")
    app.run(host="0.0.0.0", port=PORT, debug=os.environ.get("FLASK_DEBUG") == "1")
