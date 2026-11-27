"""Classifier train karo (Random Forest, hand-crafted features).

Dataset ka folder aisa hona chahiye: data/PlantVillage/<class_name>/<images>

    python train/train_classifier.py --data data/PlantVillage --per-class 300
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.features import FEATURE_NAMES  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect(data_dir: Path, per_class: int, size: int, seed: int):
    random.seed(seed)
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not classes:
        sys.exit(f"{data_dir} mein koi class folder nahi mila.")
    X, y = [], []
    for ci, cls in enumerate(classes):
        files = [p for p in (data_dir / cls).iterdir() if p.suffix in EXT]
        random.shuffle(files)
        files = files[:per_class]
        ok = 0
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                X.append(run_pipeline(img, size=size)["features"])
                y.append(ci)
                ok += 1
            except Exception as e:  # kharab image skip
                print("skip", p.name, e)
        print(f"[{ci + 1}/{len(classes)}] {cls}: {ok} images")
    return np.array(X), np.array(y), classes


def save_confusion(cm, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.5), max(4, n * 0.45)))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PlantVillage")
    ap.add_argument("--out", default=str(ROOT / "models" / "leaf_model.joblib"))
    ap.add_argument("--per-class", type=int, default=300, help="har class se kitni images")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    X, y, classes = collect(Path(a.data), a.per_class, a.size, a.seed)
    print("Features shape:", X.shape)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=a.seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print(f"\nTest accuracy: {accuracy_score(yte, pred) * 100:.1f}%\n")
    print(classification_report(yte, pred, labels=range(len(classes)), target_names=classes, zero_division=0))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_confusion(confusion_matrix(yte, pred, labels=range(len(classes))), classes, out.with_name("confusion_matrix.png"))
    joblib.dump({"model": model, "classes": classes, "feature_names": list(FEATURE_NAMES)}, out)
    print("Model save hua:", out)


if __name__ == "__main__":
    main()
