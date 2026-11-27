"""Step 3: Feature extraction - color, shape aur texture.

Classifier ko ek fixed-length number vector chahiye. Yahan har leaf image se
wo vector banta hai.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

MIN_SPOT_AREA = 20

FEATURE_NAMES = (
    ["h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std"]
    + [f"h_hist_{i}" for i in range(8)]
    + ["yellow_ratio", "brown_ratio", "dark_ratio", "disease_ratio"]
    + ["n_spots", "mean_spot_area", "max_spot_area", "mean_circularity", "mean_solidity"]
    + ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "laplacian_var"]
)


def _spots(disease):
    cnts, _ = cv2.findContours(disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) > MIN_SPOT_AREA]


def extract_features(bgr, leaf, disease, masks) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lp = leaf > 0
    n_leaf = max(int(lp.sum()), 1)

    # --- color ---
    px = hsv[lp] if lp.any() else hsv.reshape(-1, 3)
    mean, std = px.mean(axis=0), px.std(axis=0)
    hist = cv2.calcHist([hsv], [0], leaf, [8], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    ratios = [float((masks[k] > 0).sum()) / n_leaf for k in ("yellow", "brown", "dark")]
    disease_ratio = float((disease > 0).sum()) / n_leaf

    # --- shape of spots ---
    cnts = _spots(disease)
    areas = [cv2.contourArea(c) for c in cnts]
    circ, sol = [], []
    for c in cnts:
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        circ.append(4 * np.pi * a / (p * p + 1e-9))
        sol.append(a / (cv2.contourArea(cv2.convexHull(c)) + 1e-9))
    shape = [
        float(len(cnts)),
        float(np.mean(areas)) / n_leaf if areas else 0.0,
        float(max(areas)) / n_leaf if areas else 0.0,
        float(np.mean(circ)) if circ else 0.0,
        float(np.mean(sol)) if sol else 0.0,
    ]

    # --- texture (GLCM on leaf bounding box) ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(lp)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else gray
    q = (crop // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(q, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    tex = [float(graycoprops(glcm, p).mean()) for p in ("contrast", "homogeneity", "energy", "correlation")]
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    vec = np.array([*mean, *std, *hist, *ratios, disease_ratio, *shape, *tex, lap], dtype=np.float32)
    assert vec.shape[0] == len(FEATURE_NAMES)
    return vec
