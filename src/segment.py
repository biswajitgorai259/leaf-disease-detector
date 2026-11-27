"""Step 2: Segmentation.

1) Patta alag karna: HSV seed + GrabCut refinement + sabse bada component.
2) Bimari ke spots pinpoint karna:
   - Gehre dhabbe (kaale/brown): apne aas-paas ke patte se zyada gehre
   - Chhote peele dhabbe: spot maane jaate hain
   - Bada peela area: chlorosis (yellowing) maana jata hai, spot nahi
   - Patli veins aur green rang ko bimari se alag rakhte hain
"""
import cv2
import numpy as np

KERNEL3 = np.ones((3, 3), np.uint8)


def _fill_holes(m):
    if m[0, 0]:  # corner patte ke andar hai, fill karna safe nahi
        return m
    h, w = m.shape
    ff = m.copy()
    msk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 255)
    return m | cv2.bitwise_not(ff)


def leaf_mask(bgr):
    """Patta alag karo. HSV se seed, phir GrabCut se kinara saaf karo."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    seed = ((hsv[..., 1] > 35) & (hsv[..., 2] > 35)).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = seed * 255

    frac = seed.sum() / float(h * w)
    if 0.01 < frac < 0.98:
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        gc[seed > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(seed, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            pass  # GrabCut fail ho to HSV seed hi use hoga

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    return _fill_holes(m)


def lesion_masks(bgr, leaf, sensitivity=1.0):
    """Bimari ke spots dhoondhna. Returns lesion mask, alag alag masks, chlorosis aur spot list."""
    lp = leaf > 0
    inner = cv2.erode(leaf, np.ones((9, 9), np.uint8)) > 0  # patte ke kinaare ki shadow hatao
    leaf_px = max(int(lp.sum()), 1)
    min_area = max(8, int(0.0006 * leaf_px))
    sens = max(sensitivity, 0.1)

    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Local background: patte ke bahar ko patte ke median se bhar do, phir local median nikalo
    med = int(np.median(L[lp])) if lp.any() else 128
    local = cv2.medianBlur(np.where(lp, L, med).astype(np.uint8), 41).astype(np.int16)

    delta = max(8, int(18 / sens))
    not_green = (H <= 30) | (H >= 95)
    dark = (((local - L.astype(np.int16)) > delta) & not_green) | (V < 45)
    dark &= inner
    brown = (H >= 5) & (H <= 20) & (S >= int(70 / sens)) & (V >= 40) & (V <= 200) & inner & ~dark
    yellow = (H >= 18) & (H <= 35) & (S >= 60) & (V >= 110) & inner

    def clean(m):
        return cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, KERNEL3)

    dark_m, brown_m, yel_m = clean(dark), clean(brown), clean(yellow)
    # brown ki kinaari ring (dark spot ke aas-paas) hata do, warna ek hi dhabbe ke do pin banenge
    near_dark = cv2.dilate(dark_m, np.ones((15, 15), np.uint8))
    brown_m = cv2.bitwise_and(brown_m, cv2.bitwise_not(near_dark))

    dark_keep = np.zeros_like(dark_m)
    brown_keep = np.zeros_like(dark_m)
    yel_spot = np.zeros_like(dark_m)
    chl = np.zeros_like(dark_m)
    spots = []

    def thin(w, h, a):  # patli veins / lines skip
        return max(w, h) / max(1, min(w, h)) > 4.5 and a < 0.35 * w * h

    # 1) gehre dhabbe: rang dekh kar brown ya dark tay karo
    n, labels, st, cen = cv2.connectedComponentsWithStats(dark_m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area or thin(w, h, a):
            continue
        comp = labels == i
        hm, sm = float(np.median(H[comp])), float(np.median(S[comp]))
        kind = "brown" if (5 <= hm <= 20 and sm >= 60) else "dark"
        (brown_keep if kind == "brown" else dark_keep)[comp] = 255
        spots.append({"type": kind, "area": int(a), "cx": float(cen[i][0]), "cy": float(cen[i][1])})

    # 2) alag brown dhabbe
    n, labels, st, cen = cv2.connectedComponentsWithStats(brown_m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area or thin(w, h, a):
            continue
        brown_keep[labels == i] = 255
        spots.append({"type": "brown", "area": int(a), "cx": float(cen[i][0]), "cy": float(cen[i][1])})

    # 3) peele: chhota = spot, bada (leaf ka 8%+) = chlorosis (yellowing area, spot nahi)
    n, labels, st, cen = cv2.connectedComponentsWithStats(yel_m)
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        comp = labels == i
        if a < 0.08 * leaf_px:
            yel_spot[comp] = 255
            spots.append({"type": "yellow", "area": a, "cx": float(cen[i][0]), "cy": float(cen[i][1])})
        else:
            chl[comp] = 255

    lesion = cv2.bitwise_or(cv2.bitwise_or(dark_keep, brown_keep), yel_spot)
    return {"lesion": lesion, "dark": dark_keep, "brown": brown_keep, "chlorosis": chl, "spots": spots}


def overlay(bgr, disease, style="outline", min_area=20):
    """Bimari wale hisse pe laal outline (ya halka fill)."""
    out = bgr.copy()
    cnts, _ = cv2.findContours(disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    if style == "fill":
        layer = out.copy()
        cv2.drawContours(layer, cnts, -1, (0, 0, 255), -1)
        out = cv2.addWeighted(layer, 0.4, out, 0.6, 0)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out, cnts
