"""Step 1: Preprocessing - decode, resize, noise removal, contrast enhancement."""
import io

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Uploaded file ke bytes ko BGR image mein badlo.

    Phone photos ka EXIF rotation (Pillow se) theek kiya jata hai, taaki photo ulti na aaye.
    """
    try:
        from PIL import Image, ImageOps

        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read this file. Please use a JPG or PNG image.")
    return img


def resize_max(bgr: np.ndarray, size: int = 400) -> np.ndarray:
    """Lambi side ko `size` pixel pe laao, aspect ratio same rakho."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)


def denoise(bgr: np.ndarray) -> np.ndarray:
    """Halka Gaussian blur: chhota noise hat jata hai."""
    return cv2.GaussianBlur(bgr, (5, 5), 0)


def enhance_contrast(bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE sirf brightness (L channel) pe, rang bigadte nahi."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(bgr: np.ndarray, size: int = 400, clahe: bool = False) -> np.ndarray:
    img = denoise(resize_max(bgr, size))
    if clahe:
        img = enhance_contrast(img)
    return img
