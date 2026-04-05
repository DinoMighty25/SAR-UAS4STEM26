import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
from pymavlink import mavutil


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def bbox_to_polygon(bbox):
    x1, y1, x2, y2 = bbox
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def mask_crop(gray, polygon, pad=15):
    """Crop the bounding region of the mask polygon with padding."""
    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    ih, iw = gray.shape[:2]
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, iw)
    y1 = min(y + h + pad, ih)
    crop = gray[y0:y1, x0:x1]
    return crop if crop.size else None


def rotate_crop(gray, poly):
    """Rotate-crop the QR region using the polygon's minimum area rectangle."""
    rect = cv2.minAreaRect(poly.astype(np.float32))
    center, (rw, rh), angle = rect
    if rw < rh:
        angle += 90

    h, w = gray.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a = abs(rot_mat[0, 0])
    sin_a = abs(rot_mat[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    rot_mat[0, 2] += (new_w - w) / 2
    rot_mat[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(gray, rot_mat, (new_w, new_h),
                              borderMode=cv2.BORDER_REPLICATE)
    new_ctr = rot_mat @ [center[0], center[1], 1.0]
    side = int(max(rw, rh))
    pad = max(20, side // 3)
    half = side // 2 + pad
    y0 = max(int(new_ctr[1]) - half, 0)
    y1 = min(int(new_ctr[1]) + half, new_h)
    x0 = max(int(new_ctr[0]) - half, 0)
    x1 = min(int(new_ctr[0]) + half, new_w)
    crop = rotated[y0:y1, x0:x1]
    return crop if crop.size else None


def try_decode(img):
    """Decode a QR image with multiple preprocessing strategies."""
    if img is None or img.size == 0:
        return None
    gray = to_gray(img)

    # raw
    hits = pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])
    if hits:
        return hits[0].data.decode("utf-8")

    # otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hits = pyzbar_decode(otsu, symbols=[ZBarSymbol.QRCODE])
    if hits:
        return hits[0].data.decode("utf-8")

    # blur + otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hits = pyzbar_decode(blur_otsu, symbols=[ZBarSymbol.QRCODE])
    if hits:
        return hits[0].data.decode("utf-8")

    # adaptive threshold
    if gray.shape[0] > 30:
        block = min(51, max(3, (min(gray.shape[:2]) // 4) | 1))
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, 10)
        hits = pyzbar_decode(adapt, symbols=[ZBarSymbol.QRCODE])
        if hits:
            return hits[0].data.decode("utf-8")

    # CLAHE + otsu (handles uneven drone lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, clahe_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hits = pyzbar_decode(clahe_otsu, symbols=[ZBarSymbol.QRCODE])
    if hits:
        return hits[0].data.decode("utf-8")

    # sharpen + otsu (helps with motion blur at altitude)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    _, sharp_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hits = pyzbar_decode(sharp_otsu, symbols=[ZBarSymbol.QRCODE])
    if hits:
        return hits[0].data.decode("utf-8")

    return None


def decode_qr(frame, poly_xy, bbox):
    """
    Simplified pipeline:
      1) Mask crop (tight crop from segmentation polygon)
      2) Rotate-crop fallback
      3) Bbox crop fallback
    """
    poly = np.float32(poly_xy)
    gray = to_gray(frame)

    # 1) mask crop
    crop = mask_crop(gray, poly)
    text = try_decode(crop)
    if text:
        return text

    # 2) rotate crop
    rcrop = rotate_crop(gray, poly)
    text = try_decode(rcrop)
    if text:
        return text

    # 3) bbox crop fallback
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0)
    pad = max(10, side // 8)
    h, w = gray.shape
    crop = gray[max(int(y0) - pad, 0):min(int(y1) + pad, h),
                max(int(x0) - pad, 0):min(int(x1) + pad, w)]
    if crop.size:
        text = try_decode(crop)
        if text:
            return text

    return None


def send_qr(url, master):
    msg = url.upper()
    if "BENT-METAL-BALLPEEN" in msg:
        msg = "BENT-METAL-BALLPEEN"
    elif "BENT-NAIL-CLAW" in msg:
        msg = "BENT-NAIL-CLAW"
    elif "TOWER" in msg:
        msg = "TOWER"
    elif "MOVING-TARGET" in msg:
        msg = "MOVING-TARGET"
    elif "dHrQl" in msg:
        msg = "STATIONARY-TARGET"
    elif "BALLPEEN" in msg:
        msg = "BALLPEEN-HAMMER"
    elif "CLAW" in msg:
        msg = "CLAW-HAMMER"
    else:
        msg = url

    master.mav.statustext_send(
        mavutil.mavlink.MAV_SEVERITY_WARNING,
        f"QR: {msg}".encode('utf-8')
    )
