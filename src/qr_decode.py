#!/usr/bin/env python3

from pyzbar.pyzbar import decode as decodeQR, ZBarSymbol
import cv2
import numpy as np
import time

_SHARPEN_KERNEL = np.array(
    ((-1., -1., -1.), (-1., 9., -1.), (-1., -1., -1.)), dtype=np.float32
)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def safe_crop(frame, x1, y1, x2, y2):
    img_h, img_w = frame.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(img_w, x2), min(img_h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = frame[sy1:sy2, sx1:sx2]
    pad_top = sy1 - y1
    pad_bottom = y2 - sy2
    pad_left = sx1 - x1
    pad_right = x2 - sx2
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return crop


def _decode_bytes(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _pyzbar(img):
    r = decodeQR(img, symbols=[ZBarSymbol.QRCODE])
    return _decode_bytes(r[0].data) if r else None


def _pad(img, pad=30):
    bg = (255, 255, 255) if len(img.shape) == 3 else 255
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)


def _rotate(img, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    bg = (255, 255, 255) if len(img.shape) == 3 else 255
    return cv2.warpAffine(img, M, (nw, nh), borderValue=bg)


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img


def _perspective_correct(crop):
    gray = _to_gray(crop)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < gray.shape[0] * gray.shape[1] * 0.1:
        return None
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).flatten()
    ordered = np.array([box[s.argmin()], box[d.argmin()], box[s.argmax()], box[d.argmax()]], dtype=np.float32)
    side = int(max(rect[1]))
    if side < 10:
        return None
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(crop, M, (side, side))


def _detect_angle(crop):
    gray = _to_gray(crop)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    _, _, angle = cv2.minAreaRect(largest)
    return angle


def _gamma(gray, g):
    table = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, table)


def _deglare(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.add(gray, tophat)


def _unsharp(gray, strength=1.5):
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def _quick_decode(img):
    t = _pyzbar(img)
    if t: return t

    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = _pyzbar(binary)
    if t: return t

    t = _pyzbar(255 - gray)
    if t: return t

    return None


def _enhanced_decode(img):
    t = _quick_decode(img)
    if t: return t

    gray = _to_gray(img)

    t = _pyzbar(_CLAHE.apply(gray))
    if t: return t

    sharp = cv2.filter2D(gray, -1, _SHARPEN_KERNEL)
    t = _pyzbar(cv2.GaussianBlur(sharp, (3, 3), 0))
    if t: return t

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    t = _pyzbar(adaptive)
    if t: return t

    t = _pyzbar(_deglare(gray))
    if t: return t

    t = _pyzbar(_gamma(gray, 0.5))
    if t: return t

    t = _pyzbar(_gamma(gray, 2.0))
    if t: return t

    t = _pyzbar(cv2.bilateralFilter(gray, 9, 75, 75))
    if t: return t

    return None


def try_decode_crop(crop, timeout=2.0):
    deadline = time.monotonic() + timeout

    padded = _pad(crop)

    warped = _perspective_correct(padded)
    warped_padded = _pad(warped) if warped is not None else None

    min_dim = min(crop.shape[:2])
    if min_dim < 30:
        scales = (1, 4, 6, 3, 8)
    elif min_dim < 50:
        scales = (1, 3, 2, 4)
    elif min_dim < 100:
        scales = (1, 2, 0.5, 3)
    else:
        scales = (1, 0.5, 2, 0.25, 3, 4)

    for scale in scales:
        if time.monotonic() > deadline: return None

        for base in (warped_padded, padded) if warped_padded is not None else (padded,):
            if scale == 1:
                img = base
            else:
                img = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]
                if w < 25 or h < 25 or w > 1024 or h > 1024:
                    continue

            t = _pyzbar(img)
            if t: return t

            t = _pyzbar(255 - img)
            if t: return t

            gray = _to_gray(img)

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = _pyzbar(binary)
            if t: return t

            sharp = cv2.filter2D(gray, -1, _SHARPEN_KERNEL)
            for ksize in ((5, 5), (7, 7), (3, 3)):
                t = _pyzbar(cv2.GaussianBlur(sharp if ksize == (3, 3) else gray, ksize, 0))
                if t: return t

    if time.monotonic() > deadline: return None

    angle = _detect_angle(padded)
    if abs(angle) > 5:
        t = _quick_decode(_rotate(padded, -angle))
        if t: return t

    for a in (45, 90, 135, 180):
        if time.monotonic() > deadline: return None
        t = _quick_decode(_rotate(padded, a))
        if t: return t

    if time.monotonic() > deadline: return None

    t = _enhanced_decode(padded)
    if t: return t

    if warped_padded is not None:
        t = _enhanced_decode(warped_padded)
        if t: return t

    return None
