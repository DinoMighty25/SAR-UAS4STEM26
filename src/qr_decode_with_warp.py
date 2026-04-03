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


# ── Karrach warp (replaces _perspective_correct) ──────────────────────────────

def _order_quad(pts):
    sums = pts.sum(1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.float32([pts[sums.argmin()], pts[diffs.argmin()],
                       pts[sums.argmax()], pts[diffs.argmax()]])


def _polygon_corners(poly):
    pts = poly.astype(np.float32)
    if len(pts) < 4:
        return None
    rect = cv2.minAreaRect(pts)
    ang = rect[2] * np.pi / 180
    cos, sin = np.cos(ang), np.sin(ang)
    rot = np.column_stack([pts[:, 0] * cos + pts[:, 1] * sin,
                           -pts[:, 0] * sin + pts[:, 1] * cos])
    idx = [np.argmin(rot[:, 0] + rot[:, 1]),
           np.argmax(rot[:, 0] - rot[:, 1]),
           np.argmax(rot[:, 0] + rot[:, 1]),
           np.argmin(rot[:, 0] - rot[:, 1])]
    corners = np.float32([pts[i] for i in idx])
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < 5:
                return _order_quad(cv2.boxPoints(rect).astype(np.float32))
    return _order_quad(corners)


def _expand_quad(quad, margin=0.12):
    center = quad.mean(0)
    return np.float32(center + (quad - center) * (1 + margin))


def _warp_quad(img, quad, size):
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    try:
        M = cv2.getPerspectiveTransform(np.float32(quad), dst)
        return cv2.warpPerspective(img, M, (size, size))
    except cv2.error:
        return None


def _scanline_score(binary):
    ph, pw = binary.shape
    for y in range(ph // 3, 2 * ph // 3, 2):
        row = binary[y]
        runs = []
        val, length = row[0], 1
        for x in range(1, pw):
            if row[x] == val:
                length += 1
            else:
                runs.append((length, val))
                val, length = row[x], 1
        runs.append((length, val))
        for j in range(len(runs) - 4):
            segs = runs[j:j+5]
            if any(segs[k][1] == segs[k+1][1] for k in range(4)):
                continue
            b1, w2, b3, w4, b5 = [s[0] for s in segs]
            total = b1 + w2 + b3 + w4 + b5
            if total < 7:
                continue
            u = total / 7.0
            if not (0.5*u < b1 < 2*u and 0.5*u < w2 < 2*u
                    and 1.5*u < b3 < 5*u
                    and 0.5*u < w4 < 2*u and 0.5*u < b5 < 2*u):
                continue
            if b3 <= max(b1 + w2, w4 + b5):
                continue
            return 1.0
    return 0.0


def _nesting_score(binary):
    contours, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0.0
    hier = hier[0]
    best = 0.0
    for i in range(len(contours)):
        depth, child = 0, hier[i][2]
        while child != -1:
            depth += 1
            child = hier[child][2]
        if depth < 2 or cv2.contourArea(contours[i]) < 30:
            continue
        rw, rh = cv2.minAreaRect(contours[i])[1]
        if rw > 0 and rh > 0 and max(rw, rh) / min(rw, rh) < 2.0:
            best = max(best, min(depth, 3) * 0.4)
    return best


def _fp_score(gray, center, radius):
    h, w = gray.shape
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    x0, y0 = max(cx - r, 0), max(cy - r, 0)
    x1, y1 = min(cx + r, w), min(cy + r, h)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    best = 0.0
    for block_sz, thresh_c in [(31, 10), (51, 5), (21, 15)]:
        block_sz = min(block_sz, max(3, min(patch.shape) // 2 * 2 - 1))
        if block_sz < 3:
            continue
        binary = cv2.adaptiveThreshold(
            patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_sz, thresh_c)
        best = max(best, _scanline_score(binary), _nesting_score(binary))
        if best > 0.5:
            break
    return best


def _pick_p4(gray, quad):
    side = (np.linalg.norm(quad[0] - quad[1])
            + np.linalg.norm(quad[1] - quad[2])) / 2
    scores = [_fp_score(gray, corner, side * 0.3) for corner in quad]

    p4 = int(np.argmin(scores))

    # FIX: require at least 2 other corners to look like finder patterns.
    # If not, try each corner as P4 (caller will brute-force anyway),
    # but pick the one whose remaining 3 corners score highest overall.
    num_good = sum(scores[i] > 0.3 for i in range(4) if i != p4)
    if num_good < 2:
        best_combo, best_p4 = -1, 2
        for candidate in range(4):
            others = sum(scores[i] for i in range(4) if i != candidate)
            if others > best_combo:
                best_combo = others
                best_p4 = candidate
        p4 = best_p4

    return p4


def _edge_proj_score(gray, quad):
    h, w = gray.shape[:2]
    pts = np.clip(np.float32(quad), [0, 0], [w-1, h-1])
    N = 100
    dst = np.float32([[0, 0], [N-1, 0], [N-1, N-1], [0, N-1]])
    try:
        warp = cv2.warpPerspective(
            gray, cv2.getPerspectiveTransform(pts, dst), (N, N)
        ).astype(np.float32)
    except cv2.error:
        return -1.0
    dy = np.abs(warp[1:] - warp[:-1])
    dx = np.abs(warp[:, 1:] - warp[:, :-1])
    return float(np.std(dy.sum(1)) + np.std(dx.sum(0)))


def _refine_p4(gray, quad, p4_idx):
    best = quad.copy()
    best_score = _edge_proj_score(gray, best)
    vec1 = quad[(p4_idx - 1) % 4] - quad[p4_idx]
    vec2 = quad[(p4_idx + 1) % 4] - quad[p4_idx]
    dir1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    dir2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
    step = max(1.0, (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 100)

    for direction in (dir1, dir2):
        cur_step = step
        for _ in range(3):
            if cur_step < 0.25:
                break
            misses = 0
            for __ in range(20):
                trial = best.copy()
                trial[p4_idx] += direction * cur_step
                score = _edge_proj_score(gray, trial)
                if score > best_score:
                    best_score = score
                    best = trial
                    misses = 0
                else:
                    misses += 1
                    if misses >= 3:
                        break  # FIX: just stop this direction/step combo, don't overshoot
            cur_step /= 2

    return best


# ── decode helpers (with binarization on warped images) ───────────────────────

def _try_rotations(img):
    """Try pyzbar at 4 rotations WITH binarization — critical for warped images."""
    gray = _to_gray(img)
    for k in range(4):
        rotated = np.rot90(gray, k) if k else gray

        # raw grayscale
        t = _pyzbar(rotated)
        if t:
            return t

        # Otsu binarization — often needed after perspective warp
        _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = _pyzbar(binary)
        if t:
            return t

        # adaptive threshold — handles uneven lighting from warp
        adaptive = cv2.adaptiveThreshold(
            rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
        t = _pyzbar(adaptive)
        if t:
            return t

        # inverted — catches light-on-dark QR codes
        t = _pyzbar(255 - rotated)
        if t:
            return t

    return None


def _try_warp_with_enhancements(gray, quad, size):
    """Warp a quad then try multiple decode strategies on the result."""
    warped = _warp_quad(gray, quad, size)
    if warped is None:
        return None

    # padded version often helps pyzbar find the quiet zone
    padded = _pad(warped, 20)

    t = _try_rotations(padded)
    if t:
        return t

    # CLAHE on warped — helps with low-contrast warps
    clahe = _CLAHE.apply(warped if len(warped.shape) == 2 else _to_gray(warped))
    t = _try_rotations(_pad(clahe, 20))
    if t:
        return t

    # sharpen on warped
    gray_w = warped if len(warped.shape) == 2 else _to_gray(warped)
    sharp = cv2.filter2D(gray_w, -1, _SHARPEN_KERNEL)
    t = _pyzbar(_pad(sharp, 20))
    if t:
        return t

    return None


def _karrach_warp(gray, poly):
    """Polygon corners → pick P4 → refine P4 → expand → warp. Returns decoded text or None."""
    quad = _polygon_corners(poly)
    if quad is None:
        return None
    side = int(max(cv2.minAreaRect(poly)[1]))
    if side < 20:
        return None
    p4 = _pick_p4(gray, quad)
    refined = _refine_p4(gray, quad, p4)
    ordered = np.roll(refined, -((p4 - 2) % 4), axis=0)

    # FIX: try multiple expansion margins — detector bbox tightness varies
    for margin in (0.12, 0.20, 0.05, 0.30):
        big = _expand_quad(ordered, margin)
        for size in (side, int(side * 1.2), int(side * 0.8), 300):
            t = _try_warp_with_enhancements(gray, big, size)
            if t:
                return t

    # plan B: brute-force each corner as P4
    for alt in range(4):
        if alt == p4:
            continue
        alt_ordered = np.roll(quad, -((alt - 2) % 4), axis=0)
        alt_quad = _expand_quad(alt_ordered, 0.12)
        for size in (side, int(side * 1.2), 300):
            t = _try_warp_with_enhancements(gray, alt_quad, size)
            if t:
                return t

    return None


def _hull_poly(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < gray.shape[0] * gray.shape[1] * 0.05:
        return None
    return cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)


# ── decode helpers ────────────────────────────────────────────────────────────

def _quick_decode(img):
    t = _pyzbar(img)
    if t: return t

    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = _pyzbar(binary)
    if t: return t

    t = _pyzbar(255 - gray)
    if t: return t

    # FIX: also try adaptive threshold in quick decode
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    t = _pyzbar(adaptive)
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


# ── public entry point ────────────────────────────────────────────────────────

def try_decode_crop(crop, timeout=2.0):
    deadline = time.monotonic() + timeout

    # FIX: try a direct quick decode BEFORE padding/warping — if the crop
    # is already well-framed, this saves all the warp overhead
    t = _quick_decode(crop)
    if t: return t

    padded = _pad(crop)
    gray_padded = _to_gray(padded)

    # attempt 1: Karrach warp from hull
    hull = _hull_poly(gray_padded)
    if hull is not None:
        t = _karrach_warp(gray_padded, hull)
        if t: return t

    if time.monotonic() > deadline: return None

    # attempt 2: scale loop with binarization
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

        if scale == 1:
            img = padded
        else:
            img = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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

    # attempt 3: de-rotate by detected angle
    angle = _detect_angle(padded)
    if abs(angle) > 5:
        t = _quick_decode(_rotate(padded, -angle))
        if t: return t

    for a in (45, 90, 135, 180):
        if time.monotonic() > deadline: return None
        t = _quick_decode(_rotate(padded, a))
        if t: return t

    if time.monotonic() > deadline: return None

    # attempt 4: enhanced (CLAHE, sharpen, gamma, deglare, bilateral)
    t = _enhanced_decode(padded)
    if t: return t

    return None
