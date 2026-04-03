#!/usr/bin/env python3


import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol


def order_quad(pts):
    """Sort 4 points into TL / TR / BR / BL."""
    sums = pts.sum(1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.float32([pts[sums.argmin()], pts[diffs.argmin()],
                       pts[sums.argmax()], pts[diffs.argmax()]])


def polygon_corners(poly):
    """Pick the 4 tightest corners off a polygon.
    Rotates into the minAreaRect frame so we grab actual boundary
    points, not convex-hull overshoots."""
    pts = poly.astype(np.float32)
    if len(pts) < 4:
        return None

    rect = cv2.minAreaRect(pts)
    ang = rect[2] * np.pi / 180
    cos, sin = np.cos(ang), np.sin(ang)
    rot = np.column_stack([pts[:, 0]*cos + pts[:, 1]*sin,
                           -pts[:, 0]*sin + pts[:, 1]*cos])

    idx = [np.argmin(rot[:, 0] + rot[:, 1]),
           np.argmax(rot[:, 0] - rot[:, 1]),
           np.argmax(rot[:, 0] + rot[:, 1]),
           np.argmin(rot[:, 0] - rot[:, 1])]

    corners = np.float32([pts[i] for i in idx])

    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < 5:
                return order_quad(cv2.boxPoints(rect).astype(np.float32))

    return order_quad(corners)


def expand_quad(quad, margin=0.12):
    """Push each corner outward from centroid."""
    center = quad.mean(0)
    return np.float32(center + (quad - center) * (1 + margin))


def bbox_to_polygon(bbox):
    """Convert bbox (x1,y1,x2,y2) to rectangle polygon."""
    x1, y1, x2, y2 = bbox
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])



def fp_score(gray, center, radius):
    """Score how likely a region around `center` contains a finder pattern."""
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
        binary = cv2.adaptiveThreshold(
            patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_sz, thresh_c)
        best = max(best, _scanline_score(binary))
        best = max(best, _nesting_score(binary))
        if best > 0.5:
            break
    return best


def _scanline_score(binary):
    """Scan middle rows for 1:1:3:1:1 ratio."""
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
            segs = runs[j:j + 5]
            if any(segs[k][1] == segs[k + 1][1] for k in range(4)):
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
    """Check for contour nesting depth >= 2."""
    contours, hier = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def pick_p4(gray, quad):
    """Return the index of the corner without a finder pattern."""
    side = (np.linalg.norm(quad[0] - quad[1])
            + np.linalg.norm(quad[1] - quad[2])) / 2
    scores = [fp_score(gray, corner, side * 0.3) for corner in quad]
    p4 = int(np.argmin(scores))
    if sum(scores[i] > 0.3 for i in range(4) if i != p4) < 2:
        p4 = 2
    return p4



def warp_quad(img, quad, size):
    """Perspective-warp a quadrilateral to a square."""
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    try:
        M = cv2.getPerspectiveTransform(np.float32(quad), dst)
        return cv2.warpPerspective(img, M, (size, size))
    except cv2.error:
        return None


def edge_proj_score(gray, quad):
    """Std-dev of H & V edge projections. Higher = better quad boundary."""
    h, w = gray.shape[:2]
    pts = np.clip(np.float32(quad), [0, 0], [w - 1, h - 1])
    N = 100
    dst = np.float32([[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]])
    try:
        warp = cv2.warpPerspective(
            gray, cv2.getPerspectiveTransform(pts, dst), (N, N)
        ).astype(np.float32)
    except cv2.error:
        return -1.0
    dy = np.abs(warp[1:] - warp[:-1])
    dx = np.abs(warp[:, 1:] - warp[:, :-1])
    return float(np.std(dy.sum(1)) + np.std(dx.sum(0)))


def refine_p4(gray, quad, p4_idx):
    """Walk P4 to maximise edge_proj_score."""
    best = quad.copy()
    best_score = edge_proj_score(gray, best)

    vec1 = quad[(p4_idx - 1) % 4] - quad[p4_idx]
    vec2 = quad[(p4_idx + 1) % 4] - quad[p4_idx]
    dir1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    dir2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
    step = max(1.0, (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 100)

    for direction in (dir1, dir2):
        cur_dir = direction.copy()
        cur_step = step
        for _ in range(3):
            if cur_step < 0.25:
                break
            misses = 0
            for __ in range(20):
                trial = best.copy()
                trial[p4_idx] += cur_dir * cur_step
                score = edge_proj_score(gray, trial)
                if score > best_score:
                    best_score = score
                    best = trial
                    misses = 0
                else:
                    misses += 1
                    if misses >= 2:
                        best[p4_idx] -= cur_dir * cur_step * 2
                        best_score = edge_proj_score(gray, best)
                        cur_dir = -cur_dir
                        break
            cur_step /= 2

    return best



def rotate_crop(gray, poly):
    """De-rotate by the QR's tilt angle, then crop with padding."""
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



def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def safe_crop(img, x1, y1, x2, y2, pad=0):
    """Crop with bounds checking and optional padding."""
    h, w = img.shape[:2]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def try_decode(img):
    """Run pyzbar with multiple binarization strategies."""
    if img is None or img.size == 0:
        return None
    gray = to_gray(img)

    candidates = [
        img,
        cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]
    if gray.shape[0] > 30:
        block = min(51, max(3, (min(gray.shape[:2]) // 4) | 1))
        candidates.append(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, 10))

    for processed in candidates:
        hits = pyzbar_decode(processed, symbols=[ZBarSymbol.QRCODE])
        if hits:
            return hits[0].data.decode("utf-8")
    return None


def try_all_rotations(img):
    """try_decode at 0/90/180/270."""
    if img is None or img.size == 0:
        return None
    for k in range(4):
        rotated = np.rot90(img, k) if k else img
        text = try_decode(rotated)
        if text:
            return text
    return None


def decode_qr(frame, poly_xy, bbox):
    """
    Full QR decode pipeline.
    
    1. De-rotate and crop
    2. Perspective warp via polygon corners + finder-pattern P4 detection
    3. Edge-projection refinement of P4
    4. Try alternate P4 corners
    5. Raw bbox crop as last resort
    
    Args:
        frame: BGR or grayscale image
        poly_xy: (N, 2) polygon points (segmentation or bbox rectangle)
        bbox: (x1, y1, x2, y2) bounding box tuple
        
    Returns:
        Decoded text string or None
    """
    poly = np.float32(poly_xy)
    side = int(max(cv2.minAreaRect(poly)[1]))
    if side < 20:
        return None
    gray = to_gray(frame)

    # 1) de-rotate and crop
    crop = rotate_crop(gray, poly)
    if crop is not None:
        text = try_decode(crop)
        if text:
            return text

    # 2) perspective warp via polygon corners
    quad = polygon_corners(poly)
    if quad is None:
        return None

    p4 = pick_p4(gray, quad)
    refined = refine_p4(gray, quad, p4)
    ordered = np.roll(refined, -((p4 - 2) % 4), axis=0)
    big = expand_quad(ordered, 0.12)

    result = None
    for size in (side, int(side * 1.2), int(side * 0.8), 300):
        warped = warp_quad(gray, big, size)
        if warped is not None:
            result = try_all_rotations(warped)
            if result:
                break

    # 3) try each corner as P4
    if not result:
        for alt in range(4):
            if alt == p4:
                continue
            alt_quad = expand_quad(
                np.roll(quad, -((alt - 2) % 4), axis=0), 0.12)
            warped = warp_quad(gray, alt_quad, side)
            if warped is not None:
                result = try_all_rotations(warped)
                if result:
                    break

    if result:
        return result

    # 4) last resort: raw bbox crop
    x0, y0, x1, y1 = bbox
    pad = max(10, side // 8)
    crop = safe_crop(gray, x0, y0, x1, y1, pad=pad)
    if crop is not None:
        text = try_decode(crop)
        if text:
            return text

    return None
