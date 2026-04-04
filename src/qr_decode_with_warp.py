import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def safe_crop(img, x1, y1, x2, y2, pad=0):
    h, w = img.shape[:2]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def bbox_to_polygon(bbox):
    x1, y1, x2, y2 = bbox
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def find_finder_patterns(gray, polygon):
    """
    Locate 3 finder pattern centers using contour hierarchy.
    Finder patterns have nested structure: outer > white > inner (2+ nesting).
    Returns (fp_tl, fp_tr, fp_bl) where fp_tl is the right-angle corner,
    or None if detection fails.
    """
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    block = min(51, max(3, (min(gray.shape[:2]) // 4) | 1))
    binary = cv2.adaptiveThreshold(
        masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 10)
    if np.mean(binary[mask > 0]) > 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.bitwise_and(binary, binary, mask=mask)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) < 3:
        return None
    hierarchy = hierarchy[0]

    candidates = []
    for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
        child = hier[2]
        if child == -1:
            continue
        if hierarchy[child][2] == -1:
            continue

        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        r = cv2.minAreaRect(cnt)
        rw, rh = r[1]
        if rw == 0 or rh == 0:
            continue
        if max(rw, rh) / min(rw, rh) > 1.5:
            continue

        child_area = cv2.contourArea(contours[child])
        if child_area == 0 or area / child_area > 6:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        candidates.append((M["m10"]/M["m00"], M["m01"]/M["m00"], area))

    if len(candidates) < 3:
        return None

    candidates.sort(key=lambda x: x[2], reverse=True)
    centers = np.array([[c[0], c[1]] for c in candidates[:3]], dtype=np.float32)

    d01 = np.linalg.norm(centers[0] - centers[1])
    d02 = np.linalg.norm(centers[0] - centers[2])
    d12 = np.linalg.norm(centers[1] - centers[2])
    sums = [d01 + d02, d01 + d12, d02 + d12]
    tl_idx = int(np.argmin(sums))
    other = [i for i in range(3) if i != tl_idx]

    tl = centers[tl_idx]
    v1 = centers[other[0]] - tl
    v2 = centers[other[1]] - tl
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if cross > 0:
        tr, bl = centers[other[0]], centers[other[1]]
    else:
        tr, bl = centers[other[1]], centers[other[0]]

    return tl, tr, bl


def estimate_corners(fp_tl, fp_tr, fp_bl):
    """
    Extend finder pattern centers to outer QR corners (Karrach eq. 6).
    P4 = P1 + P3 - P2 (parallelogram rule).
    """
    d_tr = np.linalg.norm(fp_tr - fp_tl)
    d_bl = np.linalg.norm(fp_bl - fp_tl)
    mw = (d_tr + d_bl) / 2 / 14
    if mw < 1:
        mw = max(d_tr, d_bl) / 20
    ext = mw * np.sqrt(18)

    dir_tl_tr = (fp_tr - fp_tl) / (d_tr + 1e-8)
    dir_tl_bl = (fp_bl - fp_tl) / (d_bl + 1e-8)

    p2 = fp_tl - (dir_tl_tr + dir_tl_bl) * ext / np.sqrt(2)

    perp_tr = np.array([-dir_tl_tr[1], dir_tl_tr[0]])
    if np.dot(perp_tr, fp_tr - fp_bl) < 0:
        perp_tr = -perp_tr
    p3 = fp_tr + (dir_tl_tr + perp_tr) * ext / np.sqrt(2)

    perp_bl = np.array([dir_tl_bl[1], -dir_tl_bl[0]])
    if np.dot(perp_bl, fp_bl - fp_tr) < 0:
        perp_bl = -perp_bl
    p1 = fp_bl + (dir_tl_bl + perp_bl) * ext / np.sqrt(2)

    p4 = p1 + p3 - p2
    return p1, p2, p3, p4


def edge_proj_score(gray, pts):
    h, w = gray.shape[:2]
    pts = np.clip(np.float32(pts), [0, 0], [w - 1, h - 1])
    N = 100
    dst = np.float32([[0, 0], [N-1, 0], [N-1, N-1], [0, N-1]])
    try:
        M = cv2.getPerspectiveTransform(pts, dst)
        warp = cv2.warpPerspective(gray, M, (N, N)).astype(np.float32)
    except cv2.error:
        return -1.0
    dy = np.abs(warp[1:] - warp[:-1])
    dx = np.abs(warp[:, 1:] - warp[:, :-1])
    return float(np.std(dy.sum(1)) + np.std(dx.sum(0)))


def refine_p4(gray, p1, p2, p3, p4, max_iters=10):
    """
    Refine P4 by maximizing std dev of edge projections (Karrach Section 4.1).
    """
    best_p4 = p4.copy()
    quad = np.float32([p2, p3, p4, p1])
    best_score = edge_proj_score(gray, quad)
    step = 2.0

    dir_v = (p3 - p4) / (np.linalg.norm(p3 - p4) + 1e-8)
    dir_h = (p1 - p4) / (np.linalg.norm(p1 - p4) + 1e-8)

    for _ in range(max_iters):
        if step < 0.5:
            break
        improved = False
        for d in [dir_v, -dir_v, dir_h, -dir_h]:
            test = best_p4 + d * step
            q = np.float32([p2, p3, test, p1])
            s = edge_proj_score(gray, q)
            if s > best_score:
                best_score = s
                best_p4 = test
                improved = True
        if not improved:
            step /= 2

    return best_p4


def warp_to_square(image, p1, p2, p3, p4, size):
    src = np.float32([p2, p3, p4, p1])
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    M, _ = cv2.findHomography(src, dst)
    if M is None:
        return None
    return cv2.warpPerspective(image, M, (size, size))


def rotate_crop(gray, poly):
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
    Full pipeline:
      1) Rotate-crop (fast, works for mild tilt)
      2) Finder-pattern homography (Karrach method, handles perspective)
      3) Bbox crop fallback
    """
    poly = np.float32(poly_xy)
    side = int(max(cv2.minAreaRect(poly)[1]))
    if side < 20:
        return None
    gray = to_gray(frame)

    # 1) rotate crop
    crop = rotate_crop(gray, poly)
    if crop is not None:
        text = try_decode(crop)
        if text:
            return text

    # 2) finder-pattern based homography
    fp = find_finder_patterns(gray, poly)
    if fp is not None:
        fp_tl, fp_tr, fp_bl = fp
        p1, p2, p3, p4 = estimate_corners(fp_tl, fp_tr, fp_bl)
        p4 = refine_p4(gray, p1, p2, p3, p4)

        for sz in (side, int(side * 1.1), 300):
            warped = warp_to_square(gray, p1, p2, p3, p4, sz)
            if warped is not None:
                text = try_all_rotations(warped)
                if text:
                    return text

    # 3) bbox crop fallback
    x0, y0, x1, y1 = bbox
    pad = max(10, side // 8)
    crop = safe_crop(gray, x0, y0, x1, y1, pad=pad)
    if crop is not None:
        text = try_decode(crop)
        if text:
            return text

    return None
