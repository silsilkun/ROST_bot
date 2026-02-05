import cv2
import numpy as np


# =========================
# Pipeline params (이 파일 상단)
# =========================
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 2.00
VALID_RATIO_MIN = 0.30

TOP_PERCENT_PRIMARY = 3.0
TOP_PERCENT_FALLBACK = 5.0
BAND_MAD_K = 3.0
BAND_MIN_M = 0.008
BAND_MAX_M = 0.030

MORPH_KERNEL = 3
CLOSE_ITERS = 2
OPEN_ITERS = 1

MIN_AREA = 800
D_MIN = 6.0
DT_KEEP_RATIO = 0.85
DT_CENTER_WEIGHT = 0.30

PCA_MIN_POINTS = 100
PCA_CORE_RATIO = 0.60
PCA_MIN_CORE_POINTS = 120
PCA_ANISO_RATIO_MIN = 1.20

YAW_ARROW_LEN_PX = 60
YAW_OFFSET_DEG = 90.0
YAW_SIGN = -1.0

Z_APPROACH_CM = 3.0
EDGE_MARGIN_PX = 20
Z_APPROACH_MIN_CM = 8.1

REJECT_IF_NO_YAW = True


def _wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def _wrap_deg_180_for_gripper(a: float) -> float:
    x = float((a + 180.0) % 360.0 - 180.0)
    if x > 90.0:
        x -= 180.0
    elif x < -90.0:
        x += 180.0
    return float(x)


def _robust_mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _parse_roi_xyxy(roi_xyxy, img_w, img_h):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI invalid")
    return x1, y1, (x2 - x1), (y2 - y1)


def _is_near_roi_edge(u, v, roi_xyxy, margin_px):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    m = int(max(0, margin_px))
    return (u <= x1 + m) or (u >= x2 - 1 - m) or (v <= y1 + m) or (v >= y2 - 1 - m)


def _make_top_mask(depth_roi, valid_mask, top_percent):
    depth_valid = depth_roi[valid_mask]
    if depth_valid.size == 0:
        return None

    z_top = float(np.percentile(depth_valid, top_percent))
    mad = _robust_mad(depth_valid)
    band = float(np.clip(BAND_MAD_K * mad, BAND_MIN_M, BAND_MAX_M))
    return ((depth_roi <= (z_top + band)) & valid_mask).astype(np.uint8)


def _clean_mask(mask_u8):
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    out = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=CLOSE_ITERS)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    return (out > 0).astype(np.uint8)


def _select_best_blob(mask_u8):
    num, labels = cv2.connectedComponents(mask_u8)
    best = None
    best_score = -1.0

    for i in range(1, num):
        m = (labels == i)
        area = int(m.sum())
        if area < MIN_AREA:
            continue

        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1))
        compact = float(area) / (bbox_area + 1e-9)

        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        dmax = float(dist.max())
        if dmax < D_MIN:
            continue

        score = (dmax ** 2) * compact
        if score > best_score:
            best_score = score
            best = m

    return best


def _choose_pick_point(dist, mask_bool):
    dmax = float(dist.max())
    if dmax <= 0:
        return None

    M = cv2.moments(mask_bool.astype(np.uint8))
    if M["m00"] > 0:
        xc = float(M["m10"] / M["m00"])
        yc = float(M["m01"] / M["m00"])
    else:
        ys0, xs0 = np.where(mask_bool)
        if xs0.size == 0:
            return None
        xc, yc = float(xs0.mean()), float(ys0.mean())

    kr = float(np.clip(DT_KEEP_RATIO, 0.5, 0.99))
    cand = (dist >= (kr * dmax)) & mask_bool
    ys, xs = np.where(cand)

    if xs.size == 0:
        v0, u0 = np.unravel_index(int(dist.argmax()), dist.shape)
        return int(u0), int(v0)

    dx = xs - xc
    dy = ys - yc
    center_d2 = (dx * dx + dy * dy)
    center_d2 = center_d2 / (float(center_d2.max()) + 1e-9)

    dist_n = dist[ys, xs] / (dmax + 1e-9)
    cw = float(np.clip(DT_CENTER_WEIGHT, 0.0, 1.0))
    score = (1.0 - cw) * dist_n - cw * center_d2

    k = int(np.argmax(score))
    return int(xs[k]), int(ys[k])


def _detect_pick_uv(depth_snap_m, roi_xyxy):
    H, W = depth_snap_m.shape[:2]
    x0, y0, w, h = _parse_roi_xyxy(roi_xyxy, W, H)

    depth_roi = depth_snap_m[y0:y0 + h, x0:x0 + w]
    valid = (depth_roi > DEPTH_MIN_M) & (depth_roi < DEPTH_MAX_M)

    valid_ratio = float(valid.sum()) / float(depth_roi.size + 1e-9)
    if valid_ratio < VALID_RATIO_MIN:
        return None

    for top_percent in (TOP_PERCENT_PRIMARY, TOP_PERCENT_FALLBACK):
        top_mask = _make_top_mask(depth_roi, valid, top_percent)
        if top_mask is None:
            continue

        top_mask = _clean_mask(top_mask)
        if int(top_mask.sum()) < MIN_AREA:
            continue

        blob = _select_best_blob(top_mask)
        if blob is None:
            continue

        dist = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5)
        pick = _choose_pick_point(dist, blob)
        if pick is None:
            continue

        u_roi, v_roi = pick
        return (x0 + u_roi), (y0 + v_roi), blob

    return None


def _pca_axis_from_mask(mask_bool):
    m = mask_bool.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax <= 1e-6:
        return None

    core = (dist >= (PCA_CORE_RATIO * dmax)) & mask_bool
    ys, xs = np.where(core)
    if xs.size < PCA_MIN_CORE_POINTS:
        ys, xs = np.where(mask_bool)

    if xs.size < PCA_MIN_POINTS:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    pts0 = pts - pts.mean(axis=0)

    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    e1 = float(eigvals[order[0]])
    e2 = float(eigvals[order[1]]) if eigvals.size > 1 else 0.0
    v1 = eigvecs[:, order[0]]

    ratio = (e1 / (e2 + 1e-9)) if e2 > 0 else 999.0

    vx, vy = float(v1[0]), float(v1[1])
    n = (vx * vx + vy * vy) ** 0.5 + 1e-9
    vx, vy = vx / n, vy / n

    # sign stabilize: generally point upward
    if vy > 0.0:
        vx, vy = -vx, -vy

    yaw_img_deg = float(np.degrees(np.arctan2(vy, vx)))
    return {"vx": vx, "vy": vy, "ratio": ratio, "yaw_img_deg": yaw_img_deg}


def _map_yaw_image_to_work(yaw_img_deg: float) -> float:
    return _wrap_deg(YAW_SIGN * (yaw_img_deg - YAW_OFFSET_DEG))

def _angle_y_axis_pm90(vx: float, vy: float) -> float:
    """
    y축 기준 각도 (-90 ~ +90)
    - 0°  : 세로(y축)
    - +90 : 오른쪽(x+)
    - -90 : 왼쪽(x-)
    """
    ang = float(np.degrees(np.arctan2(vx, vy)))  # 핵심: atan2(vx, vy)

    # line(180° 대칭) 접기 -> [-90, 90]
    if ang > 90.0:
        ang -= 180.0
    elif ang < -90.0:
        ang += 180.0
    return float(ang)


class PickPipeline:
    def __init__(self, coord):
        self.coord = coord

    def run(self, snapshot, clicked_uv, roi_xyxy):
        # (2) clicked -> flat world xy (snapshot depth 기준)
        flat_clicked_xy = None
        if clicked_uv:
            flat_clicked_xy = []
            for (u, v) in clicked_uv:
                Pw = self.coord.pixel_to_world_from_depthmap(u, v, snapshot.depth_snap_m)
                if Pw is None:
                    flat_clicked_xy.append([None, None])
                else:
                    flat_clicked_xy.append([float(Pw[0]), float(Pw[1])])

        det = _detect_pick_uv(snapshot.depth_snap_m, roi_xyxy)
        if det is None:
            return {"ok": False, "u": None, "v": None, "xyz_angle": None, "u2v2": None,
                    "flat_clicked_xy": flat_clicked_xy}

        u_img, v_img, blob_mask = det

        Pw = self.coord.pixel_to_world_from_depthmap(u_img, v_img, snapshot.depth_snap_m)
        if Pw is None:
            return {"ok": False, "u": u_img, "v": v_img, "xyz_angle": None, "u2v2": None,
                    "flat_clicked_xy": flat_clicked_xy}

        X, Y, Z = float(Pw[0]), float(Pw[1]), float(Pw[2])
        Zapp = Z + Z_APPROACH_CM

        if _is_near_roi_edge(u_img, v_img, roi_xyxy, EDGE_MARGIN_PX) and (Zapp <= Z_APPROACH_MIN_CM):
            return {"ok": False, "u": u_img, "v": v_img, "xyz_angle": None, "u2v2": None,
                    "flat_clicked_xy": flat_clicked_xy}

        pca = _pca_axis_from_mask(blob_mask)
        if pca is None:
            if REJECT_IF_NO_YAW:
                return {"ok": False, "u": u_img, "v": v_img, "xyz_angle": None, "u2v2": None,
                        "flat_clicked_xy": flat_clicked_xy}
            # yaw 없이 성공 허용이면 angle=None로 반환 (화살표도 없음)
            return {"ok": True, "u": u_img, "v": v_img, "xyz_angle": (X, Y, Zapp, None), "u2v2": None,
                    "flat_clicked_xy": flat_clicked_xy}

        # 이제부터는 pca가 dict인 게 보장됨
        if REJECT_IF_NO_YAW and (float(pca["ratio"]) < PCA_ANISO_RATIO_MIN):
            return {"ok": False, "u": u_img, "v": v_img, "xyz_angle": None, "u2v2": None,
                    "flat_clicked_xy": flat_clicked_xy}


        vx, vy = float(pca["vx"]), float(pca["vy"])
        angle = -_angle_y_axis_pm90(vx, vy)
        H, W = snapshot.color_bgr.shape[:2]
        u2 = int(np.clip(u_img + vx * YAW_ARROW_LEN_PX, 0, W - 1))
        v2 = int(np.clip(v_img + vy * YAW_ARROW_LEN_PX, 0, H - 1))

        return {"ok": True, "u": u_img, "v": v_img, "xyz_angle": (X, Y, Zapp, angle), "u2v2": (u2, v2),
                "flat_clicked_xy": flat_clicked_xy}
