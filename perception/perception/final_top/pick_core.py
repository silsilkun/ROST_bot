# pick_core.py
# Core algorithm: depth-snapshot based pick + PCA yaw (world, robust)
# - Uses median depth map (depth_snap_m) for segmentation AND pixel->world Z
# - Robust yaw: search P2 along PCA axis inside mask + fallback policies

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ==============================
# Config
# ==============================
@dataclass
class PickConfig:
    WIDTH: int = 1280
    HEIGHT: int = 720

    ROI: Tuple[int, int, int, int] = (480, 127, 818, 350)

    DEPTH_MIN: float = 0.2
    DEPTH_MAX: float = 2.0

    TOP_PERCENT_PRIMARY: float = 3.0
    TOP_PERCENT_FALLBACK: float = 5.0

    BAND_MIN_M: float = 0.008
    BAND_MAX_M: float = 0.030
    BAND_MAD_K: float = 3.0

    MIN_AREA: int = 800
    D_MIN: float = 6.0

    DT_KEEP_RATIO: float = 0.85
    DT_CENTER_WEIGHT: float = 0.30

    Z_APPROACH_CM: float = 3.0

    MORPH_KERNEL: int = 3
    CLOSE_ITERS: int = 2
    OPEN_ITERS: int = 1

    # PCA yaw
    PCA_MIN_POINTS: int = 100
    PCA_ANISO_RATIO_MIN: float = 1.20

    # yaw sampling / robustness
    YAW_STEP_LIST_PX: Tuple[int, ...] = (12, 18, 25, 35, 50, 70)  # search steps
    YAW_MIN_WORLD_DIST_CM: float = 0.6                             # min XY separation in cm

    # pixel->world depth sampling radius (for depth_snap map)
    R: int = 2

    # calibration / env correction
    SAVE_FILE: str = "camcalib.npz"
    M_TO_CM: float = 100.0
    FLIP_XYZ: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    OFFSET_CM: Tuple[float, float, float] = (81.5, 15.9, 0.0)


# ==============================
# Result
# ==============================
@dataclass
class PickResult:
    ok: bool
    msg: str

    u_img: int = -1
    v_img: int = -1
    bbox_img: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2

    xyz_approach_cm: Optional[Tuple[float, float, float]] = None

    yaw_work_deg: Optional[float] = None
    yaw_work_rad: Optional[float] = None

    # for drawing arrow in image
    u2: Optional[int] = None
    v2: Optional[int] = None

    debug: Optional[Dict[str, Any]] = None


# ==============================
# Calibration / Coordinate
# ==============================
_CALIB_CACHE = None


def load_calib(save_file: str):
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE

    if not os.path.exists(save_file):
        raise FileNotFoundError(f"캘리브 파일 없음: {save_file}")

    data = np.load(save_file)
    _CALIB_CACHE = {
        "T": data["T_cam_to_work"].astype(np.float64),
        "K": data["camera_matrix"].astype(np.float64),
        "D": data["dist_coeffs"].astype(np.float64),
    }
    return _CALIB_CACHE


class Coordinate:
    def __init__(self, cfg: PickConfig):
        c = load_calib(cfg.SAVE_FILE)
        self.cfg = cfg
        self.T = c["T"]
        self.K = c["K"]
        self.D = c["D"]

    def pixel_to_world_from_depthmap(self, u: int, v: int, depth_snap_m: np.ndarray) -> Optional[np.ndarray]:
        """
        depth_snap_m: (H,W) float32 meters (aligned to color)
        Returns Pw (4,) with XYZ in cm + homogeneous 1.
        """
        cfg = self.cfg
        H, W = depth_snap_m.shape[:2]
        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))

        depths = []
        for du in range(-cfg.R, cfg.R + 1):
            for dv in range(-cfg.R, cfg.R + 1):
                uu = u + du
                vv = v + dv
                if 0 <= uu < W and 0 <= vv < H:
                    d = float(depth_snap_m[vv, uu])
                    if d > 0.0:
                        depths.append(d)

        if not depths:
            return None

        Z_cm = float(np.median(depths)) * cfg.M_TO_CM

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        pts = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        uc, vc = float(und[0, 0, 0]), float(und[0, 0, 1])

        # Keep your axis mapping (u->Yc, v->Xc)
        Yc = (uc - cx) * Z_cm / fx
        Xc = (vc - cy) * Z_cm / fy
        Pc = np.array([Xc, Yc, Z_cm, 1.0], dtype=np.float64)

        Pw = self.T @ Pc

        Pw[0] = cfg.FLIP_XYZ[0] * Pw[0] + cfg.OFFSET_CM[0]
        Pw[1] = cfg.FLIP_XYZ[1] * Pw[1] + cfg.OFFSET_CM[1]
        Pw[2] = cfg.FLIP_XYZ[2] * Pw[2] + cfg.OFFSET_CM[2]
        return Pw


# ==============================
# Helpers
# ==============================
def parse_roi_xyxy(roi_xyxy, img_w, img_h):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI invalid after clamp: {(x1, y1, x2, y2)}")
    return x1, y1, (x2 - x1), (y2 - y1)


def robust_mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def make_top_mask(depth_roi: np.ndarray, valid_mask: np.ndarray, top_percent: float, cfg: PickConfig):
    depth_valid = depth_roi[valid_mask]
    if depth_valid.size == 0:
        return None, None, None

    z_top = float(np.percentile(depth_valid, top_percent))
    mad = robust_mad(depth_valid)
    band = float(np.clip(cfg.BAND_MAD_K * mad, cfg.BAND_MIN_M, cfg.BAND_MAX_M))

    top_mask = (depth_roi <= (z_top + band)).astype(np.uint8)
    return top_mask, z_top, band


def clean_mask(mask_u8: np.ndarray, cfg: PickConfig):
    k = np.ones((cfg.MORPH_KERNEL, cfg.MORPH_KERNEL), np.uint8)
    out = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=cfg.CLOSE_ITERS)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k, iterations=cfg.OPEN_ITERS)
    return out


def safe_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ==============================
# Pick point selection (DT + centroid soft pull)
# ==============================
def choose_pick_point_from_dist(dist: np.ndarray, mask_bool: np.ndarray, cfg: PickConfig):
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

    kr = float(np.clip(cfg.DT_KEEP_RATIO, 0.5, 0.99))
    cand = (dist >= (kr * dmax)) & mask_bool
    ys, xs = np.where(cand)

    if xs.size == 0:
        v0, u0 = np.unravel_index(int(dist.argmax()), dist.shape)
        return int(u0), int(v0), dmax, float(dist[v0, u0])

    dx = (xs - xc)
    dy = (ys - yc)
    center_d2 = dx * dx + dy * dy
    center_d2 = center_d2 / (float(center_d2.max()) + 1e-9)

    dist_n = dist[ys, xs] / (dmax + 1e-9)

    cw = float(np.clip(cfg.DT_CENTER_WEIGHT, 0.0, 1.0))
    score = (1.0 - cw) * dist_n - cw * center_d2

    k = int(np.argmax(score))
    u, v = int(xs[k]), int(ys[k])
    return u, v, dmax, float(dist[v, u])


# ==============================
# PCA axis + robust world yaw
# ==============================
def pca_axis_from_mask(mask_bool: np.ndarray, cfg: PickConfig):
    ys, xs = np.where(mask_bool)
    if xs.size < cfg.PCA_MIN_POINTS:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean

    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    e1 = float(eigvals[order[0]])
    e2 = float(eigvals[order[1]]) if eigvals.size > 1 else 0.0
    v1 = eigvecs[:, order[0]]  # (vx, vy)

    ratio = (e1 / (e2 + 1e-9)) if e2 > 0 else 999.0

    vx, vy = float(v1[0]), float(v1[1])
    n = (vx * vx + vy * vy) ** 0.5 + 1e-9
    vx, vy = vx / n, vy / n

    return {"vx": vx, "vy": vy, "ratio": ratio}


def _roi_mask_contains(best_mask_roi: np.ndarray, x0: int, y0: int, u_img: int, v_img: int) -> bool:
    """Check if (u_img,v_img) lies in ROI and inside best_mask_roi."""
    ur = u_img - x0
    vr = v_img - y0
    if ur < 0 or vr < 0:
        return False
    if vr >= best_mask_roi.shape[0] or ur >= best_mask_roi.shape[1]:
        return False
    return bool(best_mask_roi[vr, ur])


def compute_world_yaw_from_pca_robust(
    u_img: int,
    v_img: int,
    pca_info: dict,
    best_mask_roi: np.ndarray,
    x0: int,
    y0: int,
    depth_snap_m: np.ndarray,
    coord: Coordinate,
    cfg: PickConfig,
):
    """
    Robust yaw:
    - P1 from pick point
    - Search P2 along PCA axis in BOTH directions and multiple pixel steps
    - P2 must be inside best mask + valid world transform + min world distance
    """
    if pca_info is None:
        return None

    P1 = coord.pixel_to_world_from_depthmap(u_img, v_img, depth_snap_m)
    if P1 is None:
        return None

    vx, vy = float(pca_info["vx"]), float(pca_info["vy"])

    # Search along +axis and -axis
    dirs = [(vx, vy), (-vx, -vy)]
    for sx, sy in dirs:
        for step in cfg.YAW_STEP_LIST_PX:
            u2 = int(np.clip(u_img + sx * step, 0, cfg.WIDTH - 1))
            v2 = int(np.clip(v_img + sy * step, 0, cfg.HEIGHT - 1))

            # must remain inside best mask region
            if not _roi_mask_contains(best_mask_roi, x0, y0, u2, v2):
                continue

            P2 = coord.pixel_to_world_from_depthmap(u2, v2, depth_snap_m)
            if P2 is None:
                continue

            dx = float(P2[0] - P1[0])
            dy = float(P2[1] - P1[1])
            if (dx * dx + dy * dy) < (cfg.YAW_MIN_WORLD_DIST_CM * cfg.YAW_MIN_WORLD_DIST_CM):
                continue

            yaw = float(np.arctan2(dy, dx))
            return {
                "yaw_work_rad": yaw,
                "yaw_work_deg": float(np.degrees(yaw)),
                "u2": u2,
                "v2": v2,
            }

    return None


# ==============================
# Main compute
# ==============================
def compute_pick_from_snapshot(depth_snap_m: np.ndarray, cfg: PickConfig, coord: Coordinate) -> PickResult:
    """
    depth_snap_m: (H,W) float32 meters, aligned to color frame
    """
    try:
        x0, y0, w, h = parse_roi_xyxy(cfg.ROI, cfg.WIDTH, cfg.HEIGHT)
    except Exception as e:
        return PickResult(False, f"ROI_ERR: {e}")

    depth_roi = depth_snap_m[y0:y0 + h, x0:x0 + w]

    valid = (depth_roi > cfg.DEPTH_MIN) & (depth_roi < cfg.DEPTH_MAX)
    valid_ratio = float(valid.sum()) / float(depth_roi.size + 1e-9)
    if valid_ratio < 0.30:
        return PickResult(False, f"SENSOR_BAD (valid_ratio={valid_ratio:.2f})", debug={"valid_ratio": valid_ratio})

    top_percent_used = cfg.TOP_PERCENT_PRIMARY
    top_mask, z_top, band = make_top_mask(depth_roi, valid, cfg.TOP_PERCENT_PRIMARY, cfg)
    if top_mask is None:
        return PickResult(False, "NO_VALID_DEPTH", debug={"valid_ratio": valid_ratio})

    top_mask = clean_mask(top_mask, cfg)

    if int(top_mask.sum()) < cfg.MIN_AREA:
        top_percent_used = cfg.TOP_PERCENT_FALLBACK
        top_mask, z_top, band = make_top_mask(depth_roi, valid, cfg.TOP_PERCENT_FALLBACK, cfg)
        if top_mask is None:
            return PickResult(False, "NO_VALID_DEPTH", debug={"valid_ratio": valid_ratio})
        top_mask = clean_mask(top_mask, cfg)

    num, labels = cv2.connectedComponents(top_mask)

    best = None
    best_score = -1.0
    best_bbox = None
    best_area = 0
    best_compact = 0.0
    best_dmax = 0.0

    for i in range(1, num):
        mask = (labels == i)
        area = int(mask.sum())
        if area < cfg.MIN_AREA:
            continue

        bbox = safe_bbox_from_mask(mask)
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1))
        compact = float(area) / (bbox_area + 1e-9)

        dist_i = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        dmax_i = float(dist_i.max())
        if dmax_i < cfg.D_MIN:
            continue

        score = (dmax_i ** 2) * compact
        if score > best_score:
            best_score = score
            best = mask
            best_bbox = bbox
            best_area = area
            best_compact = compact
            best_dmax = dmax_i

    if best is None:
        return PickResult(False, "NO_PICK (no blob)", debug={"valid_ratio": valid_ratio})

    dist = cv2.distanceTransform(best.astype(np.uint8), cv2.DIST_L2, 5)
    pick = choose_pick_point_from_dist(dist, best, cfg)
    if pick is None:
        return PickResult(False, "NO_PICK (bad dist)", debug={"valid_ratio": valid_ratio})

    u_roi, v_roi, dmax, chosen_d = pick
    if chosen_d < cfg.D_MIN:
        return PickResult(False, f"TOO_THIN (chosen_d={chosen_d:.1f})",
                          debug={"chosen_d": float(chosen_d), "dmin": cfg.D_MIN})

    u_img = x0 + int(u_roi)
    v_img = y0 + int(v_roi)

    Pw = coord.pixel_to_world_from_depthmap(u_img, v_img, depth_snap_m)
    if Pw is None:
        return PickResult(False, "XYZ_FAIL")

    X, Y, Z = float(Pw[0]), float(Pw[1]), float(Pw[2])
    Z_approach = Z + cfg.Z_APPROACH_CM

    # PCA yaw
    yaw_work_deg = None
    yaw_work_rad = None
    u2 = v2 = None

    pca_info = pca_axis_from_mask(best, cfg)
    pca_ratio = float(pca_info["ratio"]) if pca_info is not None else 0.0

    if pca_info is not None:
        # If near-circular: force yaw=0 (policy) to avoid None/noisy values
        if pca_ratio < cfg.PCA_ANISO_RATIO_MIN:
            yaw_work_deg = 0.0
            yaw_work_rad = 0.0
            u2, v2 = u_img, v_img
        else:
            yw = compute_world_yaw_from_pca_robust(
                u_img=u_img, v_img=v_img, pca_info=pca_info,
                best_mask_roi=best, x0=x0, y0=y0,
                depth_snap_m=depth_snap_m, coord=coord, cfg=cfg
            )
            if yw is not None:
                yaw_work_deg = float(yw["yaw_work_deg"])
                yaw_work_rad = float(yw["yaw_work_rad"])
                u2, v2 = int(yw["u2"]), int(yw["v2"])
            else:
                # If we couldn't find P2: leave None (viewer can hold last_yaw)
                yaw_work_deg = None
                yaw_work_rad = None
                u2, v2 = None, None

    xmin, ymin, xmax, ymax = best_bbox
    bbox_img = (x0 + xmin, y0 + ymin, x0 + xmax, y0 + ymax)

    dbg = {
        "valid_ratio": valid_ratio,
        "top_percent": float(top_percent_used),
        "z_top": float(z_top) if z_top is not None else 0.0,
        "band_m": float(band) if band is not None else 0.0,
        "best_area": int(best_area),
        "best_compact": float(best_compact),
        "best_dmax": float(best_dmax),
        "dmax": float(dmax),
        "chosen_d": float(chosen_d),
        "dt_keep_ratio": float(cfg.DT_KEEP_RATIO),
        "dt_center_weight": float(cfg.DT_CENTER_WEIGHT),
        "pca_ratio": float(pca_ratio),
        "yaw_step_list": list(cfg.YAW_STEP_LIST_PX),
        "yaw_min_world_cm": float(cfg.YAW_MIN_WORLD_DIST_CM),
    }

    return PickResult(
        ok=True,
        msg="OK",
        u_img=u_img,
        v_img=v_img,
        bbox_img=bbox_img,
        xyz_approach_cm=(X, Y, Z_approach),
        yaw_work_deg=yaw_work_deg,
        yaw_work_rad=yaw_work_rad,
        u2=u2,
        v2=v2,
        debug=dbg,
    )
