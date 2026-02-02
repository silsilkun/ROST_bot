#!/usr/bin/env python3
# realsense_pick_yaw_allinone.py
# All-in-one: RealSense viewer + snapshot median depth + pick + PCA yaw (stable)
# - PCA direction sign fix (no random 180° flips)
# - Yaw computed from IMAGE PCA -> mapped to WORK yaw via (offset, sign)
# - Near-circular: yaw=None (viewer holds last_yaw), not forced 0.0
# - Gripper yaw fold: final yaw is folded to [-90, +90] (180° symmetry)
# - Handles device busy (EBUSY) start error safely
# - pipeline.stop() only if started

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pyrealsense2 as rs


# =========================================================
# Utils
# =========================================================
def wrap_deg(a: float) -> float:
    """Wrap degrees to (-180, 180]."""
    return float((a + 180.0) % 360.0 - 180.0)


def wrap_deg_180_for_gripper(a: float) -> float:
    """
    Fold yaw to [-90, +90] assuming 180° symmetry of the gripper.
    1) wrap to (-180, 180]
    2) fold to [-90, 90]
    """
    x = (a + 180.0) % 360.0 - 180.0
    if x > 90.0:
        x -= 180.0
    elif x < -90.0:
        x += 180.0
    return float(x)


# =========================================================
# Config / Result
# =========================================================
@dataclass
class PickConfig:
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30

    ROI: Tuple[int, int, int, int] = (480, 127, 818, 350)

    # depth valid range (meters)
    DEPTH_MIN: float = 0.2
    DEPTH_MAX: float = 2.0

    # top-percent selection
    TOP_PERCENT_PRIMARY: float = 3.0
    TOP_PERCENT_FALLBACK: float = 5.0

    # adaptive band (meters)
    BAND_MIN_M: float = 0.008
    BAND_MAX_M: float = 0.030
    BAND_MAD_K: float = 3.0

    # blob filters
    MIN_AREA: int = 800
    D_MIN: float = 6.0  # pixels in distanceTransform

    # pick selection (DT + centroid soft pull)
    DT_KEEP_RATIO: float = 0.85
    DT_CENTER_WEIGHT: float = 0.30

    # approach offset
    Z_APPROACH_CM: float = 3.0

    # morphology
    MORPH_KERNEL: int = 3
    CLOSE_ITERS: int = 2
    OPEN_ITERS: int = 1

    # PCA yaw
    PCA_MIN_POINTS: int = 100
    PCA_ANISO_RATIO_MIN: float = 1.20

    # snapshot
    SNAP_N: int = 3
    VALID_RATIO_MIN: float = 0.30

    # pixel->world depth sampling radius (depth_snap map)
    R: int = 2

    # calibration / env correction
    SAVE_FILE: str = "camcalib.npz"
    M_TO_CM: float = 100.0
    FLIP_XYZ: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    OFFSET_CM: Tuple[float, float, float] = (81.5, 15.9, 0.0)

    # =====================================================
    # Yaw mapping (IMAGE -> WORK)
    # =====================================================
    # image yaw = atan2(vy, vx) in degrees (x right, y down)
    # work yaw = wrap( sign * (img - offset) )
    YAW_MODE: str = "image"
    YAW_OFFSET_DEG: float = 90.0
    YAW_SIGN: float = -1.0

    # arrow length in pixels for visualization
    YAW_ARROW_LEN_PX: int = 60


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

    # for arrow drawing
    u2: Optional[int] = None
    v2: Optional[int] = None

    debug: Optional[Dict[str, Any]] = None


# =========================================================
# Calibration / Coordinate
# =========================================================
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
                    if d > 0.0 and (cfg.DEPTH_MIN <= d <= cfg.DEPTH_MAX):
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


# =========================================================
# Core helpers
# =========================================================
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


# =========================================================
# PCA axis (with sign stabilization)
# =========================================================
def pca_axis_from_mask(mask_bool: np.ndarray, cfg: PickConfig,
                       core_ratio: float = 0.60,   # 0.55~0.70 추천
                       min_core_points: int = 120):
    """
    PCA on CORE pixels only (DT high region).
    This suppresses boundary/support contamination.
    """
    m = mask_bool.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax <= 1e-6:
        return None

    core = (dist >= (core_ratio * dmax)) & mask_bool
    ys, xs = np.where(core)

    # fallback
    if xs.size < min_core_points:
        ys, xs = np.where(mask_bool)

    if xs.size < cfg.PCA_MIN_POINTS:
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

    # sign stabilization: point upward in image (y decreasing)
    if vy > 0.0:
        vx, vy = -vx, -vy

    yaw_img_deg = float(np.degrees(np.arctan2(vy, vx)))
    return {
        "vx": vx, "vy": vy, "ratio": ratio,
        "yaw_img_deg": yaw_img_deg,
        "core_ratio": float(core_ratio),
        "core_pts": int(xs.size),
        "dmax": float(dmax),
    }



# =========================================================
# Compute pick from snapshot (core algorithm)
# =========================================================
def compute_pick_from_snapshot(depth_snap_m: np.ndarray, cfg: PickConfig, coord: Coordinate) -> PickResult:
    try:
        x0, y0, w, h = parse_roi_xyxy(cfg.ROI, cfg.WIDTH, cfg.HEIGHT)
    except Exception as e:
        return PickResult(False, f"ROI_ERR: {e}")

    depth_roi = depth_snap_m[y0:y0 + h, x0:x0 + w]

    valid = (depth_roi > cfg.DEPTH_MIN) & (depth_roi < cfg.DEPTH_MAX)
    valid_ratio = float(valid.sum()) / float(depth_roi.size + 1e-9)
    if valid_ratio < cfg.VALID_RATIO_MIN:
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

    # =========================
    # PCA yaw (image-based)
    # =========================
    yaw_work_deg = None
    yaw_work_rad = None
    u2 = v2 = None

    pca_info = pca_axis_from_mask(best, cfg)
    pca_ratio = float(pca_info["ratio"]) if pca_info is not None else 0.0
    yaw_img_deg = float(pca_info["yaw_img_deg"]) if pca_info is not None else None

    if pca_info is not None:
        if pca_ratio < cfg.PCA_ANISO_RATIO_MIN:
            yaw_work_deg = None
            yaw_work_rad = None
            u2, v2 = None, None
        else:
            yaw_work_deg = wrap_deg(cfg.YAW_SIGN * (yaw_img_deg - cfg.YAW_OFFSET_DEG))
            yaw_work_rad = float(np.radians(yaw_work_deg))

            vx, vy = float(pca_info["vx"]), float(pca_info["vy"])
            u2 = int(np.clip(u_img + vx * cfg.YAW_ARROW_LEN_PX, 0, cfg.WIDTH - 1))
            v2 = int(np.clip(v_img + vy * cfg.YAW_ARROW_LEN_PX, 0, cfg.HEIGHT - 1))

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
        "yaw_img_deg": float(yaw_img_deg) if yaw_img_deg is not None else None,
        "yaw_offset_deg": float(cfg.YAW_OFFSET_DEG),
        "yaw_sign": float(cfg.YAW_SIGN),
        "yaw_mode": cfg.YAW_MODE,
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


# =========================================================
# UI: draw / render / snapshot
# =========================================================
def draw_roi(img, roi_xyxy, color=(0, 255, 0), th=2):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
    return out


def render_result(color_img: np.ndarray, res: PickResult, cfg: PickConfig,
                  yaw_display=None, yaw_held=False) -> np.ndarray:
    out = color_img.copy()
    out = draw_roi(out, cfg.ROI)

    if not res.ok:
        cv2.putText(out, res.msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out

    if res.bbox_img is not None:
        x1, y1, x2, y2 = res.bbox_img
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.circle(out, (int(res.u_img), int(res.v_img)), 8, (0, 0, 255), -1)

    if res.u2 is not None and res.v2 is not None and (res.u2, res.v2) != (res.u_img, res.v_img):
        cv2.arrowedLine(out, (res.u_img, res.v_img), (int(res.u2), int(res.v2)),
                        (255, 0, 0), 2, tipLength=0.2)

    lines = []
    lines.append(f"uv=({res.u_img},{res.v_img})")
    if res.xyz_approach_cm is not None:
        X, Y, Z = res.xyz_approach_cm
        lines.append(f"XYZ_approach=[{X:.1f},{Y:.1f},{Z:.1f}] cm")

    if yaw_display is None:
        lines.append("yaw_work=None")
    else:
        suffix = " (held)" if yaw_held else ""
        lines.append(f"yaw_work={yaw_display:.1f} deg{suffix}")

    dbg = res.debug or {}
    lines.append(f"valid={dbg.get('valid_ratio',0):.2f} top%={dbg.get('top_percent',0)} band={dbg.get('band_m',0)*100:.1f}cm")
    if dbg.get("yaw_img_deg", None) is not None:
        lines.append(f"yaw_img={dbg.get('yaw_img_deg',0):.1f} off={dbg.get('yaw_offset_deg',0):.1f} sign={dbg.get('yaw_sign',0):.1f}")

    y0 = 50
    for i, s in enumerate(lines):
        y = y0 + 35 * i
        cv2.putText(out, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out


def get_snapshot_median(pipeline: rs.pipeline, align: rs.align, cfg: PickConfig, snap_n: int):
    depths = []
    snap_color = None

    for _ in range(snap_n):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f or not depth_f:
            continue

        if snap_color is None:
            snap_color = np.asanyarray(color_f.get_data()).copy()

        depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * 0.001
        depths.append(depth_m)

    if snap_color is None or len(depths) < max(1, snap_n // 2):
        return None, None

    depth_snap_m = np.median(np.stack(depths, axis=0), axis=0).astype(np.float32)
    return snap_color, depth_snap_m


# =========================================================
# Main loop
# =========================================================
def main():
    cfg = PickConfig()
    coord = Coordinate(cfg)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, cfg.WIDTH, cfg.HEIGHT, rs.format.bgr8, cfg.FPS)
    config.enable_stream(rs.stream.depth, cfg.WIDTH, cfg.HEIGHT, rs.format.z16, cfg.FPS)

    align = rs.align(rs.stream.color)

    cv2.namedWindow("LIVE", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RESULT", cv2.WINDOW_AUTOSIZE)

    print("SPACE: snapshot+compute | r: reset | esc/q: quit")

    last_result = None
    last_yaw_deg = None

    started = False
    try:
        try:
            pipeline.start(config)
            started = True
        except RuntimeError as e:
            print("\n[RealSense START FAILED]")
            print(str(e))
            print("\nPossible fixes:")
            print("1) Close realsense-viewer / other camera apps / ROS nodes using the device")
            print("2) Kill remaining python/rs processes")
            print("3) Unplug/replug the camera (USB3)")
            return

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            live = draw_roi(color_img, cfg.ROI)

            if last_result is not None and last_result.ok:
                cv2.circle(live, (last_result.u_img, last_result.v_img), 6, (0, 0, 255), -1)

            cv2.imshow("LIVE", live)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("r"):
                last_result = None
                last_yaw_deg = None
                blank = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "RESET", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                cv2.imshow("RESULT", blank)
                print("RESET")

            if key == ord(" "):
                snap_color, depth_snap_m = get_snapshot_median(pipeline, align, cfg, snap_n=cfg.SNAP_N)
                if snap_color is None:
                    last_result = PickResult(False, "SENSOR_BAD (no frames)")
                    cv2.imshow("RESULT", render_result(color_img, last_result, cfg, last_yaw_deg, False))
                    print(last_result.msg)
                    continue

                res = compute_pick_from_snapshot(depth_snap_m, cfg, coord)
                last_result = res

                # yaw hold policy
                yaw_held = False
                yaw_to_send = res.yaw_work_deg
                if yaw_to_send is not None:
                    last_yaw_deg = float(yaw_to_send)
                else:
                    yaw_to_send = last_yaw_deg
                    yaw_held = (yaw_to_send is not None)

                # ✅ fold yaw to [-90, 90] for gripper symmetry (final)
                if yaw_to_send is not None:
                    yaw_to_send = wrap_deg_180_for_gripper(yaw_to_send)

                # SEND print (include pca_ratio / yaw_img for diagnosis)
                if res.ok and res.xyz_approach_cm is not None:
                    xyz = res.xyz_approach_cm
                    dbg = res.debug or {}
                    pr = dbg.get("pca_ratio", 0.0)
                    yi = dbg.get("yaw_img_deg", None)
                    if yaw_to_send is not None:
                        held_txt = " (held)" if yaw_held else ""
                        if yi is None:
                            print(f"SEND → xyz={list(xyz)} yaw_deg={yaw_to_send:.1f}{held_txt} pca_ratio={pr:.2f}")
                        else:
                            print(f"SEND → xyz={list(xyz)} yaw_deg={yaw_to_send:.1f}{held_txt} pca_ratio={pr:.2f} yaw_img={yi:.1f}")
                    else:
                        print(f"SEND → xyz={list(xyz)} yaw_deg=None pca_ratio={pr:.2f}")
                else:
                    print(res.msg)

                view = render_result(snap_color, res, cfg, yaw_to_send, yaw_held)
                cv2.imshow("RESULT", view)

    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
