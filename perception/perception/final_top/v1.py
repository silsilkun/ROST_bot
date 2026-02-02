# realsense_loop_style_pick_xyxy.py
import os
import cv2
import numpy as np
import pyrealsense2 as rs

# =========================================================
# FIXED: stream / windows / ROI (xyxy)
# =========================================================
WIDTH, HEIGHT, FPS = 1280, 720, 30

# 창 크기(표시용) 고정
WIN_W, WIN_H = 1280, 720

# ROI를 (x1, y1, x2, y2)로 고정
ROI = (480, 127, 834, 357)  # (x1, y1, x2, y2)  <-- 원하는 형식

# Depth-based pick parameters
DEPTH_MIN = 0.2
DEPTH_MAX = 2.0
TOP_PERCENT = 3
BAND_M = 0.02
MIN_AREA = 800
D_MIN = 6
SNAP_N = 3
VALID_RATIO_MIN = 0.30
Z_APPROACH_CM = 3.0

# =========================================================
# Coordinate (캘리브 기반, 너 코드 유지)
# =========================================================
SAVE_FILE = "camcalib.npz"

R = 2
M_TO_CM = 100.0
FLIP_XYZ = (-1.0, -1.0, -1.0)
OFFSET_CM = (81.5, 15.9, 0.0)

_CALIB = None


def load_calib():
    global _CALIB
    if _CALIB is not None:
        return _CALIB
    if not os.path.exists(SAVE_FILE):
        raise FileNotFoundError(f"캘리브 파일 없음: {SAVE_FILE}")

    data = np.load(SAVE_FILE)
    _CALIB = {
        "T": data["T_cam_to_work"].astype(np.float64),
        "K": data["camera_matrix"].astype(np.float64),
        "D": data["dist_coeffs"].astype(np.float64),
    }
    return _CALIB


class Coordinate:
    def __init__(self):
        c = load_calib()
        self.T = c["T"]
        self.K = c["K"]
        self.D = c["D"]

    def pixel_to_world(self, u, v, depth_frame):
        u = int(u)
        v = int(v)

        # 1) depth median 5x5
        depths = []
        for du in range(-R, R + 1):
            for dv in range(-R, R + 1):
                d = float(depth_frame.get_distance(u + du, v + dv))
                if d > 0.0:
                    depths.append(d)
        if not depths:
            return None
        Z = float(np.median(depths)) * M_TO_CM  # cm

        # 2) intrinsics
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        # 3) undistort pixel
        pts = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        uc, vc = und[0, 0]
        uc, vc = float(uc), float(vc)

        # 4) pixel -> camera (원본 축 매핑 유지)
        Yc = (uc - cx) * Z / fx
        Xc = (vc - cy) * Z / fy
        Pc = np.array([Xc, Yc, Z, 1.0], dtype=np.float64)

        # 5) camera -> work
        Pw = self.T @ Pc

        # 6) real env correction (기존 유지)
        Pw[0] = FLIP_XYZ[0] * Pw[0] + OFFSET_CM[0]
        Pw[1] = FLIP_XYZ[1] * Pw[1] + OFFSET_CM[1]
        Pw[2] = FLIP_XYZ[2] * Pw[2] + OFFSET_CM[2]
        return Pw


# =========================================================
# helpers
# =========================================================
def parse_roi_xyxy(roi_xyxy, img_w, img_h):
    """roi_xyxy=(x1,y1,x2,y2) -> (x0,y0,w,h) with clamping + asserts"""
    x1, y1, x2, y2 = map(int, roi_xyxy)

    # clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI invalid after clamp: {(x1,y1,x2,y2)}")

    x0, y0 = x1, y1
    w, h = x2 - x1, y2 - y1
    return x0, y0, w, h


def _draw_points(img: np.ndarray, points, radius=6):
    if not points:
        return img
    out = img.copy()
    for p in points:
        if len(p) < 2:
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), radius, (0, 0, 255), -1)
    return out


def _draw_roi_xyxy(img: np.ndarray, roi_xyxy):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def _safe_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# =========================================================
# compute pick on snapshot (Spacebar)
# =========================================================
def compute_pick_snapshot(pipeline, align, roi_xyxy, coord: Coordinate):
    """
    returns dict:
      ok: bool
      msg: str
      u_img, v_img: int
      bbox_img: (x1,y1,x2,y2)
      xyz_approach: [X,Y,Z_approach]
      debug: dict
      snap_color: np.ndarray
    """
    x0, y0, w, h = parse_roi_xyxy(roi_xyxy, WIDTH, HEIGHT)

    depths = []
    snap_color = None
    snap_depth_frame = None

    # 1) capture N depth frames, median
    for _ in range(SNAP_N):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f or not depth_f:
            continue

        if snap_color is None:
            snap_color = np.asanyarray(color_f.get_data()).copy()
        snap_depth_frame = depth_f

        depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * 0.001  # mm->m
        depths.append(depth_m)

    if snap_color is None or snap_depth_frame is None or len(depths) < max(1, SNAP_N // 2):
        return {"ok": False, "msg": "SENSOR_BAD (no frames)"}

    depth_snap = np.median(np.stack(depths, axis=0), axis=0)
    depth_roi = depth_snap[y0:y0 + h, x0:x0 + w]

    # 2) valid gate
    valid = (depth_roi > DEPTH_MIN) & (depth_roi < DEPTH_MAX)
    valid_ratio = float(valid.sum()) / float(depth_roi.size)
    if valid_ratio < VALID_RATIO_MIN:
        return {
            "ok": False,
            "msg": f"SENSOR_BAD (valid_ratio={valid_ratio:.2f})",
            "snap_color": snap_color,
        }

    # 3) top mask (depth 기준: 가까운 곳)
    depth_valid = depth_roi[valid]
    z_top = np.percentile(depth_valid, TOP_PERCENT)
    top_mask = (depth_roi <= (z_top + BAND_M)).astype(np.uint8)

    # 4) cleanup
    kernel = np.ones((3, 3), np.uint8)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) connected components -> best blob
    num, labels = cv2.connectedComponents(top_mask)
    best = None
    best_score = 0.0
    best_bbox = None
    best_area = 0
    best_compact = 0.0

    for i in range(1, num):
        mask = (labels == i)
        area = int(mask.sum())
        if area < MIN_AREA:
            continue

        bbox = _safe_bbox_from_mask(mask)
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        bbox_area = float((xmax - xmin + 1) * (ymax - ymin + 1))
        compact = float(area) / bbox_area

        score = float(area) * compact
        if score > best_score:
            best_score = score
            best = mask
            best_bbox = bbox
            best_area = area
            best_compact = compact

    if best is None:
        return {"ok": False, "msg": "NO_PICK (no blob)", "snap_color": snap_color}

    # 6) distance transform center
    dist = cv2.distanceTransform(best.astype(np.uint8), cv2.DIST_L2, 5)
    v_roi, u_roi = np.unravel_index(int(dist.argmax()), dist.shape)
    max_dist = float(dist[v_roi, u_roi])

    if max_dist < D_MIN:
        return {"ok": False, "msg": f"TOO_THIN (max_dist={max_dist:.1f})", "snap_color": snap_color}

    u_img = x0 + int(u_roi)
    v_img = y0 + int(v_roi)

    # 7) world conversion (single point)
    Pw = coord.pixel_to_world(u_img, v_img, snap_depth_frame)
    if Pw is None:
        return {"ok": False, "msg": "XYZ_FAIL", "snap_color": snap_color}

    X, Y, Z = float(Pw[0]), float(Pw[1]), float(Pw[2])
    Z_approach = Z + Z_APPROACH_CM

    # bbox -> image coords
    xmin, ymin, xmax, ymax = best_bbox
    bbox_img = (x0 + xmin, y0 + ymin, x0 + xmax, y0 + ymax)

    return {
        "ok": True,
        "msg": "OK",
        "u_img": u_img,
        "v_img": v_img,
        "bbox_img": bbox_img,
        "xyz_approach": [X, Y, Z_approach],
        "debug": {
            "valid_ratio": valid_ratio,
            "z_top": float(z_top),
            "best_area": best_area,
            "best_compact": best_compact,
            "max_dist": max_dist,
        },
        "snap_color": snap_color,
    }


def render_result_view(res, roi_xyxy):
    """RESULT 창에 오버레이 그려서 반환"""
    img = res.get("snap_color", None)
    if img is None:
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.putText(canvas, "NO IMAGE", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        return canvas

    out = img.copy()

    # ROI
    x1, y1, x2, y2 = map(int, roi_xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not res.get("ok", False):
        cv2.putText(out, res.get("msg", "FAIL"), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        return out

    # bbox
    bx1, by1, bx2, by2 = res["bbox_img"]
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

    # pick point
    u, v = res["u_img"], res["v_img"]
    cv2.circle(out, (u, v), 8, (0, 0, 255), -1)

    # text
    X, Y, Zapp = res["xyz_approach"]
    dbg = res.get("debug", {})
    cv2.putText(out, f"uv=({u},{v})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(out, f"XYZ_approach=[{X:.1f},{Y:.1f},{Zapp:.1f}] cm", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(out,
                f"valid={dbg.get('valid_ratio', 0):.2f} area={dbg.get('best_area', 0)} comp={dbg.get('best_compact', 0):.2f} maxD={dbg.get('max_dist', 0):.1f}px",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out


# =========================================================
# RealSense loop (네 스타일: callbacks)
# =========================================================
def run(
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    on_save=None,
    on_reset=None,
    on_click=None,
    update_depth_frame=None,
    update_color_image=None,
    get_points=None,
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    align = rs.align(rs.stream.color)

    live_name = "LIVE"
    result_name = "RESULT"

    cv2.namedWindow(live_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(result_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(live_name, WIN_W, WIN_H)
    cv2.resizeWindow(result_name, WIN_W, WIN_H)

    if on_click is not None:
        cv2.setMouseCallback(live_name, on_click)

    print("SPACE: 계산 | r: 리셋 | esc/q: 종료")

    try:
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if update_depth_frame is not None:
                update_depth_frame(depth_frame)
            if update_color_image is not None:
                update_color_image(color_image)

            display = color_image
            display = _draw_roi_xyxy(display, ROI)

            if get_points is not None:
                pts = get_points()
                if pts:
                    display = _draw_points(display, pts)

            cv2.imshow(live_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r") and on_reset:
                on_reset()
                continue
            elif key == ord(" ") and on_save:
                on_save(pipeline, align)
                continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 종료")


# =========================================================
# App state + callbacks
# =========================================================
class App:
    def __init__(self):
        self.depth_frame = None
        self.color_image = None
        self.points = []
        self.coord = Coordinate()

    def update_depth_frame(self, df):
        self.depth_frame = df

    def update_color_image(self, img):
        self.color_image = img

    def get_points(self):
        return self.points

    def on_reset(self):
        self.points = []
        blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank, "RESET", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        cv2.imshow("RESULT", blank)
        print("RESET")

    def on_save(self, pipeline, align):
        res = compute_pick_snapshot(pipeline, align, ROI, self.coord)

        if res.get("ok", False):
            self.points = [(res["u_img"], res["v_img"])]
            print(f"SEND → {res['xyz_approach']}")
        else:
            self.points = []
            print(res.get("msg", "FAIL"))

        view = render_result_view(res, ROI)
        cv2.imshow("RESULT", view)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"CLICK: ({x},{y})")


if __name__ == "__main__":
    app = App()
    run(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        on_save=app.on_save,
        on_reset=app.on_reset,
        on_click=app.on_click,
        update_depth_frame=app.update_depth_frame,
        update_color_image=app.update_color_image,
        get_points=app.get_points,
    )
