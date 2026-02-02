# app_viewer.py
# RealSense UI runner: snapshot (median) -> pick_core.compute_pick_from_snapshot -> render

import cv2
import numpy as np
import pyrealsense2 as rs

from pick_core import PickConfig, Coordinate, compute_pick_from_snapshot, PickResult


# ==============================
# UI helpers
# ==============================
def draw_roi(img, roi_xyxy, color=(0, 255, 0), th=2):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
    return out


def render_result(color_img: np.ndarray, res: PickResult, cfg: PickConfig) -> np.ndarray:
    out = color_img.copy()

    # ROI
    out = draw_roi(out, cfg.ROI)

    if not res.ok:
        cv2.putText(out, res.msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out 

    # bbox
    if res.bbox_img is not None:
        x1, y1, x2, y2 = res.bbox_img
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # pick
    cv2.circle(out, (int(res.u_img), int(res.v_img)), 8, (0, 0, 255), -1)

    # yaw arrow (if we have u2,v2)
    if res.u2 is not None and res.v2 is not None and (res.u2, res.v2) != (res.u_img, res.v_img):
        cv2.arrowedLine(out, (res.u_img, res.v_img), (int(res.u2), int(res.v2)),
                        (255, 0, 0), 2, tipLength=0.2)

    # text lines (loop to reduce code)
    lines = []
    lines.append(f"uv=({res.u_img},{res.v_img})")
    if res.xyz_approach_cm is not None:
        X, Y, Z = res.xyz_approach_cm
        lines.append(f"XYZ_approach=[{X:.1f},{Y:.1f},{Z:.1f}] cm")

    if res.yaw_work_deg is None:
        lines.append("yaw_work=None")
    else:
        lines.append(f"yaw_work={res.yaw_work_deg:.1f} deg")

    dbg = res.debug or {}
    lines.append(f"valid={dbg.get('valid_ratio',0):.2f} top%={dbg.get('top_percent',0)} band={dbg.get('band_m',0)*100:.1f}cm")
    lines.append(f"area={dbg.get('best_area',0)} comp={dbg.get('best_compact',0):.2f} bestD={dbg.get('best_dmax',0):.1f}px chosenD={dbg.get('chosen_d',0):.1f}px")
    lines.append(f"keep={dbg.get('dt_keep_ratio',0):.2f} center_w={dbg.get('dt_center_weight',0):.2f} pca_ratio={dbg.get('pca_ratio',0):.2f}")

    y0 = 50
    for i, s in enumerate(lines):
        y = y0 + 35 * i
        cv2.putText(out, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out


# ==============================
# Snapshot acquisition
# ==============================
def get_snapshot_median(pipeline: rs.pipeline, align: rs.align, cfg: PickConfig, snap_n: int = 3):
    """
    Returns:
        color_img (H,W,3) uint8 BGR
        depth_snap_m (H,W) float32 meters (median over snap_n)
    """
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

        depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * 0.001  # mm -> m
        depths.append(depth_m)

    if snap_color is None or len(depths) < max(1, snap_n // 2):
        return None, None

    depth_snap_m = np.median(np.stack(depths, axis=0), axis=0).astype(np.float32)
    return snap_color, depth_snap_m


# ==============================
# Main loop
# ==============================
def main():
    cfg = PickConfig()
    coord = Coordinate(cfg)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, cfg.WIDTH, cfg.HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, cfg.WIDTH, cfg.HEIGHT, rs.format.z16, 30)

    align = rs.align(rs.stream.color)

    cv2.namedWindow("LIVE", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RESULT", cv2.WINDOW_AUTOSIZE)

    print("SPACE: snapshot+compute | r: reset | esc/q: quit")

    last_result = None

    try:
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            live = draw_roi(color_img, cfg.ROI)

            # show last pick point on LIVE (optional)
            if last_result is not None and last_result.ok:
                cv2.circle(live, (last_result.u_img, last_result.v_img), 6, (0, 0, 255), -1)

            cv2.imshow("LIVE", live)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("r"):
                last_result = None
                blank = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, "RESET", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                cv2.imshow("RESULT", blank)
                print("RESET")

            if key == ord(" "):
                snap_color, depth_snap_m = get_snapshot_median(pipeline, align, cfg, snap_n=3)
                if snap_color is None:
                    last_result = PickResult(False, "SENSOR_BAD (no frames)")
                    cv2.imshow("RESULT", render_result(color_img, last_result, cfg))
                    print(last_result.msg)
                    continue

                res = compute_pick_from_snapshot(depth_snap_m, cfg, coord)
                last_result = res

                # "SEND" print
                if res.ok and res.xyz_approach_cm is not None:
                    xyz = res.xyz_approach_cm
                    if res.yaw_work_deg is not None:
                        print(f"SEND → xyz={list(xyz)} yaw_deg={res.yaw_work_deg:.1f}")
                    else:
                        print(f"SEND → xyz={list(xyz)} yaw_deg=None")
                else:
                    print(res.msg)

                view = render_result(snap_color, res, cfg)
                cv2.imshow("RESULT", view)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 종료")


if __name__ == "__main__":
    main()
