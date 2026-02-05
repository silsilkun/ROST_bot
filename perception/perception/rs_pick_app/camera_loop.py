import os
import cv2
import numpy as np
import pyrealsense2 as rs

from snapshot import snapshot_median_depth
from render import render_live_view, render_result_view


# =========================
# Runtime params
# =========================
WIDTH, HEIGHT, FPS = 1280, 720, 30
WIN_LIVE = "LIVE"
WIN_RESULT = "RESULT"
ROI_XYXY = (485, 135, 800, 350)

WARMUP_FIRST_SPACE_N = 20
CM_TO_MM = 10.0


class CameraLoop:
    def __init__(self, events, pipeline):
        self.events = events
        self.pipeline = pipeline

        self.rs_pipe = rs.pipeline()
        self.rs_cfg = rs.config()
        self.rs_cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        self.rs_cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        self.align = rs.align(rs.stream.color)

        self.latest_depth_m = None
        self._warmed_once = False

        # {"ok": bool, "color_path": str|None, "flat_clicked_xy": list|None, "xyz_angle": list|None}
        # flat_clicked_xy: [[X_mm, Y_mm], ...]
        # xyz_angle: [X_mm, Y_mm, Zapp_mm, yaw_deg]
        self.last_payload = None

    def get_frame(self):
        frames = self.rs_pipe.wait_for_frames()
        frames = self.align.process(frames)

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return None

        color = np.asanyarray(cf.get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * 0.001
        return color, depth_m

    def warmup_once_if_needed(self):
        if self._warmed_once:
            return
        for _ in range(WARMUP_FIRST_SPACE_N):
            _ = self.get_frame()
        self._warmed_once = True

    def run(self):
        cv2.namedWindow(WIN_LIVE, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            self.events.on_mouse(event, x, y, self.latest_depth_m)

        cv2.setMouseCallback(WIN_LIVE, on_mouse)

        started = False
        try:
            self.rs_pipe.start(self.rs_cfg)
            started = True
            print("LEFT CLICK: point | SPACE: detect | r: clear | q/ESC: quit")

            while True:
                out = self.get_frame()
                if out is None:
                    continue

                color, depth_m = out
                self.latest_depth_m = depth_m

                live = render_live_view(color_img=color, roi_xyxy=ROI_XYXY, clicked_uv=self.events.clicked_uv)
                cv2.imshow(WIN_LIVE, live)

                key = cv2.waitKey(1) & 0xFF
                cmd = self.events.on_key(key)

                if cmd["quit"]:
                    break

                if cmd["reset"]:
                    print("RESET: clicked points cleared")

                if cmd["do_space"]:
                    self.warmup_once_if_needed()

                    snap = snapshot_median_depth(get_frame=self.get_frame)
                    if snap is None:
                        print("FAIL")
                        self.last_payload = {
                            "ok": False,
                            "color_path": None,
                            "flat_clicked_xy": None,
                            "xyz_angle": None,
                        }
                        fail = render_result_view(
                            color_img=color,
                            roi_xyxy=ROI_XYXY,
                            ok=False, u=None, v=None, xyz_angle=None, u2v2=None
                        )
                        cv2.imshow(WIN_RESULT, fail)
                        continue

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    out_dir = os.path.join(base_dir, "result")
                    os.makedirs(out_dir, exist_ok=True)
                    out_color = os.path.join(out_dir, "color.jpg")
                    cv2.imwrite(out_color, snap.color_bgr)

                    result = self.pipeline.run(snapshot=snap, clicked_uv=self.events.clicked_uv, roi_xyxy=ROI_XYXY)

                    # xyz_angle: cm -> mm
                    xyz = result.get("xyz_angle")  # (X_cm, Y_cm, Zapp_cm, yaw_deg) or None
                    xyz_mm = None
                    if xyz is not None:
                        X_cm, Y_cm, Zapp_cm, yaw_deg = xyz
                        xyz_mm = [X_cm * CM_TO_MM, Y_cm * CM_TO_MM, Zapp_cm * CM_TO_MM, yaw_deg]

                    # flat_clicked_xy: cm -> mm
                    flat_cm = result.get("flat_clicked_xy")  # [[x_cm, y_cm], ...] or None
                    flat_mm = None
                    if flat_cm is not None:
                        flat_mm = [[x * CM_TO_MM, y * CM_TO_MM] for x, y in flat_cm]
                    # ros용 저장 이곳을 보고 가져가시오
                    self.last_payload = {
                        "ok": bool(result.get("ok", False)),
                        "color_path": out_color,
                        "flat_clicked_xy": flat_mm,
                        "xyz_angle": xyz_mm,
                    }

                    # prints (mm 기준)
                    if self.last_payload["flat_clicked_xy"] is not None:
                        print(f"flat_clicked_xy_mm={self.last_payload['flat_clicked_xy']}")

                    if self.last_payload["ok"] and self.last_payload["xyz_angle"] is not None:
                        print(f"xyz_angle_mm={self.last_payload['xyz_angle']}")
                    else:
                        print("FAIL")

                    view = render_result_view(
                        color_img=snap.color_bgr,
                        roi_xyxy=ROI_XYXY,
                        ok=result["ok"],
                        u=result["u"],
                        v=result["v"],
                        xyz_angle=result["xyz_angle"],  # UI 표시는 기존 cm 유지(원하면 render쪽도 바꿀 수 있음)
                        u2v2=result["u2v2"],
                    )
                    cv2.imshow(WIN_RESULT, view)

        finally:
            if started:
                self.rs_pipe.stop()
            cv2.destroyAllWindows()
            print("RealSense 종료")
